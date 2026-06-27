pub mod config;
pub mod auth;
pub mod translate_request;
pub mod translate_response;
pub mod logger;

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json,
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::time::{interval, Duration};

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<config::Config>,
    pub token_mgr: Arc<auth::TokenManager>,
    pub http_client: reqwest::Client,
}

#[tokio::main]
async fn main() {
    let config_path = std::env::var("CONFIG_PATH")
        .unwrap_or_else(|_| "config.yaml".to_string());

    let cfg = match config::Config::load(&config_path) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Failed to load config from {}: {}", config_path, e);
            std::process::exit(1);
        }
    };

    // Initialize logger
    let to_stderr = !cfg.logging_to_file || cfg.debug;
    logger::init(&cfg.log_dir, cfg.logs_max_total_size_mb, to_stderr, &cfg.log_level);

    if cfg.ericai_provider().is_none() {
        logger::error("main", "No 'ericai' provider found in config");
        std::process::exit(1);
    }

    let token_mgr = Arc::new(auth::TokenManager::new(&cfg.auth_dir));
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .pool_max_idle_per_host(128)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .build()
        .expect("Failed to create HTTP client");

    let state = AppState {
        cfg: cfg.clone(),
        token_mgr,
        http_client,
    };

    let app = axum::Router::new()
        .route("/v1/models", get(handle_models))
        .route("/v1/responses", post(handle_responses).get(handle_responses_ws))
        .route("/v1/chat/completions", post(handle_chat_completions))
        // Codex can send very large request bodies (up to ~30MB for long
        // conversation contexts with 196K+ tokens). Axum's default body
        // limit is 2MB, which causes "413 Payload Too Large" errors that
        // manifest as "stream disconnected before completion" in codex.
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50 MB
        // Fallback routes for codex's hardcoded ChatGPT-style base URL
        // (https://chatgpt.com/backend-api/codex). Only used if someone
        // configures base_url as http://localhost:8317/backend-api/codex
        // instead of the standard /v1. With base_url=/v1 (default config),
        // these routes are never hit — codex calls /v1/responses instead.
        .route("/backend-api/codex/responses", post(handle_responses).get(handle_responses_ws))
        .route("/backend-api/codex/responses/compact", post(handle_compact))
        .route("/v1/responses/compact", post(handle_compact))
        .route("/health", get(handle_health))
        .with_state(state);

    let addr = format!("{}:{}", cfg.host, cfg.port);
    logger::info("main", &format!("ericai-proxy listening on http://{}", addr));

    // Bind with retry — give the old process time to fully release the port
    let listener = {
        let mut bound: Option<tokio::net::TcpListener> = None;
        for attempt in 1..=5u32 {
            match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => { bound = Some(l); break; }
                Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                    logger::warn("main", &format!(
                        "Port {} in use (attempt {}/5), retrying in 1s…", cfg.port, attempt
                    ));
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
                Err(e) => {
                    eprintln!("Failed to bind to {}: {}", addr, e);
                    std::process::exit(1);
                }
            }
        }
        match bound {
            Some(l) => l,
            None => {
                eprintln!("Failed to bind to {} after 5 retries: port still in use.", addr);
                eprintln!("Run: ./start.sh --restart   or   fuser -k {}/tcp", cfg.port);
                std::process::exit(1);
            }
        }
    };
    axum::serve(listener, app).await.expect("Server error");
}

// ─── GET /v1/models ─────────────────────────────────────────────────────────
pub async fn handle_models(State(state): State<AppState>) -> impl IntoResponse {
    let models: Vec<Value> = state.cfg.all_models().iter().map(|(alias, name)| {
        json!({
            "id": alias,
            "object": "model",
            "created": 1700000000,
            "owned_by": "ericai",
            "ericai_backend_id": name,
        })
    }).collect();

    Json(json!({"object": "list", "data": models}))
}

// ─── GET /health ────────────────────────────────────────────────────────────
pub async fn handle_health() -> impl IntoResponse {
    Json(json!({"status": "ok", "service": "ericai-proxy"}))
}

// ─── POST /v1/responses ────────────────────────────────────────────────────
pub async fn handle_responses(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let request_json: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            logger::error("responses", &format!("Invalid JSON: {}", e));
            return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e))
                .into_response();
        }
    };

    let model_alias = request_json.get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

   // Resolve alias → backend model ID
    let backend_model = state.cfg.resolve_model(&model_alias);

   let is_stream = request_json.get("stream")
       .and_then(|v| v.as_bool())
       .unwrap_or(true);

   // Translate request
   let mut cc_payload = translate_request::responses_to_chat_completions(
       &backend_model,
       &request_json,
       is_stream,
   );
   cc_payload = translate_request::inject_thinking(cc_payload, &backend_model);

   // Context size guard: estimate tokens and warn if exceeds model's context window
   let context_window = {
       let catalog_path = std::env::var("HOME")
           .map(|h| format!("{}/.codex/model_catalog.json", h))
           .ok();
       catalog_path
           .and_then(|p| std::fs::read_to_string(&p).ok())
           .and_then(|s| serde_json::from_str::<Value>(&s).ok())
           .and_then(|c| c.get("models").and_then(|m| m.as_array()).map(|a| a.to_vec()))
           .and_then(|arr| {
               arr.iter()
                   .find(|m| m.get("slug").and_then(|s| s.as_str()) == Some(&model_alias))
                   .and_then(|m| m.get("context_window")).and_then(|c| c.as_u64())
           })
           .unwrap_or(196608u64)
   };
   let estimated_tokens = cc_payload.get("messages")
       .and_then(|m| m.as_array())
       .map(|arr| {
           arr.iter().map(|msg| {
               msg.get("content").and_then(|c| c.as_str()).unwrap_or("").len() / 4
           }).sum::<usize>()
       }).unwrap_or(0usize);
   if estimated_tokens as u64 > context_window {
       // Progressive trimming: remove older middle messages to fit context window
       // Keep: first message (system/instructions), last N messages (recent context)
       // Target: 80% of context window to leave room for output
       let target_tokens = (context_window as f64 * 0.8) as usize;
       logger::warn("responses", &format!(
           "Context size guard: estimated {} tokens > context_window {} for model {} — trimming",
           estimated_tokens, context_window, model_alias
       ));

       if let Some(messages) = cc_payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
           let total = messages.len();
           if total > 4 {
               // Keep first 2 (system + first user) and progressively keep more from the end
               // until we fit under target_tokens
               let mut best_keep = 4; // minimum: first 2 + last 2
               for keep_from_end in (2..=total-2).rev() {
                   let trimmed: Vec<&Value> = messages.iter().take(2)
                       .chain(messages.iter().skip(total - keep_from_end))
                       .collect();
                   let trimmed_tokens: usize = trimmed.iter().map(|msg| {
                       msg.get("content").and_then(|c| c.as_str()).unwrap_or("").len() / 4
                   }).sum();
                   if trimmed_tokens <= target_tokens {
                       best_keep = keep_from_end;
                       break;
                   }
               }

               // Build the trimmed messages array
               let head: Vec<Value> = messages.iter().take(2).cloned().collect();
               let tail: Vec<Value> = messages.iter().skip(total - best_keep).cloned().collect();

               // Insert a compaction notice where we cut
               let mut new_messages = head;
               new_messages.push(json!({
                   "role": "system",
                   "content": format!(
                       "[Context trimmed: {} older messages removed to fit context window. {} of {} messages retained.]",
                       total - 2 - best_keep, best_keep + 2, total
                   )
               }));
               new_messages.extend(tail);

               let new_tokens: usize = new_messages.iter().map(|msg| {
                   msg.get("content").and_then(|c| c.as_str()).unwrap_or("").len() / 4
               }).sum();

               logger::info("responses", &format!(
                   "Trimmed: {} messages → {} messages, {} → {} tokens",
                   total, new_messages.len(), estimated_tokens, new_tokens
               ));

               *messages = new_messages;
           }
       }

       // Re-check: if still too large after trimming, return error
       let new_estimated = cc_payload.get("messages")
           .and_then(|m| m.as_array())
           .map(|arr| {
               arr.iter().map(|msg| {
                   msg.get("content").and_then(|c| c.as_str()).unwrap_or("").len() / 4
               }).sum::<usize>()
           }).unwrap_or(0usize);
       if new_estimated as u64 > context_window {
           logger::error("responses", &format!(
               "Still too large after trimming: {} tokens — returning error",
               new_estimated
           ));
           let err_response = json!({
                "error": {
                    "message": format!(
                        "Context too large: even after trimming, {} tokens exceeds model {} context window ({}). Start a new session.",
                        new_estimated, model_alias, context_window
                    ),
                    "type": "context_length_exceeded",
                }
            });
            return (StatusCode::PAYLOAD_TOO_LARGE, Json(err_response)).into_response();
        }
    }

    // Get auth token
    let token = match state.token_mgr.get_token().await {
        Ok(t) => t,
        Err(e) => {
            logger::error("auth", &format!("Token error: {}", e));
            return (StatusCode::UNAUTHORIZED, format!("Auth failed: {}", e))
                .into_response();
        }
    };

    let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
    let url = format!("{}/chat/completions", base_url);

    logger::info("responses", &format!(
        "POST {} model={} stream={}", url, backend_model, is_stream
    ));

    let resp = match state.http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Accept", "text/event-stream")
        .json(&cc_payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            logger::error("responses", &format!("Upstream error: {}", e));
            return (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e))
                .into_response();
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        logger::error("responses", &format!("Upstream returned {}: {}", status, logger::extract_error_summary(&body_text, 300)));
        logger::error_request("v1-responses", &cc_payload.to_string(), &body_text, status.as_u16());
        return (status, body_text).into_response();
    }
    if is_stream {
        let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
        let mut translator = translate_response::ResponseTranslator::new(
            &response_id,
            Some(request_json.clone()),
        );

        let byte_stream = resp.bytes_stream();

        // Use a channel so we can detect client disconnects.  When the
        // client (codex) drops the connection — e.g. when the user clicks
        // Stop — Axum stops polling the response body.  The channel sender
        // then fails on the next `send()`, and we abort the upstream stream.
        use tokio::sync::mpsc;
        let (tx, rx) = mpsc::channel::<Result<axum::body::Bytes, std::convert::Infallible>>(32);

        let cancel_token = tokio_util::sync::CancellationToken::new();
        let upstream_cancel = cancel_token.clone();

        // Spawn the upstream reader task
        tokio::spawn(async move {
            let mut stream = byte_stream;
            let mut buffer = String::new();
            // SSE keepalive: send a comment line every 15 seconds while
            // waiting for upstream data.  This prevents the HTTP connection
            // from timing out when EricAI takes a long time to process
            // large resumed session contexts (60+ seconds).
            let mut sse_keepalive = tokio::time::interval(Duration::from_secs(15));
            sse_keepalive.tick().await; // consume immediate first tick

            loop {
                tokio::select! {
                    // Poll upstream for next chunk
                    chunk_result = stream.next() => {
                        match chunk_result {
                            Some(Ok(chunk)) => {
                               buffer.push_str(&String::from_utf8_lossy(&chunk));
                               while let Some(pos) = buffer.find('\n') {
                                    let events = translator.process_line(&buffer[..pos]);
                                    buffer.drain(..pos + 1);
                                   for event in events {
                                       if tx.send(Ok(axum::body::Bytes::from(event.into_bytes()))).await.is_err() {
                                           // Client disconnected — stop reading upstream
                                           upstream_cancel.cancel();
                                            return;
                                        }
                                    }
                                }
                            }
                            Some(Err(e)) => {
                                logger::error("responses", &format!("Stream error: {}", e));
                                break;
                            }
                            None => break, // upstream finished
                        }
                    }
                    // SSE keepalive: send comment to keep connection alive
                    _ = sse_keepalive.tick() => {
                        // SSE comments start with ':' and are ignored by clients
                        // but keep the TCP connection alive
                        if tx.send(Ok(axum::body::Bytes::from(": keepalive\n\n".to_string()))).await.is_err() {
                            return;
                        }
                    }
                    // Client disconnect detected via cancellation
                    _ = upstream_cancel.cancelled() => {
                        logger::info("responses", "Client disconnected, aborting upstream stream");
                        return;
                    }
                }
            }

            // Flush remaining buffer
            if !buffer.trim().is_empty() {
                let events = translator.process_line(&buffer);
                for event in events {
                    if tx.send(Ok(axum::body::Bytes::from(event.into_bytes()))).await.is_err() {
                        return;
                    }
                }
            }

            // Send [DONE]
            let events = translator.process_line("data: [DONE]");
            for event in events {
                if tx.send(Ok(axum::body::Bytes::from(event.into_bytes()))).await.is_err() {
                    return;
                }
            }
        });

        // Convert the receiver into a stream for the response body.
        // When the client disconnects, Axum drops the Body, which drops rx,
        // which causes the next tx.send() to fail → upstream reader exits.
        let translated_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let body = Body::from_stream(translated_stream);

        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("connection", "keep-alive")
            .body(body)
            .unwrap()
    } else {
        let full_body = match resp.json::<Value>().await {
            Ok(v) => v,
            Err(e) => {
                logger::error("responses", &format!("Parse error: {}", e));
                return (StatusCode::BAD_GATEWAY, format!("Parse error: {}", e))
                    .into_response();
            }
        };

        let response = translate_response::chat_completion_to_response(
            &full_body,
            Some(&request_json),
        );

        Json(response).into_response()
    }
}

// ─── GET /v1/responses (WebSocket) ──────────────────────────────────────────
pub async fn handle_responses_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(_params): Query<Value>,
) -> Response {
    ws.on_upgrade(move |mut socket| async move {
        use futures_util::StreamExt;
        logger::info("ws", "WebSocket connection opened for /v1/responses");

        // ── Connection-reuse state ─────────────────────────────────────
        // Codex reuses a single WebSocket connection across multiple turns.
        // When codex sends an incremental request (previous_response_id set),
        // it only includes the NEW input items — not the full conversation.
        // We must cache the full input from each response so we can merge
        // the delta with the cached baseline before translating to Chat
        // Completions format.
        //
        // Key: response_id  →  Value: full input array from that request
        let mut response_input_cache: std::collections::HashMap<String, Vec<Value>> =
            std::collections::HashMap::new();

        // ── Connection-reuse loop ──────────────────────────────────────
        // Codex opens ONE WebSocket connection and sends multiple turns
        // over it (prewarm + first turn + subsequent turns).  After each
        // turn completes we loop back to recv() the next request instead
        // of closing the socket.  This eliminates the constant open/close
        // churn (1-4 s per connection) that causes "stream disconnected"
        // retries in codex and wastes resources under load.
        //
        // Break conditions: socket error, client close frame, or idle
        // timeout (300 s between turns — codex may take minutes between
        // turns while the user reviews output or runs tool calls).
        'connection: loop {
            // Wait for the next request message.
            // Prewarm connections may not send data immediately; between
            // turns codex may take a while.  Use a generous 5-minute idle
            // timeout.
            let msg = match tokio::time::timeout(
                Duration::from_secs(300),
                socket.recv(),
            ).await {
                Ok(Some(Ok(m))) => m,
                Ok(Some(Err(_))) | Ok(None) => {
                    logger::info("ws", "WebSocket closed by client");
                    break 'connection;
                }
                Err(_) => {
                    logger::info("ws", "WebSocket idle timeout (300s between turns), closing");
                    let _ = socket.close().await;
                    break 'connection;
                }
            };

            // Handle protocol-level control frames
            let text = match msg {
                axum::extract::ws::Message::Text(t) => t,
                axum::extract::ws::Message::Binary(b) => String::from_utf8_lossy(&b).to_string(),
                axum::extract::ws::Message::Ping(p) => {
                    let _ = socket.send(axum::extract::ws::Message::Pong(p)).await;
                    continue 'connection;
                }
                axum::extract::ws::Message::Pong(_) => continue 'connection,
                axum::extract::ws::Message::Close(_) => {
                    logger::info("ws", "WebSocket close frame received");
                    break 'connection;
                }
            };

            let request_json: Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(e) => {
                    logger::error("ws", &format!("Invalid JSON: {}", e));
                    let _ = socket.send(axum::extract::ws::Message::Text(
                        format!("{{\"error\":\"Invalid JSON: {}\"}}", e)
                    )).await;
                    continue 'connection; // keep connection alive for retry
                }
            };

            // Prewarm: generate=false means connection setup only, no
            // inference.  Codex's prewarm_websocket() waits for
            // response.completed.  Acknowledge immediately without
            // calling upstream — this keeps the connection warm.
            let is_prewarm = request_json.get("generate")
                .and_then(|v| v.as_bool())
                .map(|g| !g)
                .unwrap_or(false);

            if is_prewarm {
                // Prewarm: codex sends generate=false to establish the connection.
                // We must return response.completed with an EMPTY response_id ("").
                // Codex stores this id and uses it as previous_response_id for the
                // next request. With an empty id, codex's prepare_websocket_request()
                // will see `last_response.response_id.is_empty()` and fall back to
                // sending the full input (ResponseCreate without previous_response_id).
                // This ensures the first real request includes the complete conversation.
                logger::info("ws", "Prewarm request (generate=false), acknowledging with empty response_id");
                let _ = socket.send(axum::extract::ws::Message::Text(
                    r#"{"type":"response.completed","response":{"id":"","status":"completed"}}"#.to_string()
                )).await;
                continue 'connection;
            }

            let model_alias = request_json.get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
               .to_string();

            let backend_model = state.cfg.resolve_model(&model_alias);

           let is_stream = request_json.get("stream")
               .and_then(|v| v.as_bool())
               .unwrap_or(true);

            // ── previous_response_id handling ──────────────────────────
            // When codex sends an incremental request (previous_response_id set),
            // the input array contains ONLY the new items since the last response.
            // We need to merge the cached full input with the new delta items to
            // reconstruct the complete conversation for the Chat Completions API.
            let mut effective_request = request_json.clone();
            let response_id_for_cache = format!("resp_{}", uuid::Uuid::new_v4().simple());

            if let Some(prev_id) = request_json.get("previous_response_id").and_then(|v| v.as_str()) {
                if !prev_id.is_empty() {
                    if let Some(cached_input) = response_input_cache.get(prev_id) {
                        // Merge: cached full input + new delta items from this request
                        let new_items = request_json.get("input")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();

                        let mut merged_input = cached_input.clone();
                        merged_input.extend(new_items);

                        effective_request["input"] = json!(merged_input);
                        logger::info("ws", &format!(
                            "Incremental request: previous_response_id={}, merged {} cached + {} new = {} total items",
                            prev_id, cached_input.len(),
                            request_json.get("input").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
                            merged_input.len()
                        ));
                    } else {
                        logger::warn("ws", &format!(
                            "previous_response_id={} not found in cache — using full input as-is",
                            prev_id
                        ));
                    }
                }
            }

            // Cache this request's full input under the new response_id for future incremental requests
            if let Some(full_input) = effective_request.get("input").and_then(|v| v.as_array()) {
                response_input_cache.insert(response_id_for_cache.clone(), full_input.clone());
                // Limit cache size to prevent unbounded memory growth
                if response_input_cache.len() > 10 {
                    // Remove oldest entries (HashMap has no order, so just remove arbitrary)
                    let key_to_remove = response_input_cache.keys().next().cloned();
                    if let Some(key) = key_to_remove {
                        response_input_cache.remove(&key);
                    }
                }
            }

            let mut cc_payload = translate_request::responses_to_chat_completions(
                &backend_model,
                &effective_request,
                is_stream,
            );
            cc_payload = translate_request::inject_thinking(cc_payload, &backend_model);

            // Context size guard: estimate tokens and warn if exceeds model's context window
            let context_window = {
                let catalog_path = std::env::var("HOME")
                    .map(|h| format!("{}/.codex/model_catalog.json", h))
                    .ok();
                catalog_path
                    .and_then(|p| std::fs::read_to_string(&p).ok())
                    .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                    .and_then(|c| c.get("models").and_then(|m| m.as_array()).map(|a| a.to_vec()))
                    .and_then(|arr| {
                        arr.iter()
                            .find(|m| m.get("slug").and_then(|s| s.as_str()) == Some(&model_alias))
                            .and_then(|m| m.get("context_window")).and_then(|c| c.as_u64())
                    })
                    .unwrap_or(196608u64)
            };
            let estimated_tokens = cc_payload.get("messages")
                .and_then(|m| m.as_array())
                .map(|arr| {
                    arr.iter().map(|msg| {
                        msg.get("content").and_then(|c| c.as_str()).unwrap_or("").len() / 4
                    }).sum::<usize>()
                }).unwrap_or(0usize);
            if estimated_tokens as u64 > context_window {
                let target_tokens = (context_window as f64 * 0.8) as usize;
                logger::warn("ws", &format!(
                    "Context size guard: estimated {} tokens > context_window {} for model {} — trimming",
                    estimated_tokens, context_window, model_alias
                ));

                if let Some(messages) = cc_payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
                    let total = messages.len();
                    if total > 4 {
                        let mut best_keep = 4;
                        for keep_from_end in (2..=total-2).rev() {
                            let trimmed: Vec<&Value> = messages.iter().take(2)
                                .chain(messages.iter().skip(total - keep_from_end))
                                .collect();
                            let trimmed_tokens: usize = trimmed.iter().map(|msg| {
                                msg.get("content").and_then(|c| c.as_str()).unwrap_or("").len() / 4
                            }).sum();
                            if trimmed_tokens <= target_tokens {
                                best_keep = keep_from_end;
                                break;
                            }
                        }

                        let head: Vec<Value> = messages.iter().take(2).cloned().collect();
                        let tail: Vec<Value> = messages.iter().skip(total - best_keep).cloned().collect();

                        let mut new_messages = head;
                        new_messages.push(json!({
                            "role": "system",
                            "content": format!(
                                "[Context trimmed: {} older messages removed to fit context window. {} of {} messages retained.]",
                                total - 2 - best_keep, best_keep + 2, total
                            )
                        }));
                        new_messages.extend(tail);

                        let new_tokens: usize = new_messages.iter().map(|msg| {
                            msg.get("content").and_then(|c| c.as_str()).unwrap_or("").len() / 4
                        }).sum();

                        logger::info("ws", &format!(
                            "Trimmed: {} messages → {} messages, {} → {} tokens",
                            total, new_messages.len(), estimated_tokens, new_tokens
                        ));

                        *messages = new_messages;
                    }
                }

                // Re-check: if still too large after trimming, send error via WebSocket
                let new_estimated = cc_payload.get("messages")
                    .and_then(|m| m.as_array())
                    .map(|arr| {
                        arr.iter().map(|msg| {
                            msg.get("content").and_then(|c| c.as_str()).unwrap_or("").len() / 4
                        }).sum::<usize>()
                    }).unwrap_or(0usize);
                if new_estimated as u64 > context_window {
                    logger::error("ws", &format!(
                        "Still too large after trimming: {} tokens — sending error",
                        new_estimated
                    ));
                    let err_msg = format!(
                        "{{\"error\":\"Context too large: even after trimming, {} tokens exceeds model {} context window ({}). Start a new session.\"}}",
                        new_estimated, model_alias, context_window
                    );
                    let _ = socket.send(axum::extract::ws::Message::Text(err_msg)).await;
                    continue 'connection; // keep connection alive
                }
            }


            let token = match state.token_mgr.get_token().await {
                Ok(t) => t,
                Err(e) => {
                    let _ = socket.send(axum::extract::ws::Message::Text(
                        format!("{{\"error\":\"Auth failed: {}\"}}", e)
                    )).await;
                    continue 'connection; // keep connection alive for retry
                }
            };

            let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
            let url = format!("{}/chat/completions", base_url);

            logger::info("ws", &format!("WS POST {} model={} stream={}", url, backend_model, is_stream));

            // Send the request to EricAI.  While waiting for the upstream
            // response headers (which can take 60+ seconds for large resumed
            // sessions), send WebSocket pings every 15 seconds to keep the
            // connection alive and prevent "Connection reset by peer".
            let send_fut = state.http_client
                .post(&url)
                .header("Authorization", format!("Bearer {}", token))
                .header("Accept", "text/event-stream")
                .json(&cc_payload)
                .send();

            let mut ping_during_connect = interval(Duration::from_secs(15));
            ping_during_connect.tick().await; // consume immediate first tick

            let mut send_fut = Box::pin(send_fut);
            let resp = loop {
                tokio::select! {
                    result = &mut send_fut => {
                        match result {
                            Ok(r) => break r,
                            Err(e) => {
                                let _ = socket.send(axum::extract::ws::Message::Text(
                                    format!("{{\"error\":\"Upstream: {}\"}}", e)
                                )).await;
                                continue 'connection; // keep connection alive for retry
                            }
                        }
                    }
                    _ = ping_during_connect.tick() => {
                        if socket.send(axum::extract::ws::Message::Ping(vec![1, 2, 3, 4])).await.is_err() {
                            break 'connection; // socket dead
                        }
                    }
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let body_text = resp.text().await.unwrap_or_default();
                logger::error("ws", &format!("Upstream {}: {}", status, logger::extract_error_summary(&body_text, 300)));
                logger::error_request("ws-responses", &cc_payload.to_string(), &body_text, status.as_u16());
                // Send error in the format codex's parse_wrapped_websocket_error_event expects:
                //   {"type":"error","status":<code>,"error":{"message":"..."}}
                // This lets codex map it to ApiError::Transport(Http) and trigger proper retry/fallback.
                let err_payload = json!({
                    "type": "error",
                    "status": status.as_u16(),
                    "error": {
                        "message": format!("Upstream {}: {}", status, logger::extract_error_summary(&body_text, 500))
                    }
                });
                let _ = socket.send(axum::extract::ws::Message::Text(
                    err_payload.to_string()
                )).await;
                continue 'connection; // keep connection alive for retry
            }

            // Use the response_id we already generated and cached
            let mut translator = translate_response::ResponseTranslator::new_raw_json(
               &response_id_for_cache,
               Some(effective_request.clone()),
           );

            let byte_stream = resp.bytes_stream();
            let mut stream = byte_stream;
            let mut buffer = String::new();

            // Track whether the upstream produced content and whether it errored.
            // If the stream errors or ends without data, we must NOT send
            // response.completed — codex treats that as success, leading to
            // degenerate model output (empty turns, pwd/ls commands, etc.).
            let mut stream_event_count: usize = 0;
            let mut stream_errored = false;

            // Keepalive: send WebSocket Ping every 15 seconds to prevent idle timeout
            // when the upstream (EricAI) takes a long time to process large contexts
            // (e.g. session resume with full conversation history).
            let mut ping_interval = interval(Duration::from_secs(15));
            ping_interval.tick().await; // consume immediate first tick

            'stream: loop {
                tokio::select! {
                    // Upstream data arrives
                    chunk_result = stream.next() => {
                        match chunk_result {
                            Some(Ok(chunk)) => {
                                buffer.push_str(&String::from_utf8_lossy(&chunk));

                               while let Some(pos) = buffer.find('\n') {
                                    let events = translator.process_line(&buffer[..pos]);
                                    buffer.drain(..pos + 1);
                                   for event in events {
                                       stream_event_count += 1;
                                       // In raw JSON mode, process_line returns JSON objects directly.
                                       if socket.send(axum::extract::ws::Message::Text(event)).await.is_err() {
                                           break 'connection; // socket dead
                                       }
                                   }
                                }
                            }
                            Some(Err(e)) => {
                                logger::error("ws", &format!("WS stream error: {}", e));
                                stream_errored = true;
                                break 'stream;
                            }
                            None => {
                                // Upstream stream ended
                                break 'stream;
                            }
                        }
                    }
                    // Keepalive ping
                    _ = ping_interval.tick() => {
                        if socket.send(axum::extract::ws::Message::Ping(vec![1, 2, 3, 4])).await.is_err() {
                            break 'connection; // socket dead
                        }
                    }
                }
            }

            // If the upstream stream errored, send an error event — NOT response.completed.
            // Codex's parse_wrapped_websocket_error_event will map this to ApiError and
            // trigger proper retry/fallback logic.
            if stream_errored {
                let err_payload = json!({
                    "type": "error",
                    "status": 502,
                    "error": {
                        "message": "Upstream stream error — connection terminated before response completed"
                    }
                });
                let _ = socket.send(axum::extract::ws::Message::Text(
                    err_payload.to_string()
                )).await;
                continue 'connection;
            }

            // Flush remaining buffer
           if !buffer.trim().is_empty() {
               let events = translator.process_line(&buffer);
               for event in events {
                   stream_event_count += 1;
                    if socket.send(axum::extract::ws::Message::Text(event)).await.is_err() {
                        break 'connection; // socket dead
                    }
               }
           }

            // If the upstream produced no content events, send an error instead
            // of response.completed. An empty response.completed causes codex to
            // treat the turn as successful with no output, leading to degenerate
            // model behavior (pwd/ls commands, 30-token responses, etc.).
            if stream_event_count == 0 {
                logger::error("ws", "Upstream stream ended with no content events — sending error instead of response.completed");
                let err_payload = json!({
                    "type": "error",
                    "status": 502,
                    "error": {
                        "message": "Upstream returned empty stream — no content events received"
                    }
                });
                let _ = socket.send(axum::extract::ws::Message::Text(
                    err_payload.to_string()
                )).await;
                continue 'connection;
            }

            // Properly flush the translator by passing [DONE], which emits
            // the full response.completed event with output/usage data.
            // This replaces the previous manual response.completed that was
            // missing output, usage, model, and other fields.
            let done_events = translator.process_line("data: [DONE]");
            for event in done_events {
                if socket.send(axum::extract::ws::Message::Text(event)).await.is_err() {
                    break 'connection; // socket dead
                }
            }

            // Turn complete — loop back to recv() the next request on the
            // SAME connection.  No log spam; only log on actual close.
        }

        logger::info("ws", "WebSocket connection closed");
    })
}

// ─── POST /v1/chat/completions (passthrough) ───────────────────────────────
pub async fn handle_chat_completions(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let mut request_json: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            logger::error("chat", &format!("Invalid JSON: {}", e));
            return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e))
                .into_response();
        }
    };

    let model = request_json.get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
       .to_string();

    let backend_model = state.cfg.resolve_model(&model);

   // Resolve alias in the request
   request_json["model"] = json!(backend_model);

    let token = match state.token_mgr.get_token().await {
        Ok(t) => t,
        Err(e) => {
            logger::error("chat", &format!("Auth failed: {}", e));
            return (StatusCode::UNAUTHORIZED, format!("Auth failed: {}", e))
                .into_response();
        }
    };

    let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
    let url = format!("{}/chat/completions", base_url);

    logger::info("chat", &format!("Passthrough POST {} model={}", url, backend_model));

    let resp = state.http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Accept", "text/event-stream")
        .json(&request_json)
        .send()
        .await;

    match resp {
        Ok(r) => {
            let status = r.status();
            let headers = r.headers().clone();

            if request_json.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
                let stream = r.bytes_stream();
                let body = Body::from_stream(stream);
                let mut response = Response::builder()
                    .status(status)
                    .header("content-type", "text/event-stream");
                for (k, v) in &headers {
                    if k != "content-encoding" && k != "transfer-encoding" {
                        response = response.header(k, v);
                    }
                }
                response.body(body).unwrap()
            } else {
                let body = r.bytes().await.unwrap_or_default();
                let mut response = Response::builder().status(status);
                for (k, v) in &headers {
                    if k != "content-encoding" && k != "transfer-encoding" {
                        response = response.header(k, v);
                    }
                }
                response.body(Body::from(body)).unwrap()
            }
        }
        Err(e) => {
            logger::error("chat", &format!("Upstream error: {}", e));
            (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e)).into_response()
        }
    }
}

// ─── POST /backend-api/codex/responses/compact ─────────────────────────────
pub async fn handle_compact(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let mut request_json: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            logger::error("compact", &format!("Invalid JSON: {}", e));
            return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e))
                .into_response();
        }
    };

    // Remove stream field if present
    if let Some(obj) = request_json.as_object_mut() {
        obj.remove("stream");
    }

    let model_alias = request_json.get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
       .to_string();

    let backend_model = state.cfg.resolve_model(&model_alias);

   let mut cc_payload = translate_request::responses_to_chat_completions(
       &backend_model,
       &request_json,
       false,
    );
    cc_payload = translate_request::inject_thinking(cc_payload, &backend_model);

    let token = match state.token_mgr.get_token().await {
        Ok(t) => t,
        Err(e) => {
            logger::error("compact", &format!("Auth failed: {}", e));
            return (StatusCode::UNAUTHORIZED, format!("Auth failed: {}", e))
                .into_response();
        }
    };

    let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
    let url = format!("{}/chat/completions", base_url);

    // Context size guard: estimate input tokens and reject if exceeds model's context window
    let context_window = {
        let catalog_path = std::env::var("HOME")
            .map(|h| format!("{}/.codex/model_catalog.json", h))
            .ok();
        let cw = catalog_path
            .and_then(|p| std::fs::read_to_string(&p).ok())
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|c| c.get("models").and_then(|m| m.as_array()).map(|a| a.to_vec()))
            .and_then(|arr| {
                arr.iter()
                    .find(|m| m.get("slug").and_then(|s| s.as_str()) == Some(&model_alias))
                    .and_then(|m| m.get("context_window")).and_then(|c| c.as_u64())
            })
            .unwrap_or(196608u64);
        cw
    };

    // Rough estimate: ~4 chars per token
    let estimated_tokens = cc_payload.get("messages")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter().map(|msg| {
                let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                content.len() / 4
            }).sum::<usize>()
        }).unwrap_or(0);

    if estimated_tokens as u64 > context_window {
        // Trim older middle messages to fit, then compact the trimmed version
        let target_tokens = (context_window as f64 * 0.8) as usize;
        logger::warn("compact", &format!(
            "Context size guard: estimated {} tokens > context_window {} — trimming before compact",
            estimated_tokens, context_window
        ));

        if let Some(messages) = cc_payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
            let total = messages.len();
            if total > 4 {
                let mut best_keep = 4;
                for keep_from_end in (2..=total-2).rev() {
                    let trimmed: Vec<&Value> = messages.iter().take(2)
                        .chain(messages.iter().skip(total - keep_from_end))
                        .collect();
                    let trimmed_tokens: usize = trimmed.iter().map(|msg| {
                        msg.get("content").and_then(|c| c.as_str()).unwrap_or("").len() / 4
                    }).sum();
                    if trimmed_tokens <= target_tokens {
                        best_keep = keep_from_end;
                        break;
                    }
                }

                let head: Vec<Value> = messages.iter().take(2).cloned().collect();
                let tail: Vec<Value> = messages.iter().skip(total - best_keep).cloned().collect();
                let mut new_messages = head;
                new_messages.push(json!({
                    "role": "system",
                    "content": format!(
                        "[Context trimmed: {} older messages removed. Summarize the remaining conversation.]",
                        total - 2 - best_keep
                    )
                }));
                new_messages.extend(tail);

                logger::info("compact", &format!(
                    "Trimmed for compact: {} → {} messages",
                    total, new_messages.len()
                ));
                *messages = new_messages;
            }
        }

        // If still too large, return synthetic compaction
        let new_estimated = cc_payload.get("messages")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter().map(|msg| {
                    msg.get("content").and_then(|c| c.as_str()).unwrap_or("").len() / 4
                }).sum::<usize>()
            }).unwrap_or(0usize);
        if new_estimated as u64 > context_window {
            logger::error("compact", &format!(
                "Still too large after trimming: {} tokens — returning synthetic compact",
                new_estimated
            ));
            let compaction_item = json!({
                "type": "compaction",
                "summary_text": "Previous conversation context was too large to compact. Starting fresh.",
                "compaction_trigger": "auto",
            });
            let response_obj = json!({
                "id": format!("resp_compact_{}", uuid::Uuid::new_v4().simple()),
                "object": "response.compaction",
                "status": "completed",
                "model": model_alias,
                "output": [compaction_item],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            });
            return Json(response_obj).into_response();
        }
    }

    // Use streaming to avoid timeouts on large contexts
    cc_payload["stream"] = json!(true);
    cc_payload["stream_options"] = json!({"include_usage": true});

    logger::info("compact", &format!("Compact POST (streaming) {} model={} est_tokens={}", url, backend_model, estimated_tokens));

    let resp = match state.http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Accept", "text/event-stream")
        .json(&cc_payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            logger::error("compact", &format!("Upstream error: {}", e));
            return (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e))
                .into_response();
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        logger::error("compact", &format!("Upstream {}: {}", status, logger::extract_error_summary(&body_text, 300)));
        logger::error_request("compact", &cc_payload.to_string(), &body_text, status.as_u16());
        return (status, body_text).into_response();
    }

    // Collect streamed response into a single text
    use futures_util::StreamExt;
    let byte_stream = resp.bytes_stream();
    let mut stream = byte_stream;
    let mut buffer = String::new();
    let mut full_text = String::new();
    let mut finish_reason: Option<String> = None;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                buffer.push_str(&String::from_utf8_lossy(&chunk));
               while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].to_string();
                    buffer.drain(..pos + 1);
                    if let Some(data) = line.strip_prefix("data: ") {
                       if data.trim() == "[DONE]" { continue; }
                       if let Ok(chunk_json) = serde_json::from_str::<Value>(data) {
                            // Extract text delta
                            if let Some(choices) = chunk_json.get("choices").and_then(|c| c.as_array()) {
                                for choice in choices {
                                    if let Some(delta) = choice.get("delta") {
                                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                            full_text.push_str(content);
                                        }
                                    }
                                    if let Some(fr) = choice.get("finish_reason").and_then(|f| f.as_str()) {
                                        finish_reason = Some(fr.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                logger::error("compact", &format!("Stream error: {}", e));
                break;
            }
        }
    }

    logger::info("compact", &format!("Compact completed: {} chars of summary (finish={:?})", full_text.len(), finish_reason));

    let compaction_item = json!({
        "type": "compaction",
        "summary_text": if full_text.is_empty() { "Session compacted." } else { &full_text },
        "compaction_trigger": "auto",
    });

    let response_obj = json!({
        "id": format!("resp_compact_{}", uuid::Uuid::new_v4().simple()),
        "object": "response.compaction",
        "status": "completed",
        "model": model_alias,
        "output": [compaction_item],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    });

    Json(response_obj).into_response()
}
