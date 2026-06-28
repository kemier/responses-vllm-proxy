#[allow(clippy::duplicate_mod)]
pub mod config;
#[allow(clippy::duplicate_mod)]
pub mod auth;
#[allow(clippy::duplicate_mod)]
pub mod translate_request;
#[allow(clippy::duplicate_mod)]
pub mod translate_response;
#[allow(clippy::duplicate_mod)]
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

/// Cached model catalog loaded once at startup from ~/.codex/model_catalog.json.
/// Previously this file was read from disk on EVERY request (HTTP, WebSocket,
/// and compact), causing 13+ disk reads per WebSocket connection with
/// incremental requests. Now it's loaded once and shared via Arc.
pub type ModelCatalog = Arc<Option<Value>>;

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<config::Config>,
    pub token_mgr: Arc<auth::TokenManager>,
    pub http_client: reqwest::Client,
    pub model_catalog: ModelCatalog,
}

/// Pre-allocated SSE keepalive comment bytes — avoids allocating a new
/// String on every 15-second keepalive tick.
static KEEPALIVE_BYTES: axum::body::Bytes =
    axum::body::Bytes::from_static(b": keepalive\n\n");

/// Look up a model's context_window from the cached catalog.
/// Falls back to 196608 (192K) if the catalog isn't loaded or the model
/// isn't found.
pub fn get_context_window(catalog: &ModelCatalog, model_alias: &str) -> u64 {
    if let Some(c) = catalog.as_ref() {
        if let Some(arr) = c.get("models").and_then(|m| m.as_array()) {
            for m in arr {
                if m.get("slug").and_then(|s| s.as_str()) == Some(model_alias) {
                    if let Some(cw) = m.get("context_window").and_then(|c| c.as_u64()) {
                        return cw;
                    }
                }
            }
        }
    }
    196608
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

    // Load model catalog once at startup — avoids per-request disk reads.
    let model_catalog: ModelCatalog = Arc::new(
        std::env::var("HOME")
            .ok()
            .map(|h| format!("{}/.codex/model_catalog.json", h))
            .and_then(|p| std::fs::read_to_string(&p).ok())
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
    );
    if model_catalog.is_some() {
        logger::info("main", "Model catalog loaded from ~/.codex/model_catalog.json");
    } else {
        logger::warn("main", "Model catalog not found — using default context_window=196608");
    }

    let state = AppState {
        cfg: cfg.clone(),
        token_mgr,
        http_client,
        model_catalog,
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

    // Bind with retry — give the old process time to fully release the port.
    // Use SO_REUSEADDR via socket2 so that a quick restart doesn't fail with
    // AddrInUse while the old socket is in TIME_WAIT.
    let listener = {
        let socket = socket2::Socket::new(
            socket2::Domain::IPV4,
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        ).expect("Failed to create socket");
        socket.set_reuse_address(true).expect("set_reuse_address failed");
        socket.set_nonblocking(true).expect("set_nonblocking failed");
        socket.bind(&socket2::SockAddr::from(
            format!("{}:{}", cfg.host, cfg.port).parse::<std::net::SocketAddr>().unwrap_or_else(|_| {
                std::net::SocketAddr::from(([0, 0, 0, 0], cfg.port))
            })
        )).expect("Failed to bind");
        socket.listen(1024).expect("Failed to listen");
        tokio::net::TcpListener::from_std(socket.into()).expect("Failed to convert to TcpListener")
    };
    // Graceful shutdown: on SIGINT/SIGTERM, stop accepting new connections
    // but drain in-flight requests before exiting. Without this, Ctrl+C in
    // start.sh instantly severs codex's proxy connection, which codex
    // retries 5x against a dead server before terminating with exit code 1.
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        logger::info("main", "Received SIGINT, draining connections before shutdown...");
        let _ = shutdown_tx.send(());
    });
    axum::serve(listener, app)
        .with_graceful_shutdown(async { shutdown_rx.await.ok(); })
        .await
        .expect("Server error");
}

/// Estimate the total token count of a Chat Completions payload.
///
/// The previous estimator only checked `msg.get("content").as_str()`, which
/// returned `None` (and thus 0 tokens) for:
///   - Array content (multimodal: text + images)
///   - Tool call messages (`tool_calls` field, not `content`)
///   - Tool result messages (`tool` role with `content`)
///   - The `tools` schema array (not counted at all)
///
/// This caused 1.7M-token requests to be estimated at ~50K tokens, so the
/// context guard never fired and the oversized payload went to upstream.
///
/// This improved estimator recursively measures the JSON string length of
/// the entire payload (messages + tools) and divides by ~4 chars/token.
/// It's still a rough heuristic (real tokenizers vary), but it catches the
/// cases the old one missed.
fn estimate_tokens(cc_payload: &Value) -> usize {
    let mut total_chars: usize = 0;

    // Count message content (handles string, array, and tool_calls)
    if let Some(messages) = cc_payload.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            // String content
            if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                total_chars += content.len();
            }
            // Array content (multimodal: [{"type":"text","text":"..."}, ...])
            if let Some(parts) = msg.get("content").and_then(|c| c.as_array()) {
                for part in parts {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        total_chars += text.len();
                    }
                    // Image content has no text but still costs tokens
                    if part.get("type").and_then(|t| t.as_str()) == Some("image_url") {
                        total_chars += 1000; // rough estimate per image
                    }
                }
            }
            // Tool calls (function name + arguments)
            if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                for tc in tool_calls {
                    if let Some(func) = tc.get("function") {
                        if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                            total_chars += name.len();
                        }
                        if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                            total_chars += args.len();
                        }
                    }
                }
            }
            // Role field
            if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                total_chars += role.len();
            }
        }
    }

    // Count tools schema (previously not counted at all)
    if let Some(tools) = cc_payload.get("tools").and_then(|t| t.as_array()) {
        for tool in tools {
            // Serialize each tool definition to measure its size
            total_chars += tool.to_string().len();
        }
    }

    // Rough: ~4 chars per token
    total_chars / 4
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

   // Context size warning: estimate tokens and log if exceeds model's context window.
   // We do NOT trim here — codex has its own context management (auto-compaction
   // at 90% of context_window, tool_output_token_limit, etc.). The proxy's
   // trimming was broken: it only counted `content` as a string (missing array
   // content, tool_calls, tool results, and the tools schema entirely), and the
   // `total > 4` guard allowed huge single messages through. Removing it lets
   // codex's proven context management handle this correctly.
   let context_window = get_context_window(&state.model_catalog, &model_alias);
   let estimated_tokens = estimate_tokens(&cc_payload);
   if estimated_tokens as u64 > context_window {
       logger::warn("responses", &format!(
           "Context size warning: ~{} tokens may exceed context_window {} for model {} — codex should auto-compact",
           estimated_tokens, context_window, model_alias
       ));
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
            // Read cursor: bytes before this offset have already been
            // processed. Instead of drain(..pos+1) which shifts all
            // remaining bytes left (O(n²) for n lines), we advance the
            // cursor and compact only once per chunk.
            let mut read_pos: usize = 0;
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
                               while let Some(rel_pos) = buffer[read_pos..].find('\n') {
                                    let pos = read_pos + rel_pos;
                                    let events = translator.process_line(&buffer[read_pos..pos]);
                                    read_pos = pos + 1;
                                   for event in events {
                                       if tx.send(Ok(axum::body::Bytes::from(event.into_bytes()))).await.is_err() {
                                           // Client disconnected — stop reading upstream
                                           upstream_cancel.cancel();
                                            return;
                                        }
                                    }
                                }
                                // Compact: remove processed bytes once per chunk
                                // instead of once per line via drain().
                                if read_pos > 0 {
                                    buffer.drain(..read_pos);
                                    read_pos = 0;
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
                        if tx.send(Ok(KEEPALIVE_BYTES.clone())).await.is_err() {
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
            if read_pos < buffer.len() && !buffer[read_pos..].trim().is_empty() {
                let events = translator.process_line(&buffer[read_pos..]);
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
        // Per-connection UUID for log tracing — distinguishes multiple codex
        // instances using the proxy simultaneously.
        let conn_id = uuid::Uuid::new_v4().simple().to_string();
        let conn_short = &conn_id[..8];
        logger::info("ws", &format!("[{}] WebSocket connection opened for /v1/responses", conn_short));

        // ── Connection-reuse state ─────────────────────────────────────
        // Codex reuses a single WebSocket connection across multiple turns.
        // When codex sends an incremental request (previous_response_id set),
        // it only includes the NEW input items — not the full conversation.
        // We must cache the full request from each response so we can merge
        // the delta with the cached baseline before translating to Chat
        // Completions format.
        //
        // We cache the ENTIRE request (not just the input array) because
        // codex does NOT resend `instructions`, `tools`, `reasoning`, etc.
        // on incremental requests. Without restoring these fields, the LLM
        // loses its system prompt and tool definitions → hallucination.
        //
       // Key: response_id  →  Value: full request JSON from that request
        let mut response_request_cache: std::collections::HashMap<String, std::sync::Arc<Value>> =
            std::collections::HashMap::new();
        // Insertion-ordered list of response_ids so we can evict the
        // oldest entry (FIFO) instead of a random one. HashMap.keys().next()
        // returns an arbitrary key, which can evict the most recently
        // inserted (and currently active) response_id — causing cache
        // misses on the very next incremental request.
        let mut cache_insertion_order: std::collections::VecDeque<String> =
            std::collections::VecDeque::new();

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
                    logger::info("ws", &format!("[{}] WebSocket closed by client", conn_short));
                    break 'connection;
                }
                Err(_) => {
                    logger::info("ws", &format!("[{}] WebSocket idle timeout (300s between turns), closing", conn_short));
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
                    logger::info("ws", &format!("[{}] WebSocket close frame received", conn_short));
                    break 'connection;
                }
            };

            let request_json: Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(e) => {
                    logger::error("ws", &format!("[{}] Invalid JSON: {}", conn_short, e));
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
                logger::info("ws", &format!("[{}] Prewarm request (generate=false), acknowledging with empty response_id", conn_short));
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
                    if let Some(cached_req_arc) = response_request_cache.get(prev_id) {
                        let cached_req = cached_req_arc.as_ref();
                        // Merge: cached full input + new delta items from this request
                        let cached_input = cached_req.get("input").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                        let new_items = request_json.get("input")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();

                        let mut merged_input = cached_input.clone();
                        merged_input.extend(new_items);

                        effective_request["input"] = json!(merged_input);

                        // Restore fields that codex does NOT resend on incremental
                        // requests: instructions (system prompt), tools, reasoning,
                        // tool_choice, parallel_tool_calls, etc.
                        // Without these, the LLM has no system prompt or tool
                        // definitions → hallucination.
                        for field in ["instructions", "tools", "reasoning", "tool_choice",
                            "parallel_tool_calls", "max_output_tokens", "max_tool_calls",
                            "temperature", "top_p", "truncation", "text",
                            "safety_identifier", "service_tier", "store",
                            "prompt_cache_key", "metadata"] {
                            if effective_request.get(field).is_none() {
                                if let Some(v) = cached_req.get(field) {
                                    effective_request[field] = v.clone();
                                }
                            }
                        }

                        logger::info("ws", &format!(
                            "[{}] Incremental request: previous_response_id={}, merged {} cached + {} new = {} total items",
                            conn_short, prev_id, cached_input.len(),
                            request_json.get("input").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
                            merged_input.len()
                        ));
                    } else {
                        // Cache miss: previous_response_id was evicted (FIFO
                        // at 50 entries) or belongs to a different connection.
                        // Sending only the delta items (1-3 new items) would
                        // give the LLM an incomplete conversation → hallucination.
                        // Instead, send an explicit 409 error so codex retries
                        // with a full request (no previous_response_id).
                        logger::error("ws", &format!(
                            "[{}] previous_response_id={} not found in cache — sending 409 to force full retry",
                            conn_short, prev_id
                        ));
                        let err_payload = json!({
                            "type": "error",
                            "status": 409,
                            "error": {
                                "message": format!("previous_response_id {} not found in cache — please retry with full context", prev_id)
                            }
                        });
                        let _ = socket.send(axum::extract::ws::Message::Text(
                            err_payload.to_string()
                        )).await;
                        continue 'connection;
                    }
                }
            }

           // Cache this request's full effective_request under the new response_id
           // for future incremental requests. Store the effective_request (after
           // merge) so the cache always has the complete conversation.
           // Use Arc to avoid cloning the full request JSON (can be large with
           // many input items) — previously cloned twice per WS turn.
           let effective_request_arc = std::sync::Arc::new(effective_request.clone());
           response_request_cache.insert(response_id_for_cache.clone(), effective_request_arc.clone());
           cache_insertion_order.push_back(response_id_for_cache.clone());
           // Limit cache size to prevent unbounded memory growth
           // Evict the OLDEST entry (FIFO), not a random one — using
           // HashMap.keys().next() returns an arbitrary key, which risks
           // evicting the most recently inserted (active) response_id.
           // Increased from 10 to 50 to reduce cache misses on long sessions.
           if response_request_cache.len() > 50 {
               if let Some(oldest_key) = cache_insertion_order.pop_front() {
                   response_request_cache.remove(&oldest_key);
               }
           }

            let mut cc_payload = translate_request::responses_to_chat_completions(
                &backend_model,
                &effective_request,
                is_stream,
            );
            cc_payload = translate_request::inject_thinking(cc_payload, &backend_model);

            // Context size warning: estimate tokens and log if exceeds model's context window.
            // We do NOT trim here — codex has its own context management (auto-compaction
            // at 90% of context_window, tool_output_token_limit, etc.). The proxy's
            // trimming was broken: it only counted `content` as a string (missing array
            // content, tool_calls, tool results, and the tools schema entirely), and the
            // `total > 4` guard allowed huge single messages through. Removing it lets
            // codex's proven context management handle this correctly.
            let context_window = get_context_window(&state.model_catalog, &model_alias);
            let estimated_tokens = estimate_tokens(&cc_payload);
            if estimated_tokens as u64 > context_window {
                logger::warn("ws", &format!(
                    "[{}] Context size warning: ~{} tokens may exceed context_window {} for model {} — codex should auto-compact",
                    conn_short, estimated_tokens, context_window, model_alias
                ));
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

           logger::info("ws", &format!("[{}] WS POST {} model={} stream={}", conn_short, url, backend_model, is_stream));
            logger::info("ws", &format!("[{}] WS POST {} model={} stream={} input_items={}",
                conn_short, url, backend_model, is_stream,
                effective_request.get("input").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0)));

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
                logger::error("ws", &format!("[{}] Upstream {}: {}", conn_short, status, logger::extract_error_summary(&body_text, 300)));
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
          // Pass Arc<Value> clone to translator — cheaper than cloning the
          // full request JSON (which can be large with many input items).
           let mut translator = translate_response::ResponseTranslator::new_raw_json_arc(
              &response_id_for_cache,
               Some(std::sync::Arc::clone(&effective_request_arc)),
          );

            let byte_stream = resp.bytes_stream();
            let mut stream = byte_stream;
            let mut buffer = String::new();
            // Read cursor: bytes before this offset have already been
            // processed. Instead of drain(..pos+1) which shifts all
            // remaining bytes left (O(n²) for n lines), we advance the
            // cursor and compact only once per chunk.
            let mut read_pos: usize = 0;

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

                               while let Some(rel_pos) = buffer[read_pos..].find('\n') {
                                    let pos = read_pos + rel_pos;
                                    let events = translator.process_line(&buffer[read_pos..pos]);
                                    read_pos = pos + 1;
                                   for event in events {
                                       stream_event_count += 1;
                                       // In raw JSON mode, process_line returns JSON objects directly.
                                       if socket.send(axum::extract::ws::Message::Text(event)).await.is_err() {
                                           break 'connection; // socket dead
                                       }
                                   }
                                }
                                // Compact: remove processed bytes once per chunk
                                // instead of once per line via drain().
                                if read_pos > 0 {
                                    buffer.drain(..read_pos);
                                    read_pos = 0;
                                }
                            }
                            Some(Err(e)) => {
                                logger::error("ws", &format!("[{}] WS stream error: {}", conn_short, e));
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
           if read_pos < buffer.len() && !buffer[read_pos..].trim().is_empty() {
               let events = translator.process_line(&buffer[read_pos..]);
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
                logger::error("ws", &format!("[{}] Upstream stream ended with no content events — sending error instead of response.completed", conn_short));
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

            // ── Append assistant output to cached request ────────────────
            // CRITICAL: After the response completes, we must append the
            // assistant's output items (messages, function calls, reasoning)
            // to the cached request's input array. Without this, on the next
            // incremental request (previous_response_id), the merged
            // conversation would contain only the original user messages +
            // new tool results — but NOT the assistant's previous text or
            // tool calls. The LLM would not see its own prior output and
            // would re-do the same work, causing an infinite loop.
            let output_items = translator.get_output_items();
            if !output_items.is_empty() {
                if let Some(cached_arc) = response_request_cache.get(&response_id_for_cache).cloned() {
                    let mut cached_arc = cached_arc;
                    let cached = Arc::make_mut(&mut cached_arc);
                    if let Some(input_arr) = cached.get_mut("input").and_then(|v| v.as_array_mut()) {
                        let before_len = input_arr.len();
                        input_arr.extend(output_items);
                        logger::info("ws", &format!(
                            "[{}] Appended {} assistant output items to cached request (input: {} → {})",
                            conn_short, input_arr.len() - before_len, before_len, input_arr.len()
                        ));
                    }
                    // Put the updated Arc back into the cache
                    response_request_cache.insert(response_id_for_cache.clone(), cached_arc);
                }
            }

            // Turn complete — loop back to recv() the next request on the
            // SAME connection.  No log spam; only log on actual close.
        }

        logger::info("ws", &format!("[{}] WebSocket connection closed", conn_short));
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

    // Context size warning: estimate tokens and log if exceeds model's context window.
    // We do NOT trim here — codex has its own context management (auto-compaction
    // at 90% of context_window, tool_output_token_limit, etc.). The proxy's
    // trimming was broken: it only counted `content` as a string (missing array
    // content, tool_calls, tool results, and the tools schema entirely), and the
    // `total > 4` guard allowed huge single messages through. Removing it lets
    // codex's proven context management handle this correctly.
    let context_window = get_context_window(&state.model_catalog, &model_alias);
    let estimated_tokens = estimate_tokens(&cc_payload);
    if estimated_tokens as u64 > context_window {
        logger::warn("compact", &format!(
            "Context size warning: ~{} tokens may exceed context_window {} — codex should auto-compact",
            estimated_tokens, context_window
        ));
    }

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
    let mut read_pos: usize = 0;
    let mut full_text = String::new();
    let mut finish_reason: Option<String> = None;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                buffer.push_str(&String::from_utf8_lossy(&chunk));
              while let Some(rel_pos) = buffer[read_pos..].find('\n') {
                    let pos = read_pos + rel_pos;
                    let line = buffer[read_pos..pos].to_string();
                    read_pos = pos + 1;
                    let line = line.trim_end();
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
               // Compact: remove processed bytes once per chunk
               if read_pos > 0 {
                   buffer.drain(..read_pos);
                   read_pos = 0;
               }
            }
            Err(e) => {
                logger::error("compact", &format!("Stream error: {}", e));
                break;
            }
        }
    }

    // Flush remaining buffer (data after last newline)
    if read_pos < buffer.len() {
        let remaining = buffer[read_pos..].trim();
        if !remaining.is_empty() {
            if let Some(data) = remaining.strip_prefix("data: ") {
                if data.trim() != "[DONE]" {
                    if let Ok(chunk_json) = serde_json::from_str::<Value>(data) {
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
