mod config;
mod auth;
mod translate_request;
mod translate_response;
mod logger;

use axum::{
    body::Body,
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json,
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use std::time::Duration;

#[derive(Clone)]
struct AppState {
    cfg: Arc<config::Config>,
    token_mgr: Arc<auth::TokenManager>,
    http_client: reqwest::Client,
    /// Caps the number of concurrent in-flight upstream requests so that
    /// burst load from many VS Code / Codex CLI clients cannot exhaust the
    /// connection pool or overwhelm EricAI. 128 is a generous default.
    upstream_sem: Arc<Semaphore>,
}


/// Extract the JSON payload from an SSE-formatted string.
///
/// The translator produces SSE frames like:
///   event: response.created
///   data: {"type":"response.created",...}
///
/// For HTTP SSE that's correct, but Codex's WebSocket client expects each WS
/// text frame to be a **raw JSON object** (it does `serde_json::from_str` on
/// the frame text). This helper strips the SSE envelope and returns just JSON.
fn sse_to_ws_message(sse: &str) -> Option<String> {
    for line in sse.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            return Some(data.to_string());
        }
        if let Some(data) = line.strip_prefix("data:") {
            return Some(data.trim().to_string());
        }
    }
    None
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
        .pool_max_idle_per_host(64)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .build()
        .expect("Failed to create HTTP client");

    let upstream_sem = Arc::new(Semaphore::new(128));

    let state = AppState {
        cfg: cfg.clone(),
        token_mgr,
        http_client,
        upstream_sem,
    };

    let app = axum::Router::new()
        .route("/v1/models", get(handle_models))
        .route("/v1/responses", post(handle_responses).get(handle_responses_ws))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/backend-api/codex/responses", post(handle_responses).get(handle_responses_ws))
        .route("/backend-api/codex/responses/compact", post(handle_compact))
        .route("/health", get(handle_health))
        .with_state(state);

    let addr = format!("{}:{}", cfg.host, cfg.port);
    logger::info("main", &format!("ericai-proxy listening on http://{}", addr));

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind");
    axum::serve(listener, app).await.expect("Server error");
}

// ─── GET /v1/models ─────────────────────────────────────────────────────────
async fn handle_models(State(state): State<AppState>) -> impl IntoResponse {
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
async fn handle_health() -> impl IntoResponse {
    Json(json!({"status": "ok", "service": "ericai-proxy"}))
}

// ─── POST /v1/responses ────────────────────────────────────────────────────
async fn handle_responses(
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
    let backend_model = state.cfg.all_models().iter()
        .find(|(alias, _)| *alias == model_alias)
        .map(|(_, name)| name.clone())
        .unwrap_or_else(|| {
            if let Some(stripped) = model_alias.strip_prefix("ericai/") {
                stripped.to_string()
            } else {
                model_alias.clone()
            }
        });

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

    // Acquire an upstream concurrency permit so burst load cannot overwhelm
    // EricAI or exhaust the connection pool. Overload -> 503.
    let _permit = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        state.upstream_sem.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(p)) => p,
        _ => {
            logger::warn("responses", "Upstream concurrency limit reached (503)");
            return (StatusCode::SERVICE_UNAVAILABLE, "Upstream busy").into_response();
        }
    };

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

    // On 401, force-refresh the token and retry once.
    let resp = if resp.status() == StatusCode::UNAUTHORIZED {
        logger::warn("responses", "Upstream 401 — force-refreshing token and retrying");
        let token = match state.token_mgr.force_refresh().await {
            Ok(t) => t,
            Err(e) => return (StatusCode::UNAUTHORIZED, format!("Auth refresh failed: {}", e)).into_response(),
        };
        match state.http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Accept", "text/event-stream")
            .json(&cc_payload)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e)).into_response(),
        }
    } else {
        resp
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

        let translated_stream = async_stream::stream! {
            let mut stream = byte_stream;
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        buffer.push_str(&String::from_utf8_lossy(&chunk));

                        while let Some(pos) = buffer.find('\n') {
                            let line = buffer[..pos].to_string();
                            buffer.drain(..pos + 1);

                            let events = translator.process_line(&line);
                            for event in events {
                                yield Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(event.into_bytes()));
                            }
                        }
                    }
                    Err(e) => {
                        logger::error("responses", &format!("Stream error: {}", e));
                        break;
                    }
                }
            }

            if !buffer.trim().is_empty() {
                let events = translator.process_line(&buffer);
                for event in events {
                    yield Ok(axum::body::Bytes::from(event.into_bytes()));
                }
            }

            let events = translator.process_line("data: [DONE]");
            for event in events {
                yield Ok(axum::body::Bytes::from(event.into_bytes()));
            }
        };

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
//
// Codex's Responses WebSocket transport is *stateful and connection-reusing*:
//
//   1. It opens a connection and sends a "prewarm" request — a `response.create`
//      with `generate: false`. The server must answer with `response.created` +
//      `response.completed` (no model output) so the connection is warm.
//   2. It then reuses the *same* connection for the real turn request, and for
//      subsequent turns. On reuse Codex sends `previous_response_id` plus only the
//      *incremental* input items — everything the server already saw (prior input
//      and prior model output) is omitted.
//
// Because our upstream (vLLM /chat/completions) is stateless, we must rebuild the
// full conversation for every generating request:
//
//     full_input = prev.full_input ++ prev.output_items ++ request.input
//
// We therefore keep per-connection state mapping each response_id we issue to the
// full input the model saw plus the output items it produced, and we loop to serve
// many requests over one socket instead of closing after the first.
async fn handle_responses_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(_params): Query<Value>,
) -> Response {
    ws.on_upgrade(move |socket| async move {
        handle_ws_connection(socket, state).await;
    })
}

/// Per-connection conversation state: response_id -> (full input items, output items).
type WsConversations = HashMap<String, (Vec<Value>, Vec<Value>)>;

/// Idle timeout while waiting for the next request frame on a (possibly reused)
/// connection. Codex keeps connections alive across turns, so this is generous.
/// Handshake probes that never send a frame close their TCP side and break us out
/// immediately via `recv() -> None`, so they do not linger for the full duration.
const WS_IDLE_TIMEOUT: Duration = Duration::from_secs(600);

/// Drives a single WebSocket connection, serving any number of `response.create`
/// frames (prewarm + real turns) until the client disconnects or goes idle.
async fn handle_ws_connection(mut socket: WebSocket, state: AppState) {
    logger::info("ws", "WebSocket connection opened for /v1/responses");
    let mut conversations: WsConversations = HashMap::new();

    loop {
        let msg = match tokio::time::timeout(WS_IDLE_TIMEOUT, socket.recv()).await {
            Ok(Some(Ok(m))) => m,
            Ok(Some(Err(_))) | Ok(None) => break,
            Err(_) => {
                logger::info("ws", "WebSocket idle timeout, closing");
                break;
            }
        };

        let text = match msg {
            Message::Text(t) => t,
            Message::Binary(b) => String::from_utf8_lossy(&b).to_string(),
            Message::Close(_) => break,
            Message::Ping(_) | Message::Pong(_) => continue,
        };

        let request_json: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                logger::error("ws", &format!("Invalid JSON: {}", e));
                let _ = ws_send_json(
                    &mut socket,
                    &ws_error_event(400, &format!("Invalid JSON: {}", e)),
                )
                .await;
                continue;
            }
        };

        if handle_ws_request(&state, &mut socket, request_json, &mut conversations)
            .await
            .is_err()
        {
            // Client socket is gone; stop serving this connection.
            break;
        }
    }

    let _ = socket.send(Message::Close(None)).await;
    let _ = socket.close().await;
    logger::info("ws", "WebSocket connection closed");
}

/// Build a Codex-compatible wrapped error event (maps to an HTTP error client-side).
fn ws_error_event(status: u16, message: &str) -> Value {
    json!({
        "type": "error",
        "status": status,
        "error": { "type": "upstream_error", "message": message }
    })
}

async fn ws_send_json(socket: &mut WebSocket, value: &Value) -> Result<(), ()> {
    socket
        .send(Message::Text(value.to_string()))
        .await
        .map_err(|_| ())
}

/// Reconstruct the full conversation input for a request, expanding any
/// `previous_response_id` reference into the stored prior input + output items.
fn rebuild_full_input(request_json: &Value, conversations: &WsConversations) -> Vec<Value> {
    let req_input: Vec<Value> = request_json
        .get("input")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    match request_json
        .get("previous_response_id")
        .and_then(|v| v.as_str())
    {
        Some(prev_id) => match conversations.get(prev_id) {
            Some((prev_full, prev_out)) => {
                let mut v =
                    Vec::with_capacity(prev_full.len() + prev_out.len() + req_input.len());
                v.extend(prev_full.iter().cloned());
                v.extend(prev_out.iter().cloned());
                v.extend(req_input);
                v
            }
            None => {
                logger::warn(
                    "ws",
                    &format!(
                        "Unknown previous_response_id {} — using incremental input only",
                        prev_id
                    ),
                );
                req_input
            }
        },
        None => req_input,
    }
}

/// Store conversation state for a response_id. Only the most recent response is
/// ever referenced via `previous_response_id`, so memory is bounded by clearing
/// old turns when the map grows large.
fn store_conversation(
    conversations: &mut WsConversations,
    response_id: String,
    full_input: Vec<Value>,
    output: Vec<Value>,
) {
    if conversations.len() >= 64 {
        conversations.clear();
    }
    conversations.insert(response_id, (full_input, output));
}

/// If `ws_msg` is a `response.completed` event, capture its output items so the
/// next reused request can rebuild the full conversation history.
fn capture_completed_output(ws_msg: &str, out: &mut Vec<Value>) {
    if let Ok(v) = serde_json::from_str::<Value>(ws_msg) {
        if v.get("type").and_then(|t| t.as_str()) == Some("response.completed") {
            if let Some(output) = v
                .get("response")
                .and_then(|r| r.get("output"))
                .and_then(|o| o.as_array())
            {
                *out = output.clone();
            }
        }
    }
}

/// Handle a single `response.create` frame.
///
/// Returns `Err(())` only when the client socket is gone (caller should close the
/// connection). Upstream/model errors are reported as `error` events but keep the
/// connection open so Codex can retry or fall back.
async fn handle_ws_request(
    state: &AppState,
    socket: &mut WebSocket,
    request_json: Value,
    conversations: &mut WsConversations,
) -> Result<(), ()> {
    let is_prewarm = request_json.get("generate").and_then(|v| v.as_bool()) == Some(false);
    let full_input = rebuild_full_input(&request_json, conversations);
    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());

    // ── Prewarm: connection setup only, no model call. ──
    if is_prewarm {
        logger::info("ws", &format!("WS prewarm (response_id={})", response_id));
        let created = json!({
            "type": "response.created",
            "response": { "id": response_id, "object": "response", "status": "in_progress" }
        });
        let completed = json!({
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "completed",
                "output": [],
                "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }
            }
        });
        store_conversation(conversations, response_id, full_input, Vec::new());
        ws_send_json(socket, &created).await?;
        ws_send_json(socket, &completed).await?;
        return Ok(());
    }

    // ── Real generating request. ──
    let model_alias = request_json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let backend_model = state
        .cfg
        .all_models()
        .iter()
        .find(|(alias, _)| *alias == model_alias)
        .map(|(_, name)| name.clone())
        .unwrap_or_else(|| {
            if let Some(stripped) = model_alias.strip_prefix("ericai/") {
                stripped.to_string()
            } else {
                model_alias.clone()
            }
        });

    // Build the upstream request from the *current* request fields but with the
    // reconstructed full input. WebSocket turns always stream.
    let mut merged = request_json.clone();
    merged["input"] = Value::Array(full_input.clone());

    let mut cc_payload =
        translate_request::responses_to_chat_completions(&backend_model, &merged, true);
    cc_payload = translate_request::inject_thinking(cc_payload, &backend_model);

    let token = match state.token_mgr.get_token().await {
        Ok(t) => t,
        Err(e) => {
            let _ = ws_send_json(socket, &ws_error_event(401, &format!("Auth failed: {}", e))).await;
            return Ok(());
        }
    };

    let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
    let url = format!("{}/chat/completions", base_url);
    logger::info(
        "ws",
        &format!(
            "WS POST {} model={} stream=true (response_id={})",
            url, backend_model, response_id
        ),
    );

    // The concurrency permit guards only the upstream call, not idle waits.
    let _permit = match tokio::time::timeout(
        Duration::from_secs(5),
        state.upstream_sem.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(p)) => p,
        _ => {
            logger::warn("ws", "Upstream concurrency limit reached");
            let _ = ws_send_json(socket, &ws_error_event(503, "Upstream busy")).await;
            return Ok(());
        }
    };

    let resp = match state
        .http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Accept", "text/event-stream")
        .json(&cc_payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let _ = ws_send_json(socket, &ws_error_event(502, &format!("Upstream: {}", e))).await;
            return Ok(());
        }
    };

    // On 401, force-refresh the token and retry once.
    let resp = if resp.status() == StatusCode::UNAUTHORIZED {
        logger::warn("ws", "Upstream 401 — force-refreshing token and retrying");
        match state.token_mgr.force_refresh().await {
            Ok(t) => match state
                .http_client
                .post(&url)
                .header("Authorization", format!("Bearer {}", t))
                .header("Accept", "text/event-stream")
                .json(&cc_payload)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let _ =
                        ws_send_json(socket, &ws_error_event(502, &format!("Upstream: {}", e)))
                            .await;
                    return Ok(());
                }
            },
            Err(e) => {
                let _ = ws_send_json(
                    socket,
                    &ws_error_event(401, &format!("Auth refresh failed: {}", e)),
                )
                .await;
                return Ok(());
            }
        }
    } else {
        resp
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        logger::error(
            "ws",
            &format!(
                "Upstream {}: {}",
                status,
                logger::extract_error_summary(&body_text, 300)
            ),
        );
        logger::error_request("ws-responses", &cc_payload.to_string(), &body_text, status.as_u16());
        let _ = ws_send_json(socket, &ws_error_event(status.as_u16(), &body_text)).await;
        return Ok(());
    }

    // Stream upstream SSE → translate → send raw JSON frames, capturing the final
    // response output so the next reused request can rebuild history.
    let mut translator =
        translate_response::ResponseTranslator::new(&response_id, Some(merged));
    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();
    let mut captured_output: Vec<Value> = Vec::new();
    let mut client_alive = true;

    'outer: while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].to_string();
                    buffer.drain(..pos + 1);
                    for event in translator.process_line(&line) {
                        if let Some(ws_msg) = sse_to_ws_message(&event) {
                            capture_completed_output(&ws_msg, &mut captured_output);
                            if socket.send(Message::Text(ws_msg)).await.is_err() {
                                client_alive = false;
                                break 'outer;
                            }
                        }
                    }
                }
            }
            Err(e) => {
                logger::error("ws", &format!("WS stream error: {}", e));
                break;
            }
        }
    }

    // Flush any trailing buffered line, then emit the terminal [DONE] translation.
    if client_alive && !buffer.trim().is_empty() {
        for event in translator.process_line(&buffer) {
            if let Some(ws_msg) = sse_to_ws_message(&event) {
                capture_completed_output(&ws_msg, &mut captured_output);
                if socket.send(Message::Text(ws_msg)).await.is_err() {
                    client_alive = false;
                    break;
                }
            }
        }
    }
    if client_alive {
        for event in translator.process_line("data: [DONE]") {
            if let Some(ws_msg) = sse_to_ws_message(&event) {
                capture_completed_output(&ws_msg, &mut captured_output);
                if socket.send(Message::Text(ws_msg)).await.is_err() {
                    client_alive = false;
                    break;
                }
            }
        }
    }

    // Remember this turn so a follow-up reused request can rebuild full history.
    store_conversation(conversations, response_id, full_input, captured_output);

    if client_alive {
        Ok(())
    } else {
        Err(())
    }
}

// ─── POST /v1/chat/completions (passthrough) ───────────────────────────────
async fn handle_chat_completions(
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

    let backend_model = state.cfg.all_models().iter()
        .find(|(alias, _)| *alias == model)
        .map(|(_, name)| name.clone())
        .unwrap_or(model);

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

    // Acquire an upstream concurrency permit so burst load cannot overwhelm
    // EricAI or exhaust the connection pool. Overload -> 503.
    let _permit = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        state.upstream_sem.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(p)) => p,
        _ => {
            logger::warn("chat", "Upstream concurrency limit reached (503)");
            return (StatusCode::SERVICE_UNAVAILABLE, "Upstream busy").into_response();
        }
    };

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
async fn handle_compact(
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

    let backend_model = state.cfg.all_models().iter()
        .find(|(alias, _)| *alias == model_alias)
        .map(|(_, name)| name.clone())
        .unwrap_or_else(|| {
            if let Some(stripped) = model_alias.strip_prefix("ericai/") {
                stripped.to_string()
            } else {
                model_alias.clone()
            }
        });

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

    logger::info("compact", &format!("Compact POST {} model={}", url, backend_model));

    // Acquire an upstream concurrency permit so burst load cannot overwhelm
    // EricAI or exhaust the connection pool. Overload -> 503.
    let _permit = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        state.upstream_sem.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(p)) => p,
        _ => {
            logger::warn("compact", "Upstream concurrency limit reached (503)");
            return (StatusCode::SERVICE_UNAVAILABLE, "Upstream busy").into_response();
        }
    };

    let resp = match state.http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
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

    let raw: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            logger::error("compact", &format!("Parse error: {}", e));
            return (StatusCode::BAD_GATEWAY, format!("Parse error: {}", e))
                .into_response();
        }
    };

    let mut response_obj = translate_response::chat_completion_to_response(
        &raw,
        Some(&request_json),
    );

    let output_text = response_obj.get("output")
        .and_then(|o| o.as_array())
        .and_then(|arr| arr.first())
        .and_then(|item| item.get("content"))
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|part| part.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("");

    let compaction_item = json!({
        "type": "compaction",
        "summary_text": output_text,
        "compaction_trigger": "auto",
    });

    response_obj["object"] = json!("response.compaction");
    response_obj["output"] = json!([compaction_item]);

    Json(response_obj).into_response()
}


#[cfg(test)]
mod ws_state_tests {
    use super::*;

    fn item(role: &str, text: &str) -> Value {
        json!({
            "type": "message",
            "role": role,
            "content": [{ "type": if role == "assistant" { "output_text" } else { "input_text" }, "text": text }]
        })
    }

    #[test]
    fn rebuild_without_previous_id_uses_request_input_only() {
        let convs: WsConversations = HashMap::new();
        let req = json!({ "input": [item("user", "hello")] });
        let full = rebuild_full_input(&req, &convs);
        assert_eq!(full, vec![item("user", "hello")]);
    }

    #[test]
    fn rebuild_with_previous_id_concatenates_prior_input_output_and_increment() {
        // Prior turn: user asked something, assistant replied (output captured).
        let prev_full = vec![item("user", "q1")];
        let prev_out = vec![item("assistant", "a1")];
        let mut convs: WsConversations = HashMap::new();
        convs.insert("resp_prev".to_string(), (prev_full.clone(), prev_out.clone()));

        // Reused request: only the new incremental item + previous_response_id.
        let req = json!({
            "previous_response_id": "resp_prev",
            "input": [item("user", "q2")]
        });
        let full = rebuild_full_input(&req, &convs);

        // full_input = prev.full ++ prev.output ++ request.input
        assert_eq!(
            full,
            vec![item("user", "q1"), item("assistant", "a1"), item("user", "q2")]
        );
    }

    #[test]
    fn rebuild_with_unknown_previous_id_falls_back_to_increment() {
        let convs: WsConversations = HashMap::new();
        let req = json!({
            "previous_response_id": "resp_missing",
            "input": [item("user", "q2")]
        });
        let full = rebuild_full_input(&req, &convs);
        assert_eq!(full, vec![item("user", "q2")]);
    }

    #[test]
    fn rebuild_chains_across_two_reused_turns() {
        let mut convs: WsConversations = HashMap::new();

        // Turn 1 (e.g. prewarm or first real request).
        let t1_input = rebuild_full_input(&json!({ "input": [item("user", "q1")] }), &convs);
        store_conversation(&mut convs, "resp_1".to_string(), t1_input, vec![item("assistant", "a1")]);

        // Turn 2 reuses resp_1, sends only q2.
        let t2_req = json!({ "previous_response_id": "resp_1", "input": [item("user", "q2")] });
        let t2_input = rebuild_full_input(&t2_req, &convs);
        assert_eq!(
            t2_input,
            vec![item("user", "q1"), item("assistant", "a1"), item("user", "q2")]
        );
        store_conversation(&mut convs, "resp_2".to_string(), t2_input, vec![item("assistant", "a2")]);

        // Turn 3 reuses resp_2, sends only q3 → full history of all three turns.
        let t3_req = json!({ "previous_response_id": "resp_2", "input": [item("user", "q3")] });
        let t3_input = rebuild_full_input(&t3_req, &convs);
        assert_eq!(
            t3_input,
            vec![
                item("user", "q1"),
                item("assistant", "a1"),
                item("user", "q2"),
                item("assistant", "a2"),
                item("user", "q3"),
            ]
        );
    }

    #[test]
    fn capture_completed_output_extracts_output_array() {
        let mut out = Vec::new();
        let completed = json!({
            "type": "response.completed",
            "response": { "id": "resp_1", "output": [item("assistant", "hi")] }
        })
        .to_string();
        capture_completed_output(&completed, &mut out);
        assert_eq!(out, vec![item("assistant", "hi")]);
    }

    #[test]
    fn capture_completed_output_ignores_other_events() {
        let mut out = Vec::new();
        let created = json!({ "type": "response.created", "response": { "id": "resp_1" } }).to_string();
        capture_completed_output(&created, &mut out);
        assert!(out.is_empty());

        let delta = json!({ "type": "response.output_text.delta", "delta": "x" }).to_string();
        capture_completed_output(&delta, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn store_conversation_bounds_memory() {
        let mut convs: WsConversations = HashMap::new();
        for i in 0..70 {
            store_conversation(&mut convs, format!("resp_{}", i), vec![item("user", "x")], Vec::new());
        }
        // Map is cleared once it reaches 64 entries, so it never grows unbounded.
        assert!(convs.len() <= 64);
        // The most recently inserted entry (the only one ever referenced) is present.
        assert!(convs.contains_key("resp_69"));
    }
}
