mod config;
mod auth;
mod translate_request;
mod translate_response;
mod logger;

use axum::{
    body::Body,
    extract::{Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json,
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    cfg: Arc<config::Config>,
    token_mgr: Arc<auth::TokenManager>,
    http_client: reqwest::Client,
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

        let translated_stream = async_stream::stream! {
            let mut stream = byte_stream;
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        buffer.push_str(&String::from_utf8_lossy(&chunk));

                        while let Some(pos) = buffer.find('\n') {
                            let line = buffer[..pos].to_string();
                            buffer = buffer[pos + 1..].to_string();

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
async fn handle_responses_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(_params): Query<Value>,
) -> Response {
    ws.on_upgrade(move |mut socket| async move {
        use futures_util::StreamExt;
        logger::info("ws", "WebSocket connection opened for /v1/responses");

        let msg = match socket.recv().await {
            Some(Ok(m)) => m,
            _ => {
                logger::warn("ws", "WebSocket closed before receiving request");
                return;
            }
        };

        let text = match msg {
            axum::extract::ws::Message::Text(t) => t,
            axum::extract::ws::Message::Binary(b) => String::from_utf8_lossy(&b).to_string(),
            _ => {
                let _ = socket.send(axum::extract::ws::Message::Text(
                    "{\"error\":\"Expected text message\"}".to_string()
                )).await;
                return;
            }
        };

        let request_json: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                logger::error("ws", &format!("Invalid JSON: {}", e));
                let _ = socket.send(axum::extract::ws::Message::Text(
                    format!("{{\"error\":\"Invalid JSON: {}\"}}", e)
                )).await;
                return;
            }
        };

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

        let is_stream = request_json.get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let mut cc_payload = translate_request::responses_to_chat_completions(
            &backend_model,
            &request_json,
            is_stream,
        );
        cc_payload = translate_request::inject_thinking(cc_payload, &backend_model);

        let token = match state.token_mgr.get_token().await {
            Ok(t) => t,
            Err(e) => {
                let _ = socket.send(axum::extract::ws::Message::Text(
                    format!("{{\"error\":\"Auth failed: {}\"}}", e)
                )).await;
                return;
            }
        };

        let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
        let url = format!("{}/chat/completions", base_url);

        logger::info("ws", &format!("WS POST {} model={} stream={}", url, backend_model, is_stream));

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
                let _ = socket.send(axum::extract::ws::Message::Text(
                    format!("{{\"error\":\"Upstream: {}\"}}", e)
                )).await;
                return;
            }
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            logger::error("ws", &format!("Upstream {}: {}", status, logger::extract_error_summary(&body_text, 300)));
            logger::error_request("ws-responses", &cc_payload.to_string(), &body_text, status.as_u16());
            let _ = socket.send(axum::extract::ws::Message::Text(
                format!("{{\"error\":\"Upstream {}: {}\"}}", status, body_text)
            )).await;
            return;
        }

        let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
        let mut translator = translate_response::ResponseTranslator::new(
            &response_id,
            Some(request_json),
        );

        let byte_stream = resp.bytes_stream();
        let mut stream = byte_stream;
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    buffer.push_str(&String::from_utf8_lossy(&chunk));

                    while let Some(pos) = buffer.find('\n') {
                        let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 1..].to_string();

                        let events = translator.process_line(&line);
                        for event in events {
                            if socket.send(axum::extract::ws::Message::Text(event)).await.is_err() {
                                return;
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

        if !buffer.trim().is_empty() {
            let events = translator.process_line(&buffer);
            for event in events {
                let _ = socket.send(axum::extract::ws::Message::Text(event)).await;
            }
        }

        let events = translator.process_line("data: [DONE]");
        for event in events {
            let _ = socket.send(axum::extract::ws::Message::Text(event)).await;
        }

        logger::info("ws", "WebSocket connection closed");
    })
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
