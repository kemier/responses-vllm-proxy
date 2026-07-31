#[allow(clippy::duplicate_mod)]
pub mod auth;
#[allow(clippy::duplicate_mod)]
pub mod config;
#[allow(clippy::duplicate_mod)]
pub mod logger;
#[allow(clippy::duplicate_mod)]
pub mod translate_anthropic;
#[allow(clippy::duplicate_mod)]
pub mod translate_request;
#[allow(clippy::duplicate_mod)]
pub mod translate_response;

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Query, State, WebSocketUpgrade},
    http::{header::AUTHORIZATION, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json,
};
use axum::extract::Request;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::time::{interval, Duration};

/// Cached model catalog loaded once at startup from ~/.codex/model_catalog.json.
/// Previously this file was read from disk on EVERY request (HTTP, WebSocket,
/// and compact), causing 13+ disk reads per WebSocket connection with
/// incremental requests. Now it's loaded once and shared via Arc.
pub type ModelCatalog = Arc<Option<Value>>;

/// Cached backend metadata (routing_tags, lifecycle) fetched from
/// `ericai api models.list` at startup and refreshed every 5 minutes.
/// Keyed by backend model ID (e.g. "zai-org/GLM-5.2-FP8").
type BackendMetaCache = Arc<tokio::sync::RwLock<std::collections::HashMap<String, ModelBackendMeta>>>;

#[derive(Clone, Debug, Default)]
pub struct ModelBackendMeta {
    pub routing_tags: Vec<String>,
    pub lifecycle: String,
}

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<config::Config>,
    pub token_mgr: Arc<auth::TokenManager>,
    pub http_client: reqwest::Client,
    pub model_catalog: ModelCatalog,
    pub shutdown: tokio_util::sync::CancellationToken,
    pub rate_limiter: Arc<RateLimiter>,
    pub backend_meta: BackendMetaCache,
}

/// Pre-allocated SSE keepalive comment bytes — avoids allocating a new
/// String on every 15-second keepalive tick.
static KEEPALIVE_BYTES: axum::body::Bytes = axum::body::Bytes::from_static(b": keepalive\n\n");

/// Fetch model metadata (routing_tags, lifecycle) from the EricAI backend
/// via `ericai api models.list`. Returns a map keyed by backend model ID.
/// Used to populate the `tags` field in /v1/models so ericli can resolve
/// tag:fast / tag:coding without needing rayllm_metadata (which the proxy
/// strips). Falls back to config.yaml `tags` field when the backend is
/// unreachable.
async fn fetch_backend_meta() -> std::collections::HashMap<String, ModelBackendMeta> {
    let mut map = std::collections::HashMap::new();
    let ericai_bin = std::env::var("HOME")
        .ok()
        .map(|h| format!("{}/.local/bin/ericai", h))
        .unwrap_or_else(|| "ericai".to_string());

    let result = tokio::process::Command::new(&ericai_bin)
        .args(["api", "models.list"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await;

    let output = match result {
        Ok(o) if o.status.success() => o.stdout,
        _ => {
            logger::warn("backend_meta", "Could not fetch ericai api models.list — using config.yaml tags only");
            return map;
        }
    };

    let stdout = String::from_utf8_lossy(&output);
    // The output is a stream of JSON objects (not a JSON array).
    // Split on }{ boundaries and parse each.
    let mut buf = String::new();
    for line in stdout.lines() {
        buf.push_str(line);
        buf.push('\n');
    }
    // Try parsing as a single JSON first (some versions return {"data":[...]})
    if let Ok(parsed) = serde_json::from_str::<Value>(&buf) {
        let models = parsed.get("data").and_then(|d| d.as_array())
            .or_else(|| parsed.as_array());
        if let Some(models) = models {
            for m in models {
                extract_meta(m, &mut map);
            }
            return map;
        }
    }
    // Fallback: split on object boundaries
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = buf.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'{' => { if depth == 0 { start = i; } depth += 1; }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    if let Ok(obj) = serde_json::from_slice::<Value>(&bytes[start..=i]) {
                        extract_meta(&obj, &mut map);
                    }
                }
            }
            _ => {}
        }
    }
    map
}

fn extract_meta(m: &Value, map: &mut std::collections::HashMap<String, ModelBackendMeta>) {
    let mid = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
    if mid.is_empty() || mid.starts_with("tag:") { return; }
    let rayllm = m.get("rayllm_metadata");
    let engine = rayllm.and_then(|r| r.get("engine_config"));
    let tags: Vec<String> = engine
        .and_then(|e| e.get("routing_tags"))
        .and_then(|t| t.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let lifecycle = engine
        .and_then(|e| e.get("lifecycle"))
        .and_then(|l| l.as_str())
        .unwrap_or("")
        .to_string();
    if !tags.is_empty() || !lifecycle.is_empty() {
        map.insert(mid.to_string(), ModelBackendMeta { routing_tags: tags, lifecycle });
    }
}

/// Background task that refreshes backend metadata every 5 minutes.
async fn backend_meta_refresh_loop(cache: BackendMetaCache, shutdown: tokio_util::sync::CancellationToken) {
    loop {
        let new_meta = fetch_backend_meta().await;
        if !new_meta.is_empty() {
            let count = new_meta.len();
            let mut w = cache.write().await;
            *w = new_meta;
            drop(w);
            logger::info("backend_meta", &format!("Fetched {} models with routing_tags from backend", count));
        }
        // Wait 5 minutes or until shutdown
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs(300)) => {}
            _ = shutdown.cancelled() => break,
        }
    }
}

/// Simple per-IP rate limiter using a fixed window.
/// Prevents abuse when the proxy is network-exposed (CWE-770).
pub struct RateLimiter {
    buckets: std::sync::Mutex<std::collections::HashMap<String, (std::time::Instant, u32)>>,
    max_per_minute: u32,
}

impl RateLimiter {
    pub fn new(max_per_minute: u32) -> Self {
        Self {
            buckets: std::sync::Mutex::new(std::collections::HashMap::new()),
            max_per_minute,
        }
    }

    pub fn check(&self, client_ip: &str) -> bool {
        if self.max_per_minute == 0 {
            return true;
        }
        let mut buckets = self.buckets.lock().unwrap();
        let now = std::time::Instant::now();
        let window = std::time::Duration::from_secs(60);

        let entry = buckets.entry(client_ip.to_string()).or_insert((now, 0));

        if now.duration_since(entry.0) > window {
            *entry = (now, 1);
            true
        } else {
            entry.1 += 1;
            entry.1 <= self.max_per_minute
        }
    }
}

/// Look up a model's context_window from the cached catalog.
/// Falls back to 200000 (200K) if the catalog isn't loaded or the model
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
    196_608
}

#[tokio::main]
async fn main() {
    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.yaml".to_string());

    let cfg = match config::Config::load(&config_path) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Failed to load config from {}: {}", config_path, e);
            std::process::exit(1);
        }
    };

    // Initialize logger
    let to_stderr = !cfg.logging_to_file || cfg.debug;
    logger::init(
        &cfg.log_dir,
        cfg.logs_max_total_size_mb,
        to_stderr,
        &cfg.log_level,
    );

    if cfg.ericai_provider().is_none() {
        logger::error("main", "No 'ericai' provider found in config");
        std::process::exit(1);
    }

    let token_mgr = Arc::new(auth::TokenManager::new(
        &cfg.auth_dir,
        cfg.token_endpoint.clone(),
        cfg.client_id.clone(),
    ));
    let rate_limiter = Arc::new(RateLimiter::new(cfg.rate_limit_per_minute));
    // NOTE: We do NOT set a global .timeout() on the client because reqwest's
    // timeout applies to the ENTIRE request including the streaming body.
    // For long conversations with large contexts, the upstream may take 5+
    // minutes to stream the full response. A 300s timeout would kill active
    // streams mid-generation, causing "Conversation interrupted" in Claude
    // Code and "stream disconnected" in codex.
    //
    // Instead, we rely on per-request timeouts (set via .timeout() on
    // individual RequestBuilders where needed) and SSE keepalive comments
    // to prevent idle connection drops.
    let http_client = reqwest::Client::builder()
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
            .and_then(|s| serde_json::from_str::<Value>(&s).ok()),
    );
    if model_catalog.is_some() {
        logger::info(
            "main",
            "Model catalog loaded from ~/.codex/model_catalog.json",
        );
    } else {
        logger::warn(
            "main",
            "Model catalog not found — using default context_window=200000",
        );
    }

    // Shared shutdown signal — cancelled on SIGINT/SIGTERM so all WebSocket
    // handlers can close their connections gracefully.
    let shutdown_token = tokio_util::sync::CancellationToken::new();

    // Backend metadata cache — fetches routing_tags from `ericai api models.list`
    // at startup and refreshes every 5 minutes. Used to populate the `tags`
    // field in /v1/models so ericli can resolve tag:fast / tag:coding.
    let backend_meta: BackendMetaCache = Arc::new(tokio::sync::RwLock::new(
        std::collections::HashMap::new()
    ));
    {
        let bm = backend_meta.clone();
        let st = shutdown_token.clone();
        tokio::spawn(async move { backend_meta_refresh_loop(bm, st).await; });
    }

    let state = AppState {
        cfg: cfg.clone(),
        token_mgr,
        http_client,
        model_catalog,
        shutdown: shutdown_token.clone(),
        rate_limiter,
        backend_meta,
    };

    let app = axum::Router::new()
        .route("/v1/models", get(handle_models))
        .route(
            "/v1/responses",
            post(handle_responses).get(handle_responses_ws),
        )
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/messages", post(handle_messages))
        // Codex can send very large request bodies (up to ~30MB for long
        // conversation contexts with 196K+ tokens). Axum's default body
        // limit is 2MB, which causes "413 Payload Too Large" errors that
        // manifest as "stream disconnected before completion" in codex.
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50 MB
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        // Fallback routes for codex's hardcoded ChatGPT-style base URL
        // (https://chatgpt.com/backend-api/codex). Only used if someone
        // configures base_url as http://localhost:8317/backend-api/codex
        // instead of the standard /v1. With base_url=/v1 (default config),
        // these routes are never hit — codex calls /v1/responses instead.
        .route(
            "/backend-api/codex/responses",
            post(handle_responses).get(handle_responses_ws),
        )
        .route("/backend-api/codex/responses/compact", post(handle_compact))
        .route("/v1/responses/compact", post(handle_compact))
        .route("/health", get(handle_health))
        .route("/v1/summarize-session", post(handle_summarize_session))
        .with_state(state);

    let addr = format!("{}:{}", cfg.host, cfg.port);
    logger::info(
        "main",
        &format!("ericai-proxy listening on http://{}", addr),
    );

    // Bind with retry — give the old process time to fully release the port.
    // Use SO_REUSEADDR via socket2 so that a quick restart doesn't fail with
    // AddrInUse while the old socket is in TIME_WAIT.
    let listener = {
        let socket = socket2::Socket::new(
            socket2::Domain::IPV4,
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        )
        .expect("Failed to create socket");
        socket
            .set_reuse_address(true)
            .expect("set_reuse_address failed");
        socket
            .set_nonblocking(true)
            .expect("set_nonblocking failed");
        socket
            .bind(&socket2::SockAddr::from(
                format!("{}:{}", cfg.host, cfg.port)
                    .parse::<std::net::SocketAddr>()
                    .unwrap_or_else(|_| std::net::SocketAddr::from(([0, 0, 0, 0], cfg.port))),
            ))
            .expect("Failed to bind");
        socket.listen(1024).expect("Failed to listen");
        tokio::net::TcpListener::from_std(socket.into()).expect("Failed to convert to TcpListener")
    };
    // Graceful shutdown: on SIGINT OR SIGTERM, cancel all WebSocket
    // connections (so codex gets a proper Close frame instead of a TCP RST),
    // stop accepting new connections, and give in-flight HTTP requests a
    // few seconds to finish before the process exits.
    //
    // Without this, Ctrl+C in start.sh severs codex's WebSocket connection
    // (axum 0.7's with_graceful_shutdown does NOT close upgraded WS
    // connections — it only stops accepting NEW ones), and `make stop`
    // (pkill → SIGTERM) kills the proxy instantly with no shutdown at all.
    // Codex then retries 5× against a dead server and terminates with
    // exit code 1.
    let shutdown_token = tokio_util::sync::CancellationToken::new();
    tokio::spawn({
        let token = shutdown_token.clone();
        async move {
            // Wait for SIGINT (Ctrl+C) OR SIGTERM (kill / pkill / make stop).
            let sigint = tokio::signal::ctrl_c();
            let sigterm = async {
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("install SIGTERM handler")
                    .recv()
                    .await;
            };
            tokio::pin!(sigint);
            tokio::select! {
                _ = &mut sigint => {
                    logger::info("main", "Received SIGINT, initiating graceful shutdown...");
                }
                _ = sigterm => {
                    logger::info("main", "Received SIGTERM, initiating graceful shutdown...");
                }
            }
            token.cancel();
        }
    });
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
        .with_graceful_shutdown(async move {
            shutdown_token.cancelled().await;
        })
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

    // Count tools schema (previously not counted at all).
    // Use a recursive walker instead of `tool.to_string().len()` to avoid
    // allocating a full JSON serialization of every tool definition on every
    // request (Code Mode tool descriptions are 3K+ chars each).
    if let Some(tools) = cc_payload.get("tools").and_then(|t| t.as_array()) {
        for tool in tools {
            total_chars += json_byte_len(tool);
        }
    }

    // Rough: ~4 chars per token
    total_chars / 4
}

/// Approximate the serialized JSON length of a `Value` without allocating
/// a full JSON string. This is a length-only walker — it counts the bytes
/// `serde_json::to_string` would produce (within a few percent for normal
/// inputs; precise enough for token estimation, which is itself a /4
/// heuristic).
fn json_byte_len(v: &Value) -> usize {
    match v {
        Value::Null => 4,        // "null"
        Value::Bool(true) => 4,  // "true"
        Value::Bool(false) => 5, // "false"
        Value::Number(n) => {
            // Numbers: just count the digits/dots via the cheap `as_*` path.
            // For non-finite (shouldn't occur in JSON), use a small fallback.
            if let Some(i) = n.as_i64() {
                let mut len = if i < 0 { 1 } else { 0 };
                let mut abs = i.unsigned_abs();
                if abs == 0 {
                    return 1;
                }
                while abs > 0 {
                    len += 1;
                    abs /= 10;
                }
                len
            } else if let Some(u) = n.as_u64() {
                let mut len = 0;
                let mut abs = u;
                if abs == 0 {
                    return 1;
                }
                while abs > 0 {
                    len += 1;
                    abs /= 10;
                }
                len
            } else {
                // floats: ~16 chars worst case
                16
            }
        }
        Value::String(s) => s.len() + 2, // surrounding quotes (escape not counted; close enough)
        Value::Array(arr) => {
            let mut len = 2; // []
            if !arr.is_empty() {
                len += arr.len() - 1; // commas
                for item in arr {
                    len += json_byte_len(item);
                }
            }
            len
        }
        Value::Object(obj) => {
            let mut len = 2; // {}
            if !obj.is_empty() {
                len += obj.len() - 1; // commas between pairs
                for (k, val) in obj {
                    len += k.len() + 3; // "key":
                    len += json_byte_len(val);
                }
            }
            len
        }
    }
}

// ─── Auth & Rate Limiting Middleware ─────────────────────────────────────────
/// Validates the API key from the Authorization header against the configured
/// list. Skips authentication for /health. If no API keys are configured,
/// authentication is disabled (backward compatibility). Also enforces per-IP
/// rate limiting.
async fn auth_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();

    // Skip auth for health checks
    if path == "/health" {
        return next.run(request).await;
    }

    // Extract client IP for rate limiting
    let client_ip = request
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Rate limit check
    if !state.rate_limiter.check(&client_ip) {
        logger::warn(
            "auth",
            &format!("Rate limit exceeded for IP: {}", client_ip),
        );
        return (StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded").into_response();
    }

    // If no API keys configured, allow all (backward compatibility)
    if state.cfg.api_keys.is_empty() {
        return next.run(request).await;
    }

    // Validate API key from Authorization header
    let auth_header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let key = &header[7..];
            if state.cfg.api_keys.iter().any(|k| k == key) {
                next.run(request).await
            } else {
                logger::warn(
                    "auth",
                    &format!("Invalid API key from IP: {}", client_ip),
                );
                (StatusCode::UNAUTHORIZED, "Invalid API key").into_response()
            }
        }
        _ => {
            logger::warn(
                "auth",
                &format!("Missing Authorization header from IP: {}", client_ip),
            );
            (StatusCode::UNAUTHORIZED, "Missing or invalid Authorization header")
                .into_response()
        }
    }
}

// ─── GET /v1/models ─────────────────────────────────────────────────────────
pub async fn handle_models(State(state): State<AppState>) -> impl IntoResponse {
    let all_models = state.cfg.all_models();
    let backend_meta = state.backend_meta.read().await;
    let mut models: Vec<Value> = Vec::with_capacity(all_models.len());
    for (alias, name) in &all_models {
            // Look up the optional context_length from the config model entry.
            let context_length = state
                .cfg
                .ericai_provider()
                .and_then(|p| {
                    p.models.iter().find(|m| &m.alias == alias).and_then(|m| m.context_length)
                });
            let mut entry = json!({
                "id": alias,
                "object": "model",
                "created": 1700000000,
                "owned_by": "ericai",
                "ericai_backend_id": name,
            });
            if let Some(ctx) = context_length {
                entry["context_length"] = json!(ctx);
            }
            // Pass capability tags so ericli can resolve tag:fast / tag:coding.
            // Priority: 1) live backend metadata (routing_tags from ericai api models.list)
            //           2) config.yaml `tags` field (static fallback)
            let backend_name = state.cfg.resolve_model(alias);
            let backend_tags = backend_meta.get(&backend_name).map(|m| m.routing_tags.clone());
            if let Some(tags) = backend_tags.filter(|t| !t.is_empty()) {
                entry["tags"] = json!(tags);
            } else if let Some(tags) = state.cfg.model_tags(alias) {
                entry["tags"] = json!(tags);
            }
            models.push(entry);
    }
    drop(backend_meta);

    Json(json!({"object": "list", "data": models}))
}

// ─── GET /health ────────────────────────────────────────────────────────────
pub async fn handle_health() -> impl IntoResponse {
    Json(json!({"status": "ok", "service": "ericai-proxy"}))
}

// ─── POST /v1/responses ────────────────────────────────────────────────────
pub async fn handle_responses(State(state): State<AppState>, body: axum::body::Bytes) -> Response {
    let request_json: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            logger::error("responses", &format!("Invalid JSON: {}", e));
            return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)).into_response();
        }
    };

    let model_alias = request_json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    // Resolve alias → backend model ID
    let backend_model = state.cfg.resolve_model(&model_alias);

    let is_stream = request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // Translate request
    let mut cc_payload =
        translate_request::responses_to_chat_completions(&backend_model, &request_json, is_stream);
    cc_payload = translate_request::inject_thinking(cc_payload, &backend_model);

    logger::debug("responses", &format!(
        "[responses] cc_payload keys: {} | max_tokens: {} | reasoning_effort: {} | chat_template_kwargs: {} | messages: {} | is_stream: {}",
        cc_payload.as_object().map(|o| o.keys().map(|k| k.as_str()).collect::<Vec<_>>().join(",")).unwrap_or_default(),
        cc_payload.get("max_tokens").map(|v| v.to_string()).unwrap_or("none".to_string()),
        cc_payload.get("reasoning_effort").map(|v| v.to_string()).unwrap_or("none".to_string()),
        cc_payload.get("chat_template_kwargs").map(|v| v.to_string()).unwrap_or("none".to_string()),
        cc_payload.get("messages").and_then(|m| m.as_array()).map(|a| a.len()).unwrap_or(0),
        is_stream,
    ));

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
            return (StatusCode::UNAUTHORIZED, format!("Auth failed: {}", e)).into_response();
        }
    };

    let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
    let url = format!("{}/chat/completions", base_url);

    logger::info(
        "responses",
        &format!("POST {} model={} stream={}", url, backend_model, is_stream),
    );

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
            logger::error("responses", &format!("Upstream error: {}", e));
            return (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e)).into_response();
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        logger::error(
            "responses",
            &format!(
                "Upstream returned {}: {}",
                status,
                logger::extract_error_summary(&body_text, 300)
            ),
        );
        logger::error_request(
            "v1-responses",
            &cc_payload.to_string(),
            &body_text,
            status.as_u16(),
        );
        return (status, body_text).into_response();
    }
    if is_stream {
        let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
        let mut translator =
            translate_response::ResponseTranslator::new(&response_id, Some(request_json.clone()));

        let byte_stream = resp.bytes_stream();

        // Use a channel so we can detect client disconnects.  When the
        // client (codex) drops the connection — e.g. when the user clicks
        // Stop — Axum stops polling the response body.  The channel sender
        // then fails on the next `send()`, and we abort the upstream stream.
        use tokio::sync::mpsc;
        let (tx, rx) = mpsc::channel::<Result<axum::body::Bytes, std::convert::Infallible>>(32);

        let cancel_token = tokio_util::sync::CancellationToken::new();
        let upstream_cancel = cancel_token.clone();
        let server_shutdown = state.shutdown.clone();

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
                     // Server shutdown — stop streaming gracefully
                     _ = server_shutdown.cancelled() => {
                         logger::info("responses", "Server shutting down, aborting upstream stream");
                         return;
                     }
                }
            }

           // Flush remaining buffer
           if read_pos < buffer.len() && !buffer[read_pos..].trim().is_empty() {
               let events = translator.process_line(&buffer[read_pos..]);
               for event in events {
                   if tx
                       .send(Ok(axum::body::Bytes::from(event.into_bytes())))
                       .await
                       .is_err()
                   {
                       return;
                   }
               }
           }

           // If the translator detected an in-stream upstream error (e.g.
           // vLLM returned `{"message":"upstream error","code":503}` as a
           // `data:` line), surface it as an error event instead of letting
           // the [DONE] flush emit a successful response.completed. Without
           // this, codex treats the turn as completed-but-empty (end_turn,
           // 0 tokens, usage 0%) and the user sees no response.
           if let Some((code, msg)) = translator.upstream_error() {
               logger::error("responses", &format!(
                   "Upstream in-stream error (code={}) — sending error event instead of response.completed: {}",
                   code, &msg[..msg.len().min(300)]
               ));
               let err_payload = json!({
                   "type": "error",
                   "status": code,
                   "error": {
                       "message": format!("Upstream error ({}): {}", code, msg)
                   }
               });
               let err_event = format!("event: error\ndata: {}\n\n", err_payload);
               if tx.send(Ok(axum::body::Bytes::from(err_event.into_bytes()))).await.is_err() {
                   return;
               }
               // Do NOT send [DONE] / response.completed — the error event
               // above signals codex to retry/fallback.
               return;
           }

           // Send [DONE]
           let events = translator.process_line("data: [DONE]");
           for event in events {
               if tx
                   .send(Ok(axum::body::Bytes::from(event.into_bytes())))
                   .await
                   .is_err()
               {
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
                return (StatusCode::BAD_GATEWAY, format!("Parse error: {}", e)).into_response();
            }
        };

        let response =
            translate_response::chat_completion_to_response(&full_body, Some(&request_json));

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

        // Track previous turn's per-message *signatures* (not the full
        // messages) to detect pure-append (good for vLLM prefix/KV-cache)
        // vs prefix-change (KV-cache break). Storing only signatures avoids
        // a multi-MB Value::clone of the entire messages array every turn
        // on long conversations with many concurrent clients.
        // Reset on each new WebSocket connection.
        let mut prev_cc_signatures: Option<Vec<String>> = None;
        let mut prev_cc_msg_count: usize = 0;

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

            // Check if the server is shutting down (SIGINT/SIGTERM received).
            if state.shutdown.is_cancelled() {
                logger::info("ws", &format!("[{}] Server shutting down, closing WebSocket", conn_short));
                let _ = socket.close().await;
                break 'connection;
            }

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
            //
            // Move `request_json` into `effective_request` (no clone) — for
            // long conversations this avoids a multi-MB Value::clone every turn.
            let mut effective_request = request_json;
            let response_id_for_cache = format!("resp_{}", uuid::Uuid::new_v4().simple());

            // Extract previous_response_id before any further reads/moves.
            let prev_id_owned: Option<String> = effective_request
                .get("previous_response_id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from);

            if let Some(prev_id) = prev_id_owned.as_deref() {
                if let Some(cached_req_arc) = response_request_cache.get(prev_id).cloned() {
                    let cached_req = cached_req_arc.as_ref();
                    // Cached input must be cloned (it lives behind Arc<Value>
                    // shared with the cache). New items are *taken* out of
                    // effective_request without cloning.
                    let mut merged_input: Vec<Value> = cached_req
                        .get("input")
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    let cached_count = merged_input.len();
                    let new_items: Vec<Value> = effective_request
                        .get_mut("input")
                        .and_then(|v| v.as_array_mut())
                        .map(std::mem::take)
                        .unwrap_or_default();
                    let new_count = new_items.len();
                    merged_input.extend(new_items);
                    let total_count = merged_input.len();
                    effective_request["input"] = Value::Array(merged_input);

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
                        conn_short, prev_id, cached_count, new_count, total_count
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

           // Cache this request's full effective_request under the new response_id
           // for future incremental requests. Store the effective_request (after
           // merge) so the cache always has the complete conversation.
           //
           // Move `effective_request` into the Arc — no clone. Subsequent
           // reads use `effective_request_arc.as_ref()` / `.get(...)`.
           let effective_request_arc = std::sync::Arc::new(effective_request);
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
                effective_request_arc.as_ref(),
                is_stream,
            );
            cc_payload = translate_request::inject_thinking(cc_payload, &backend_model);
            if logger::is_enabled(logger::LogLevel::Debug) {
                logger::debug("ws", &format!("[{}] cc_payload keys: {} | max_tokens: {} | reasoning_effort: {} | chat_template_kwargs: {} | messages: {}", conn_short, cc_payload.as_object().map(|o| o.keys().map(|k| k.as_str()).collect::<Vec<_>>().join(",")).unwrap_or_default(), cc_payload.get("max_tokens").map(|v| v.to_string()).unwrap_or("none".to_string()), cc_payload.get("reasoning_effort").map(|v| v.to_string()).unwrap_or("none".to_string()), cc_payload.get("chat_template_kwargs").map(|v| v.to_string()).unwrap_or("none".to_string()), cc_payload.get("messages").and_then(|m| m.as_array()).map(|a| a.len()).unwrap_or(0)));

                // Debug: dump message roles and content previews
                if let Some(msgs) = cc_payload.get("messages").and_then(|m| m.as_array()) {
                    let mut msg_summary = Vec::new();
                    for (i, msg) in msgs.iter().enumerate() {
                        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("?");
                        if let Some(tc) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                            let names: Vec<&str> = tc.iter().map(|c| c.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).unwrap_or("?")).collect();
                            msg_summary.push(format!("  [{}] {} tool_calls:{}", i, role, names.join(",")));
                        } else if role == "tool" {
                            let preview = msg.get("content").and_then(|v| v.as_str()).unwrap_or("").chars().take(80).collect::<String>();
                            msg_summary.push(format!("  [{}] tool -> {}", i, preview));
                        } else {
                            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
                            let preview = content.chars().take(80).collect::<String>();
                            msg_summary.push(format!("  [{}] {}: {}", i, role, preview));
                        }
                    }
                    logger::debug("ws", &format!("[{}] cc_payload messages:\n{}", conn_short, msg_summary.join("\n")));
                }
            }

            // Debug: diff messages vs previous turn to verify only-append semantics.
            // Logs the message count, how many are unchanged prefix, and what's new.
            //
            // We compare *signatures* (role + tool_call names | tool_call_id |
            // 40-char content prefix) rather than full Values, and store only
            // the signature vec across turns. This avoids a per-turn deep
            // clone of the multi-MB messages array on long conversations —
            // a significant memory churn point under high concurrency.
            {
                let msgs = cc_payload.get("messages").and_then(|m| m.as_array());
                if let Some(msgs) = msgs {
                    let curr_len = msgs.len();
                    let prev_len = prev_cc_msg_count;

                    // Build a compact signature of each message. Two messages
                    // with the same signature are treated as identical for
                    // KV-prefix purposes.
                    let sig = |m: &Value| -> String {
                        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("?");
                        if m.get("tool_calls").is_some() {
                            let names: Vec<&str> = m.get("tool_calls")
                                .and_then(|v| v.as_array())
                                .map(|tc| tc.iter().map(|c| c.get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|n| n.as_str()).unwrap_or("?")).collect())
                                .unwrap_or_default();
                            format!("{}+tc({})", role, names.join(","))
                        } else if role == "tool" {
                            let call_id = m.get("tool_call_id").and_then(|v| v.as_str()).unwrap_or("");
                            format!("tool[{}]", call_id.chars().take(12).collect::<String>())
                        } else {
                            let c = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
                            format!("{}:'{}'", role, c.chars().take(40).collect::<String>())
                        }
                    };

                    // Build current-turn signatures once. We need them for
                    // storage anyway, and the diff itself reuses them.
                    let curr_sigs: Vec<String> = msgs.iter().map(&sig).collect();

                    // Determine the shared prefix length by signature compare.
                    let shared = if let Some(ref prev_sigs) = prev_cc_signatures {
                        let mut s = 0;
                        for (i, ps) in prev_sigs.iter().enumerate() {
                            if i >= curr_len { break; }
                            if ps == &curr_sigs[i] {
                                s = i + 1;
                            } else {
                                break;
                            }
                        }
                        s
                    } else { 0 };
                    let new_count = curr_len.saturating_sub(shared);

                    let kv_break = shared < prev_len && prev_len > 0;
                    let debug_on = logger::is_enabled(logger::LogLevel::Debug);

                    if shared == prev_len && new_count > 0 && shared > 0 {
                        // Pure append — ideal case (debug only)
                        if debug_on {
                            let new_items: &[String] = &curr_sigs[shared..];
                            logger::debug("ws", &format!(
                                "[{}] msg_diff: PURE APPEND | prefix={} new={} total={} | +{}",
                                conn_short, shared, new_count, curr_len, new_items.join(", ")
                            ));
                        }
                    } else if kv_break {
                        // Prefix changed — KV-cache break! Always log (warn).
                        let removed: &[String] = if let Some(ref prev_sigs) = prev_cc_signatures {
                            &prev_sigs[shared..]
                        } else { &[] };
                        let added: &[String] = if shared < curr_len {
                            &curr_sigs[shared..]
                        } else { &[] };
                        logger::warn("ws", &format!(
                            "[{}] msg_diff: PREFIX CHANGED (KV-cache break!) | shared={} removed={} added={} total={}",
                            conn_short, shared, removed.len(), added.len(), curr_len
                        ));
                        logger::warn("ws", &format!(
                            "[{}]   removed: [{}]", conn_short, removed.join(", ")
                        ));
                        logger::warn("ws", &format!(
                            "[{}]   added:   [{}]", conn_short, added.join(", ")
                        ));
                    } else if prev_len == 0 {
                        // First turn (debug only)
                        if debug_on {
                            logger::debug("ws", &format!(
                                "[{}] msg_diff: FIRST TURN | total={}", conn_short, curr_len
                            ));
                        }
                    } else if debug_on {
                        // shared == prev_len, new_count == 0 (no change?) or edge case
                        logger::debug("ws", &format!(
                            "[{}] msg_diff: no change | prev={} curr={} shared={}",
                            conn_short, prev_len, curr_len, shared
                        ));
                    }

                    // Store signatures (small) for next turn instead of
                    // cloning the full Vec<Value> (multi-MB on long sessions).
                    prev_cc_signatures = Some(curr_sigs);
                    prev_cc_msg_count = curr_len;
                }
            }

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
                effective_request_arc.get("input").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0)));

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
                                    // Hot path: only allocate the preview string when
                                    // debug logging is actually on. With multiple codex
                                    // clients streaming, this saves an O(n) chars().collect()
                                    // per upstream SSE line.
                                    if logger::is_enabled(logger::LogLevel::Debug) {
                                        let raw_line: String = buffer[read_pos..pos].chars().take(300).collect();
                                        logger::debug("ws", &format!("[{}] ← upstream line: {}", conn_short, raw_line));
                                    }
                                    read_pos = pos + 1;
                                   for event in events {
                                       stream_event_count += 1;
                                       if logger::is_enabled(logger::LogLevel::Debug) {
                                           let preview: String = event.chars().take(500).collect();
                                           logger::debug("ws", &format!("[{}] → codex #{} (stream): {}", conn_short, stream_event_count, preview));
                                       }
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
                    // Server shutdown: send a Close frame and exit the stream loop.
                    // Without this, axum 0.7's with_graceful_shutdown waits forever
                    // for upgraded WebSocket connections that never close on their own.
                    _ = state.shutdown.cancelled() => {
                        logger::info("ws", &format!("[{}] Server shutting down, closing WebSocket during stream", conn_short));
                        let _ = socket.close().await;
                        break 'connection;
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

           // If the translator detected an in-stream upstream error (e.g.
           // vLLM returned `{"message":"upstream error","code":503}` as a
           // `data:` line), surface it as an error event instead of letting
           // the [DONE] flush emit a successful response.completed. Without
           // this, codex treats the turn as completed-but-empty (end_turn,
           // 0 tokens, usage 0%) and the user sees no response.
           if let Some((code, msg)) = translator.upstream_error() {
               logger::error("ws", &format!(
                   "[{}] Upstream in-stream error (code={}) — sending error event instead of response.completed: {}",
                   conn_short, code, &msg[..msg.len().min(300)]
               ));
               let err_payload = json!({
                   "type": "error",
                   "status": code,
                   "error": {
                       "message": format!("Upstream error ({}): {}", code, msg)
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
                   if logger::is_enabled(logger::LogLevel::Debug) {
                       let preview: String = event.chars().take(500).collect();
                       logger::debug("ws", &format!("[{}] → codex #{} (flush): {}", conn_short, stream_event_count, preview));
                   }
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
                if logger::is_enabled(logger::LogLevel::Debug) {
                    let preview: String = event.chars().take(500).collect();
                    logger::debug("ws", &format!("[{}] → codex (DONE flush): {}", conn_short, preview));
                }
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
            //
            // Perf note: Three strong refs to the cached request Arc are
            // alive at this point — the cache entry, `effective_request_arc`,
            // and `translator.request_json` (an Arc clone). `Arc::make_mut`
            // on a non-unique Arc deep-clones the inner `Value`, which for
            // long conversations is multi-MB on EVERY turn. We avoid that
            // by:
            //   1. Drop translator + effective_request_arc first.
            //   2. `remove()` the cached entry from the map (transfers
            //      ownership; the local Arc is now the sole strong ref).
            //   3. `Arc::make_mut` is then O(1) (uniquely owned, no clone).
            //   4. Mutate, reinsert. `cache_insertion_order` is unchanged
            //      (same key, just round-trips through the map).
            let output_items = translator.get_output_items();
            if logger::is_enabled(logger::LogLevel::Debug) {
                logger::debug("ws", &format!("[{}] get_output_items: {} items — types: [{}]", conn_short, output_items.len(), output_items.iter().map(|item| item.get("type").and_then(|v| v.as_str()).unwrap_or("?")).collect::<Vec<_>>().join(", ")));
            }
            // Drop other strong refs to the cached request Arc BEFORE we
            // take it out of the cache. After these drops, the Arc inside
            // the HashMap is the only remaining strong ref.
            drop(translator);
            drop(effective_request_arc);
            if !output_items.is_empty() {
                if let Some(mut cached_arc) = response_request_cache.remove(&response_id_for_cache) {
                    // Now uniquely owned → make_mut is O(1) (no deep clone).
                    let cached = Arc::make_mut(&mut cached_arc);
                    if let Some(input_arr) = cached.get_mut("input").and_then(|v| v.as_array_mut()) {
                        let before_len = input_arr.len();
                        input_arr.extend(output_items);
                        logger::info("ws", &format!(
                            "[{}] Appended {} assistant output items to cached request (input: {} → {})",
                            conn_short, input_arr.len() - before_len, before_len, input_arr.len()
                        ));
                    }
                    // Put the updated Arc back into the cache. The key is
                    // already in cache_insertion_order from the earlier
                    // insert, so order is preserved.
                    response_request_cache.insert(response_id_for_cache.clone(), cached_arc);
                }
            }

            // Turn complete — loop back to recv() the next request on the
            // SAME connection.  No log spam; only log on actual close.
            logger::debug("ws", &format!("[{}] Turn complete ({} events sent), looping back to recv() next request", conn_short, stream_event_count));
        }

        logger::info("ws", &format!("[{}] WebSocket connection closed", conn_short));
    })
}

/// Rewrite `delta.reasoning` → `delta.content` in an SSE line.
///
/// GLM-5.2 with thinking enabled streams output via `delta.reasoning`
/// instead of `delta.content`. CCR's OpenAI parser only reads `content`,
/// so we rewrite the SSE data to move reasoning into content.
///
/// This is a lightweight line-level transform — it parses each `data: `
/// line as JSON, checks for `delta.reasoning`, and if present, moves it
/// to `delta.content` (removing the `reasoning` key). Non-data lines
/// (comments, empty lines, `[DONE]`) are passed through unchanged.
fn rewrite_reasoning_to_content(line: &str) -> String {
    let trimmed = line.trim();
    if !trimmed.starts_with("data: ") || trimmed == "data: [DONE]" {
        return line.to_string();
    }

    let json_str = &trimmed[6..];
    let mut v: Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return line.to_string(), // not valid JSON, pass through
    };

    // Check if any choice has a delta with reasoning
    let mut changed = false;
    if let Some(choices) = v.get_mut("choices").and_then(|c| c.as_array_mut()) {
        for choice in choices.iter_mut() {
            if let Some(delta) = choice.get_mut("delta").and_then(|d| d.as_object_mut()) {
                if let Some(reasoning) = delta.remove("reasoning") {
                    // Move reasoning → content (append if content already exists)
                    let existing = delta
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("");
                    let reasoning_str = reasoning.as_str().unwrap_or("");
                    let combined = format!("{}{}", existing, reasoning_str);
                    delta.insert("content".to_string(), json!(combined));
                    changed = true;
                }
            }
        }
    }

    if changed {
        format!("data: {}", v)
    } else {
        line.to_string()
    }
}

/// Markers that identify a "text-only, no tools" turn.
///
/// Claude Code CLI builds its compaction/summarization prompt with a literal
/// preamble instructing the model to answer with prose and never call a tool,
/// yet it still sends the full `tools` array in the same request (it relies on
/// rejecting tool calls client-side). GLM-5.x happily ignores the prose
/// instruction and emits a `tool_use` block instead of text. Claude Code then
/// extracts no assistant text and aborts /compact with
/// "summarization produced empty response".
///
/// We therefore detect the instruction and drop the tools for that single
/// request, which forces the backend to produce text. Matching on the
/// instruction (rather than on `thinking: {"type": "disabled"}`) keeps normal
/// tool use intact for users who simply run with thinking turned off — e.g.
/// non-reasoning models such as Qwen2.5-14B-1M.
const TEXT_ONLY_MARKERS: [&str; 2] = [
    "Do NOT call any tools",
    "Tool calls will be rejected and you will fail the task",
];

/// Append all plain-text fragments of an Anthropic/OpenAI `content` value.
fn collect_message_text(content: &Value, out: &mut String) {
    match content {
        Value::String(s) => out.push_str(s),
        Value::Array(blocks) => {
            for block in blocks {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    out.push_str(t);
                    out.push('\n');
                }
            }
        }
        _ => {}
    }
}

/// True when a user message asks for a text-only reply with no tool calls.
///
/// Works for both OpenAI-shaped and Anthropic-shaped message arrays, since
/// `content` may be a bare string or a list of typed blocks in either.
pub fn is_text_only_turn(request: &Value) -> bool {
    let Some(messages) = request.get("messages").and_then(|m| m.as_array()) else {
        return false;
    };
    let mut text = String::new();
    for msg in messages {
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        if let Some(content) = msg.get("content") {
            collect_message_text(content, &mut text);
        }
    }
    TEXT_ONLY_MARKERS.iter().any(|m| text.contains(m))
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
            return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)).into_response();
        }
    };

    let model = request_json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let backend_model = state.cfg.resolve_model(&model);

    // Resolve alias in the request
    request_json["model"] = json!(backend_model);

    // Translate reasoning_effort → chat_template_kwargs for GLM models.
    // CCR / Claude Code sends effort via the request body (e.g. from --effort flag).
    // GLM-5.x in thinking mode returns output via the `reasoning` field, leaving
    // `content` empty. CCR's OpenAI parser only reads `content`, so it sees an
    // empty response and returns 502 "no text output".
    //
    // For the chat/completions passthrough (used by CCR), we need thinking
    // DISABLED by default so content goes into the `content` field. But if the
    // user explicitly selects an effort level (low/medium/high/xhigh/max), we
    // enable thinking and set the appropriate reasoning_effort — the CCR
    // adapter can then extract the reasoning content.
    //
    // The /v1/responses handler uses inject_thinking() which handles this
    // properly for codex (codex handles reasoning blocks natively).
    let thinking_enabled;
    {
        let lower_model = backend_model.to_lowercase();
        let mut effort = request_json
            .get("reasoning_effort")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase());

        // Remove reasoning_effort if it's not a recognized value
        if let Some(ref e) = effort {
            match e.as_str() {
                "low" | "medium" | "high" | "xhigh" | "max" => {}
                _ => {
                    request_json
                        .as_object_mut()
                        .map(|o| o.remove("reasoning_effort"));
                    effort = None;
                }
            }
        }

        // Honor the Anthropic client's `thinking` field. The CCR gateway keeps
        // `thinking: {"type": "disabled"}` in the body when translating
        // Anthropic → chat/completions (and drops reasoning_effort), so this is
        // the signal Claude Code's /compact turn arrives with on this route.
        let client_thinking_disabled = request_json
            .get("thinking")
            .and_then(|t| t.get("type"))
            .and_then(|v| v.as_str())
            .map(|s| s == "disabled")
            .unwrap_or(false);

        // A summarization turn must yield prose, not a tool call. Drop the
        // tools so the backend cannot emit tool_use, and keep thinking off so
        // the answer lands in `content` where the client looks for it.
        let text_only = is_text_only_turn(&request_json);
        if text_only {
            effort = None;
        }

        thinking_enabled = effort.is_some();

        if let Some(obj) = request_json.as_object_mut() {
            // `thinking` is an Anthropic-only field; the backend's
            // chat/completions endpoint has no use for it.
            obj.remove("thinking");
            if text_only {
                obj.remove("tools");
                obj.remove("tool_choice");
                obj.remove("reasoning_effort");
            }
            let kwargs = obj
                .entry("chat_template_kwargs")
                .or_insert_with(|| json!({}));
            if let Some(kw) = kwargs.as_object_mut() {
                if text_only || client_thinking_disabled {
                    // Text-only turn, or the client explicitly disabled
                    // thinking — either way, output must go to `content`.
                    kw.insert("enable_thinking".to_string(), json!(false));
                } else if let Some(ref effort) = effort {
                    // User selected an effort level — enable thinking
                    if lower_model.contains("glm") {
                        let enable = effort != "low";
                        kw.entry("enable_thinking").or_insert(json!(enable));
                    } else if lower_model.contains("qwen3") || lower_model.contains("qwen-3") {
                        let enable = matches!(effort.as_str(), "low" | "medium" | "high" | "xhigh");
                        kw.entry("enable_thinking").or_insert(json!(enable));
                    }
                } else {
                    // No effort specified — disable thinking (default behavior
                    // for CCR compatibility, so content goes to `content` field)
                    kw.entry("enable_thinking").or_insert(json!(false));
                }
            }
        }

        if text_only || client_thinking_disabled {
            logger::info(
                "chat",
                &format!(
                    "text-only turn (marker={}, thinking_disabled={}) — tools stripped={}, thinking off",
                    text_only, client_thinking_disabled, text_only
                ),
            );
        }
    }

    let token = match state.token_mgr.get_token().await {
        Ok(t) => t,
        Err(e) => {
            logger::error("chat", &format!("Auth failed: {}", e));
            return (StatusCode::UNAUTHORIZED, format!("Auth failed: {}", e)).into_response();
        }
    };

    let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
    let url = format!("{}/chat/completions", base_url);

    logger::info(
        "chat",
        &format!("Passthrough POST {} model={}", url, backend_model),
    );

    let resp = state
        .http_client
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

            if request_json
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                // When thinking is enabled (reasoning_effort set), GLM-5.2
                // streams output via `delta.reasoning` instead of
                // `delta.content`. CCR's OpenAI parser only reads `content`,
                // so it sees an empty response → 502.
                //
                // We intercept the SSE stream and rewrite `delta.reasoning`
                // → `delta.content` so CCR can read the output. We also
                // handle non-streaming responses the same way (below).
                let stream = r.bytes_stream();

                // Both the thinking-enabled and passthrough paths use a
                // channel-based stream so we can inject SSE keepalive
                // comments every 15 seconds. Without keepalive, CCR may
                // time out while waiting for GLM-5.2 to start generating
                // on large contexts (prefill can take 30-60+ seconds),
                // causing "Conversation interrupted" in Claude Code.
                use tokio::sync::mpsc;
                let (tx, rx) =
                    mpsc::channel::<Result<axum::body::Bytes, std::convert::Infallible>>(32);

                tokio::spawn(async move {
                   let mut stream = stream;
                   let mut buffer = String::new();
                   let mut read_pos: usize = 0;
                   let mut sse_keepalive = tokio::time::interval(Duration::from_secs(15));
                   sse_keepalive.tick().await; // consume immediate first tick
                   // In-stream error detection: vLLM/TGI backends sometimes return
                   // error objects (e.g. {"error":{"message":"...","code":429}}) as
                   // `data:` lines with no `choices` key. Without this, the
                   // passthrough forwards them verbatim and the client (ericli)
                   // silently skips them, then treats the stream end as a successful
                   // empty response → premature end_turn. When detected, we inject
                   // an SSE error event so the client can retry/fallback.
                   let mut upstream_error: Option<(u16, String)> = None;

                   loop {
                       tokio::select! {
                           chunk_result = stream.next() => {
                               match chunk_result {
                                   Some(Ok(chunk)) => {
                                       buffer.push_str(&String::from_utf8_lossy(&chunk));
                                      while let Some(rel_pos) = buffer[read_pos..].find('\n') {
                                          let pos = read_pos + rel_pos;
                                          let line = &buffer[read_pos..pos];
                                          read_pos = pos + 1;
                                          // Check for in-stream upstream errors BEFORE
                                          // forwarding the line. This mirrors the
                                          // /v1/responses path's extract_upstream_error
                                          // detection so ericli (which uses this
                                          // passthrough) gets the same error surfacing
                                          // as codex-acp.
                                          if upstream_error.is_none() {
                                              if let Some(stripped) = line.trim().strip_prefix("data: ") {
                                                  if stripped != "[DONE]" {
                                                      if let Ok(parsed) = serde_json::from_str::<Value>(stripped) {
                                                          if let Some((code, msg)) = translate_response::extract_upstream_error(&parsed) {
                                                              logger::error(
                                                                  "chat",
                                                                  &format!(
                                                                      "Upstream in-stream error (code={}) — injecting SSE error event: {}",
                                                                      code, &msg[..msg.len().min(300)]
                                                                  ),
                                                              );
                                                              upstream_error = Some((code, msg));
                                                              // Don't forward the error object — the
                                                              // client would skip it (no `choices`).
                                                              continue;
                                                          }
                                                      }
                                                  }
                                              }
                                          }
                                          let output = if thinking_enabled {
                                              rewrite_reasoning_to_content(line)
                                          } else {
                                              line.to_string()
                                          };
                                           let framed = format!("{}\n", output);
                                           if tx.send(Ok(axum::body::Bytes::from(framed))).await.is_err() {
                                              return;
                                          }
                                      }
                                       if read_pos > 0 {
                                           buffer.drain(..read_pos);
                                           read_pos = 0;
                                       }
                                   }
                                   Some(Err(e)) => {
                                       logger::error("chat", &format!("Stream error: {}", e));
                                       break;
                                   }
                                   None => break,
                               }
                           }
                           _ = sse_keepalive.tick() => {
                               if tx.send(Ok(KEEPALIVE_BYTES.clone())).await.is_err() {
                                   return;
                               }
                           }
                       }
                   }
                   // Flush remaining buffer
                  if read_pos < buffer.len() {
                      let remaining = &buffer[read_pos..];
                      if !remaining.trim().is_empty() {
                          let output = if thinking_enabled {
                              rewrite_reasoning_to_content(remaining)
                          } else {
                              remaining.to_string()
                          };
                           let framed = format!("{}\n", output);
                           let _ = tx.send(Ok(axum::body::Bytes::from(framed))).await;
                      }
                  }
                  // If an in-stream upstream error was detected, send an SSE
                  // error event instead of letting the stream end silently.
                  // Without this, the client sees a clean stream end with no
                  // content and treats it as a successful empty response.
                  if let Some((code, msg)) = upstream_error {
                      let err_payload = json!({
                          "type": "error",
                          "status": code,
                          "error": {
                              "message": format!("Upstream error ({}): {}", code, msg)
                          }
                      });
                      let err_event = format!("event: error\ndata: {}\n\n", err_payload);
                      let _ = tx.send(Ok(axum::body::Bytes::from(err_event.into_bytes()))).await;
                  }
               });

                let translated_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                let body = Body::from_stream(translated_stream);

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
                // Non-streaming: if thinking was enabled, GLM-5.2 may return
                // content in `reasoning` field with empty `content`. Move it.
                let body_bytes = r.bytes().await.unwrap_or_default();
                let body_bytes = if thinking_enabled {
                    if let Ok(mut v) = serde_json::from_slice::<Value>(&body_bytes) {
                        if let Some(choices) = v.get_mut("choices").and_then(|c| c.as_array_mut()) {
                            for choice in choices.iter_mut() {
                                if let Some(msg) = choice.get_mut("message").and_then(|m| m.as_object_mut()) {
                                    let content_empty = msg.get("content")
                                        .and_then(|c| c.as_str())
                                        .map(|s| s.is_empty())
                                        .unwrap_or(true);
                                    if content_empty {
                                        if let Some(reasoning) = msg.remove("reasoning") {
                                            msg.insert("content".to_string(), reasoning);
                                        }
                                    }
                                }
                            }
                        }
                        serde_json::to_vec(&v).unwrap_or_else(|_| body_bytes.to_vec())
                    } else {
                        body_bytes.to_vec()
                    }
                } else {
                    body_bytes.to_vec()
                };
                let mut response = Response::builder().status(status);
                for (k, v) in &headers {
                    if k != "content-encoding" && k != "transfer-encoding" {
                        response = response.header(k, v);
                    }
                }
                response.body(Body::from(body_bytes)).unwrap()
            }
        }
        Err(e) => {
            logger::error("chat", &format!("Upstream error: {}", e));
            (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e)).into_response()
        }
    }
}

// ─── POST /v1/messages (Anthropic Messages API → OpenAI Chat Completions) ───
/// Native Anthropic Messages API endpoint. Translates Anthropic format to
/// OpenAI Chat Completions internally, then translates the response back.
/// This bypasses CCR (Claude Code Router) entirely, fixing broken streaming.
pub async fn handle_messages(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let request_json: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            logger::error("messages", &format!("Invalid JSON: {}", e));
            return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)).into_response();
        }
    };

    let model = request_json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let backend_model = state.cfg.resolve_model(&model);
    let is_stream = request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Translate Anthropic → OpenAI Chat Completions
    let mut cc_request = translate_anthropic::anthropic_to_chat_completions(
        &backend_model,
        &request_json,
        is_stream,
    );

    // Inject enable_thinking for GLM models (same logic as handle_chat_completions)
    let lower_model = backend_model.to_lowercase();
    let effort = cc_request
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase())
        .filter(|e| matches!(e.as_str(), "low" | "medium" | "high" | "xhigh" | "max"));

    // Honor the Anthropic client's `thinking` field. Claude Code CLI sends
    // `thinking: { type: "disabled" }` for compaction/summarization turns. The
    // CLI still attaches `reasoning_effort` to the body (it's how the model
    // choice is conveyed), so keying enable_thinking off effort alone would
    // force thinking ON for a turn the client explicitly disabled — producing
    // reasoning-only output that yields no assistant text message and breaks
    // /compact ("no assistant message in summarization response"). When the
    // client disables thinking, we must disable it too, regardless of effort.
    let client_thinking_disabled = request_json
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(|v| v.as_str())
        .map(|s| s == "disabled")
        .unwrap_or(false);


    // A summarization turn must yield prose, not a tool call. Claude Code's
    // compact prompt says "Do NOT call any tools" but still ships the tool
    // schemas; GLM-5.x calls FileReadTool anyway, leaving no text block and
    // failing /compact with "summarization produced empty response". Drop the
    // tools for that request only — gating on the instruction rather than on
    // `thinking: disabled` keeps tool use working for non-reasoning models.
    let text_only = is_text_only_turn(&request_json);
    if text_only {
        if let Some(obj) = cc_request.as_object_mut() {
            obj.remove("tools");
            obj.remove("tool_choice");
            obj.remove("reasoning_effort");
        }
    }
    if let Some(obj) = cc_request.as_object_mut() {
        let kwargs = obj
            .entry("chat_template_kwargs")
            .or_insert_with(|| json!({}));
        if let Some(kw) = kwargs.as_object_mut() {
            if client_thinking_disabled || text_only {
                // Client explicitly disabled thinking, or this is a text-only
                // summarization turn — honor it.
                kw.insert("enable_thinking".to_string(), json!(false));
            } else if let Some(ref effort) = effort {
                if lower_model.contains("glm") {
                    let enable = effort != "low";
                    kw.entry("enable_thinking").or_insert(json!(enable));
                } else if lower_model.contains("qwen3") || lower_model.contains("qwen-3") {
                    let enable = matches!(effort.as_str(), "low" | "medium" | "high" | "xhigh");
                    kw.entry("enable_thinking").or_insert(json!(enable));
                }
            } else {
                kw.entry("enable_thinking").or_insert(json!(false));
            }
        }
    }

    let token = match state.token_mgr.get_token().await {
        Ok(t) => t,
        Err(e) => {
            logger::error("messages", &format!("Auth failed: {}", e));
            return (StatusCode::UNAUTHORIZED, format!("Auth failed: {}", e)).into_response();
        }
    };

    let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
    let url = format!("{}/chat/completions", base_url);

    logger::info(
        "messages",
        &format!("POST {} model={} stream={}", url, backend_model, is_stream),
    );

    // Turn-debug: log request size breakdown to diagnose rapid context growth.
    if state.cfg.log_turn_debug {
        let system_len = request_json
            .get("system")
            .map(|s| serde_json::to_string(s).unwrap_or_default().len())
            .unwrap_or(0);
        let anthropic_msg_count = request_json
            .get("messages")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let openai_msg_count = cc_request
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let msgs_bytes = serde_json::to_string(&cc_request["messages"]).unwrap_or_default().len();
        let tools_count = cc_request
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let tools_bytes = cc_request
            .get("tools")
            .map(|t| serde_json::to_string(t).unwrap_or_default().len())
            .unwrap_or(0);
        let total_req_bytes = serde_json::to_string(&cc_request).unwrap_or_default().len();
        let estimated_tokens = estimate_tokens(&cc_request);
        let context_window = get_context_window(&state.model_catalog, &model);
        logger::info(
            "messages",
            &format!(
                "[turn-debug] model={} | system_bytes={} | anthropic_msgs={} | openai_msgs={} | msgs_bytes={} | tools_count={} | tools_bytes={} | total_req_bytes={} | ~est_tokens={} | context_window={}",
                backend_model,
                system_len,
                anthropic_msg_count,
                openai_msg_count,
                msgs_bytes,
                tools_count,
                tools_bytes,
                total_req_bytes,
                estimated_tokens,
                context_window,
            ),
        );
    } else {
        logger::debug(
            "messages",
            &format!("cc_request keys: {} | stream_options: {}",
                cc_request.as_object().map(|o| o.keys().map(|k| k.as_str()).collect::<Vec<_>>().join(",")).unwrap_or_default(),
                cc_request.get("stream_options").map(|v| v.to_string()).unwrap_or("MISSING".to_string())),
        );
    }

    let resp = state
        .http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Accept", "text/event-stream")
        .json(&cc_request)
        .send()
        .await;

    match resp {
        Ok(r) => {
            let status = r.status();

            // If the upstream returned a non-success status (e.g. 429 rate
            // limit on large inputs, 5xx), do NOT stream the error body
            // through the translator — vLLM error responses are JSON objects
            // (no SSE `choices`), so translate_line() would skip every line
            // and finish() would synthesize an EMPTY assistant message. The
            // client (CCR/Claude Code) then sees a silent end_turn with ~0
            // tokens — the "GLM returns empty / ends prematurely" symptom.
            // Instead surface a real Anthropic error so the client can react.
            if !status.is_success() {
                let body_text = r.bytes().await.unwrap_or_default();
                let body_str = String::from_utf8_lossy(&body_text);
                let summary = logger::extract_error_summary(&body_str, 500);
                logger::error(
                    "messages",
                    &format!("Upstream {}: {}", status, summary),
                );
                if status.as_u16() == 429 || status.as_u16() >= 500 {
                    logger::error_request(
                        "messages",
                        &cc_request.to_string(),
                        &body_str,
                        status.as_u16(),
                    );
                }

                // A request that exceeds the model's context window is a
                // client-side sizing problem, whatever status the backend
                // chose (vLLM reports it as a 500). Re-emit it as Anthropic's
                // 400 invalid_request_error with the canonical "prompt is too
                // long: N tokens > M maximum" wording: Claude Code CLI matches
                // that phrasing and responds by compacting the conversation
                // and retrying, turning a dead session into a recovery.
                let (http_status, err_type, err_message) =
                    match translate_response::parse_context_length_error(&body_str) {
                        Some((total, max)) => {
                            logger::info(
                                "messages",
                                &format!(
                                    "Context window exceeded ({} > {}) — reporting as prompt-too-long so the client can compact",
                                    total, max
                                ),
                            );
                            (
                                StatusCode::BAD_REQUEST,
                                "invalid_request_error",
                                format!("prompt is too long: {} tokens > {} maximum", total, max),
                            )
                        }
                        None => {
                            let code = status.as_u16();
                            let mapped = StatusCode::from_u16(code)
                                .unwrap_or(StatusCode::BAD_GATEWAY);
                            let ty = if code == 429 {
                                "rate_limit_error"
                            } else {
                                "api_error"
                            };
                            (mapped, ty, format!("Upstream {}: {}", status, summary))
                        }
                    };

                let err_payload = json!({
                    "type": "error",
                    "error": {
                        "type": err_type,
                        "message": err_message
                    }
                });
                // Return a real HTTP error status with a JSON body. Returning
                // 200 here breaks non-streaming clients: the Anthropic SDK
                // parses the body as a Message, finds no content, and reports
                // "API returned an empty or malformed response (HTTP 200)".
                // A non-2xx JSON error is correct for streaming clients too —
                // nothing has been streamed yet, so the SDK raises it as an
                // API error rather than starting to parse SSE.
                return Response::builder()
                    .status(http_status)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&err_payload).unwrap_or_default()))
                    .unwrap();
            }

            if is_stream {
                let stream = r.bytes_stream();
                use tokio::sync::mpsc;
                let (tx, rx) =
                    mpsc::channel::<Result<axum::body::Bytes, std::convert::Infallible>>(32);

                let model_for_translator = backend_model.clone();
                tokio::spawn(async move {
                    let mut translator =
                        translate_anthropic::AnthropicStreamTranslator::new(&model_for_translator);
                    let mut stream = stream;
                    let mut buffer = String::new();
                    let mut read_pos: usize = 0;
                    let mut sse_keepalive = tokio::time::interval(Duration::from_secs(15));
                    sse_keepalive.tick().await;
                    // In-stream upstream errors arrive with HTTP 200, so the
                    // status check above cannot catch them. Track the first one
                    // and surface it instead of a synthesized empty message.
                    let mut upstream_error: Option<(u16, String)> = None;

                    loop {
                        tokio::select! {
                            chunk_result = stream.next() => {
                                match chunk_result {
                                    Some(Ok(chunk)) => {
                                        buffer.push_str(&String::from_utf8_lossy(&chunk));
                                        while let Some(rel_pos) = buffer[read_pos..].find('\n') {
                                            let pos = read_pos + rel_pos;
                                            let line = &buffer[read_pos..pos];
                                            read_pos = pos + 1;
                                            // Once an error is seen, stop
                                            // translating. The backend still
                                            // sends `data: [DONE]` afterwards,
                                            // which would drive finish() and
                                            // synthesize the very empty
                                            // assistant message we are avoiding.
                                            if upstream_error.is_some() {
                                                continue;
                                            }
                                            // Check for an in-stream error before handing
                                            // the line to the translator. The backend
                                            // returns HTTP 200 with an SSE error event
                                            // when prompt_tokens + max_tokens exceeds the
                                            // model's context length; the line has no
                                            // `choices`, so the translator would skip it
                                            // and finish() would synthesize an empty
                                            // assistant message, hiding the failure.
                                            if let Some(stripped) = line.trim().strip_prefix("data: ") {
                                                if stripped != "[DONE]" {
                                                    if let Ok(parsed) = serde_json::from_str::<Value>(stripped) {
                                                        if let Some((code, msg)) = translate_response::extract_upstream_error(&parsed) {
                                                            logger::error(
                                                                "messages",
                                                                &format!(
                                                                    "Upstream in-stream error (code={}) — sending SSE error event instead of an empty message: {}",
                                                                    code, &msg[..msg.len().min(300)]
                                                                ),
                                                            );
                                                            upstream_error = Some((code, msg));
                                                            // Don't translate the error object.
                                                            continue;
                                                        }
                                                    }
                                                }
                                            }
                                            let events = translator.translate_line(line);
                                            for event in events {
                                                if tx.send(Ok(axum::body::Bytes::from(event))).await.is_err() {
                                                    return;
                                                }
                                            }
                                        }
                                        if read_pos > 0 {
                                            buffer.drain(..read_pos);
                                            read_pos = 0;
                                        }
                                    }
                                    Some(Err(e)) => {
                                        logger::error("messages", &format!("Stream error: {}", e));
                                        break;
                                    }
                                    None => break,
                                }
                            }
                            _ = sse_keepalive.tick() => {
                                if tx.send(Ok(KEEPALIVE_BYTES.clone())).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                    if read_pos < buffer.len() {
                        let remaining = &buffer[read_pos..];
                        if !remaining.trim().is_empty() && upstream_error.is_none() {
                            let events = translator.translate_line(remaining);
                            for event in events {
                                let _ = tx.send(Ok(axum::body::Bytes::from(event))).await;
                            }
                        }
                    }
                    if let Some((code, msg)) = upstream_error {
                        // Close any block left open by partial output, then
                        // report the failure. Deliberately skips finish() so no
                        // empty assistant message is synthesized.
                        for event in translator.abort_for_error() {
                            let _ = tx.send(Ok(axum::body::Bytes::from(event))).await;
                        }
                        let err_type = if code == 429 {
                            "rate_limit_error"
                        } else {
                            "api_error"
                        };
                        let err_payload = json!({
                            "type": "error",
                            "error": {
                                "type": err_type,
                                "message": format!("Upstream error ({}): {}", code, msg)
                            }
                        });
                        let sse = format!("event: error\ndata: {}\n\n", err_payload);
                        let _ = tx.send(Ok(axum::body::Bytes::from(sse))).await;
                    } else {
                        let finish_events = translator.translate_line("data: [DONE]");
                        for event in finish_events {
                            let _ = tx.send(Ok(axum::body::Bytes::from(event))).await;
                        }
                    }
                });

                let translated_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                let body = Body::from_stream(translated_stream);

                Response::builder()
                    .status(status)
                    .header("content-type", "text/event-stream")
                    .body(body)
                    .unwrap()
            } else {
                let body_bytes = r.bytes().await.unwrap_or_default();
                let cc_response: Value = serde_json::from_slice(&body_bytes).unwrap_or(json!({}));
                // Same trap as the streaming branch: the backend can return
                // HTTP 200 with an error object instead of `choices`. Translating
                // it would yield an empty assistant message, so report the error.
                if let Some((code, msg)) = translate_response::extract_upstream_error(&cc_response) {
                    logger::error(
                        "messages",
                        &format!(
                            "Upstream error object in 200 response (code={}): {}",
                            code,
                            &msg[..msg.len().min(300)]
                        ),
                    );
                    let err_type = if code == 429 {
                        "rate_limit_error"
                    } else {
                        "api_error"
                    };
                    let err_payload = json!({
                        "type": "error",
                        "error": {
                            "type": err_type,
                            "message": format!("Upstream error ({}): {}", code, msg)
                        }
                    });
                    // Non-2xx, not 200: a 200 with an error body makes the
                    // Anthropic SDK report "API returned an empty or malformed
                    // response (HTTP 200)" instead of the real cause.
                    let http_status = StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_GATEWAY);
                    return Response::builder()
                        .status(http_status)
                        .header("content-type", "application/json")
                        .body(Body::from(serde_json::to_vec(&err_payload).unwrap_or_default()))
                        .unwrap();
                }
                let anthropic_response =
                    translate_anthropic::chat_completions_to_anthropic(&backend_model, &cc_response);
                Response::builder()
                    .status(status)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&anthropic_response).unwrap_or_default()))
                    .unwrap()
            }
        }
        Err(e) => {
            logger::error("messages", &format!("Upstream error: {}", e));
            let error = json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": format!("Upstream error: {}", e)
                }
            });
            (
                StatusCode::BAD_GATEWAY,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                serde_json::to_vec(&error).unwrap_or_default(),
            )
                .into_response()
        }
    }
}

// ─── POST /backend-api/codex/responses/compact ─────────────────────────────
pub async fn handle_compact(State(state): State<AppState>, body: axum::body::Bytes) -> Response {
    let mut request_json: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            logger::error("compact", &format!("Invalid JSON: {}", e));
            return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)).into_response();
        }
    };

    // Remove stream field if present
    if let Some(obj) = request_json.as_object_mut() {
        obj.remove("stream");
    }

    let model_alias = request_json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let backend_model = state.cfg.resolve_model(&model_alias);

    let mut cc_payload =
        translate_request::responses_to_chat_completions(&backend_model, &request_json, false);
    cc_payload = translate_request::inject_thinking(cc_payload, &backend_model);

    let token = match state.token_mgr.get_token().await {
        Ok(t) => t,
        Err(e) => {
            logger::error("compact", &format!("Auth failed: {}", e));
            return (StatusCode::UNAUTHORIZED, format!("Auth failed: {}", e)).into_response();
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

    logger::info(
        "compact",
        &format!(
            "Compact POST (streaming) {} model={} est_tokens={}",
            url, backend_model, estimated_tokens
        ),
    );

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
            logger::error("compact", &format!("Upstream error: {}", e));
            return (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e)).into_response();
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        logger::error(
            "compact",
            &format!(
                "Upstream {}: {}",
                status,
                logger::extract_error_summary(&body_text, 300)
            ),
        );
        logger::error_request(
            "compact",
            &cc_payload.to_string(),
            &body_text,
            status.as_u16(),
        );
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
                        if data.trim() == "[DONE]" {
                            continue;
                        }
                        if let Ok(chunk_json) = serde_json::from_str::<Value>(data) {
                            // Extract text delta
                            if let Some(choices) =
                                chunk_json.get("choices").and_then(|c| c.as_array())
                            {
                                for choice in choices {
                                    if let Some(delta) = choice.get("delta") {
                                        if let Some(content) =
                                            delta.get("content").and_then(|c| c.as_str())
                                        {
                                            full_text.push_str(content);
                                        }
                                    }
                                    if let Some(fr) =
                                        choice.get("finish_reason").and_then(|f| f.as_str())
                                    {
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
                        if let Some(choices) = chunk_json.get("choices").and_then(|c| c.as_array())
                        {
                            for choice in choices {
                                if let Some(delta) = choice.get("delta") {
                                    if let Some(content) =
                                        delta.get("content").and_then(|c| c.as_str())
                                    {
                                        full_text.push_str(content);
                                    }
                                }
                                if let Some(fr) =
                                    choice.get("finish_reason").and_then(|f| f.as_str())
                                {
                                    finish_reason = Some(fr.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    logger::info(
        "compact",
        &format!(
            "Compact completed: {} chars of summary (finish={:?})",
            full_text.len(),
            finish_reason
        ),
    );

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

// ─── POST /v1/summarize-session ────────────────────────────────────────────
// Accepts a JSON body with `messages` (array of {role, content}) from a
// chat session and returns a concise summary. Used by vscode-acp to add
// another session's context as a summary in the current conversation.
//
// Model selection strategy:
//   1. Query the EricAI backend /models endpoint for models with >= 1M
//      context. Pick the first one that is loaded/healthy.
//   2. If no 1M model is available, pick the model with the largest
//      context_window from the cached model catalog.
//   3. If the catalog is missing, fall back to the first model in config.
pub async fn handle_summarize_session(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let request_json: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            logger::error("summarize", &format!("Invalid JSON: {}", e));
            return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)).into_response();
        }
    };

    let messages = match request_json.get("messages").and_then(|m| m.as_array()) {
        Some(arr) if !arr.is_empty() => arr.clone(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "Body must contain non-empty 'messages' array",
            )
                .into_response();
        }
    };

    let session_title = request_json
        .get("title")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();

    // ── Model selection ────────────────────────────────────────────────────
    let summary_model = select_summary_model(&state).await;
    logger::info(
        "summarize",
        &format!("Selected summary model: {}", summary_model),
    );

    // ── Build summarization prompt ─────────────────────────────────────────
    let conversation_text = messages
        .iter()
        .map(|m| {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("unknown");
            let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
            // For array content (multimodal), extract text parts
            let content = if content.is_empty() {
                if let Some(parts) = m.get("content").and_then(|c| c.as_array()) {
                    parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    String::new()
                }
            } else {
                content.to_string()
            };
            format!("## {}:\n{}", role, content)
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let system_prompt =
        "You are a conversation summarizer. Summarize the following conversation concisely. \
Focus on: what was asked, what was done, key decisions, and the current state. \
Keep the summary under 500 words. Do not include code blocks — describe changes in prose.";

    let user_prompt = if session_title.is_empty() {
        format!("Summarize this conversation:\n\n{}", conversation_text)
    } else {
        format!(
            "Summarize this conversation (title: {}):\n\n{}",
            session_title, conversation_text
        )
    };

    let cc_payload = json!({
        "model": summary_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": 2000,
        "temperature": 0.3
    });

    // ── Send to EricAI backend ─────────────────────────────────────────────
    let token = match state.token_mgr.get_token().await {
        Ok(t) => t,
        Err(e) => {
            logger::error("summarize", &format!("Auth failed: {}", e));
            return (StatusCode::UNAUTHORIZED, format!("Auth failed: {}", e)).into_response();
        }
    };

    let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
    let url = format!("{}/chat/completions", base_url);

    logger::info(
        "summarize",
        &format!("POST {} model={}", url, summary_model),
    );

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
            logger::error("summarize", &format!("Upstream error: {}", e));
            return (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e)).into_response();
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        logger::error(
            "summarize",
            &format!(
                "Upstream {}: {}",
                status,
                logger::extract_error_summary(&body_text, 300)
            ),
        );
        return (status, body_text).into_response();
    }

    // ── Collect streamed response ──────────────────────────────────────────
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
                        if data.trim() == "[DONE]" {
                            continue;
                        }
                        if let Ok(chunk_json) = serde_json::from_str::<Value>(data) {
                            if let Some(choices) =
                                chunk_json.get("choices").and_then(|c| c.as_array())
                            {
                                for choice in choices {
                                    if let Some(delta) = choice.get("delta") {
                                        if let Some(content) =
                                            delta.get("content").and_then(|c| c.as_str())
                                        {
                                            full_text.push_str(content);
                                        }
                                    }
                                    if let Some(fr) =
                                        choice.get("finish_reason").and_then(|f| f.as_str())
                                    {
                                        finish_reason = Some(fr.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
                if read_pos > 0 {
                    buffer.drain(..read_pos);
                    read_pos = 0;
                }
            }
            Err(e) => {
                logger::error("summarize", &format!("Stream error: {}", e));
                break;
            }
        }
    }

    // Flush remaining buffer
    if read_pos < buffer.len() {
        let remaining = buffer[read_pos..].trim();
        if !remaining.is_empty() {
            if let Some(data) = remaining.strip_prefix("data: ") {
                if data.trim() != "[DONE]" {
                    if let Ok(chunk_json) = serde_json::from_str::<Value>(data) {
                        if let Some(choices) = chunk_json.get("choices").and_then(|c| c.as_array())
                        {
                            for choice in choices {
                                if let Some(delta) = choice.get("delta") {
                                    if let Some(content) =
                                        delta.get("content").and_then(|c| c.as_str())
                                    {
                                        full_text.push_str(content);
                                    }
                                }
                                if let Some(fr) =
                                    choice.get("finish_reason").and_then(|f| f.as_str())
                                {
                                    finish_reason = Some(fr.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    logger::info(
        "summarize",
        &format!(
            "Summary completed: {} chars (finish={:?})",
            full_text.len(),
            finish_reason
        ),
    );

    let response_obj = json!({
        "summary": if full_text.is_empty() {
            "Unable to generate summary."
        } else {
            &full_text
        },
        "model": summary_model,
    });

    Json(response_obj).into_response()
}

/// Select the best model for summarization.
/// 1. Query EricAI backend /models for 1M+ context models that are loaded.
/// 2. Fall back to the largest context_window model from the catalog.
/// 3. Fall back to the first model in config.
async fn select_summary_model(state: &AppState) -> String {
    // ── Strategy 1: Query backend for 1M models ───────────────────────────
    let token = match state.token_mgr.get_token().await {
        Ok(t) => t,
        Err(_) => {
            logger::warn(
                "summarize",
                "No token for model query — using catalog fallback",
            );
            return select_summary_model_from_catalog(state);
        }
    };

    let base_url = state.cfg.ericai_provider().unwrap().base_url.clone();
    let models_url = format!("{}/models", base_url);

    let resp = state
        .http_client
        .get(&models_url)
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await;

    if let Ok(resp) = resp {
        if resp.status().is_success() {
            if let Ok(models_json) = resp.json::<Value>().await {
                if let Some(data) = models_json.get("data").and_then(|d| d.as_array()) {
                    let mut best_1m: Option<(String, u64)> = None;
                    for m in data {
                        let mid = m.get("id").and_then(|i| i.as_str()).unwrap_or("");
                        // Skip tags and non-1M models
                        if mid.starts_with("tag:") || !mid.contains("1M") || mid.contains("CLI") {
                            continue;
                        }
                        let meta = m.get("rayllm_metadata");
                        let max_tokens = meta
                            .and_then(|m| m.get("engine_config"))
                            .and_then(|e| e.get("max_total_tokens"))
                            .and_then(|t| t.as_u64())
                            .unwrap_or(0);
                        let loaded = meta
                            .and_then(|m| m.get("demand_status"))
                            .and_then(|d| d.get("loaded"))
                            .and_then(|l| l.as_bool())
                            .unwrap_or(false);
                        let running = meta
                            .and_then(|m| m.get("demand_status"))
                            .and_then(|d| d.get("running_replicas"))
                            .and_then(|r| r.as_u64())
                            .unwrap_or(0);

                        if max_tokens >= 1_000_000 && (loaded || running > 0) {
                            // Prefer loaded models; pick the first one found
                            if best_1m.is_none()
                                || (loaded && best_1m.as_ref().unwrap().1 < max_tokens)
                            {
                                best_1m = Some((mid.to_string(), max_tokens));
                            }
                        }
                    }
                    if let Some((model_id, max_tokens)) = best_1m {
                        logger::info(
                            "summarize",
                            &format!(
                                "Found 1M context model: {} (max_tokens={})",
                                model_id, max_tokens
                            ),
                        );
                        return model_id;
                    }
                }
            }
        }
    }

    logger::info(
        "summarize",
        "No 1M model found on backend — falling back to catalog",
    );
    select_summary_model_from_catalog(state)
}

/// Fall back: pick the model with the largest context_window from the catalog.
fn select_summary_model_from_catalog(state: &AppState) -> String {
    if let Some(catalog) = state.model_catalog.as_ref() {
        if let Some(arr) = catalog.get("models").and_then(|m| m.as_array()) {
            let mut best: Option<(String, u64)> = None;
            for m in arr {
                let slug = m.get("slug").and_then(|s| s.as_str()).unwrap_or("");
                let ctx = m
                    .get("context_window")
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0);
                if best.is_none() || ctx > best.as_ref().unwrap().1 {
                    best = Some((slug.to_string(), ctx));
                }
            }
            if let Some((slug, ctx)) = best {
                logger::info(
                    "summarize",
                    &format!("Catalog fallback: {} (context_window={})", slug, ctx),
                );
                // Resolve alias to backend model ID
                return state.cfg.resolve_model(&slug);
            }
        }
    }

    // Last resort: first model in config
    let models = state.cfg.all_models();
    if let Some((_alias, name)) = models.first() {
        logger::warn("summarize", &format!("Last-resort fallback: {}", name));
        return name.clone();
    }

    "Qwen/Qwen2.5-14B-Instruct-1M".to_string()
}
