#[allow(clippy::duplicate_mod)]
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
    extract::{ConnectInfo, DefaultBodyLimit, Query, State, WebSocketUpgrade},
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

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<config::Config>,
    pub http_client: reqwest::Client,
    pub model_catalog: ModelCatalog,
    pub shutdown: tokio_util::sync::CancellationToken,
    pub rate_limiter: Arc<RateLimiter>,
    pub model_throttle: Arc<ModelThrottle>,
    /// Maps a peer `SocketAddr` (IP:port — unique per TCP connection) to a
    /// stable 8-char UUID. All HTTP requests arriving over the SAME TCP
    /// connection share one UUID, so a burst of `/v1/chat/completions` POSTs
    /// from one client can be traced as a single logical session.
    ///
    /// Without this, `conn_id_middleware` generates a fresh UUID per HTTP
    /// request, so 11 requests from one connection produce 11 different IDs —
    /// even though they came from the same client. This cache fixes that.
    ///
    /// The map is bounded (see `MAX_CONN_ID_CACHE`) to prevent unbounded
    /// growth from short-lived connections; stale entries for closed
    /// connections are harmless (a new connection gets a new port → new UUID).
    /// Eviction order is unspecified (HashMap does not preserve insertion
    /// order) — only the size bound is guaranteed.
    pub conn_id_cache: Arc<std::sync::Mutex<std::collections::HashMap<String, String>>>,
}

/// Maximum number of peer→UUID entries to keep in `AppState::conn_id_cache`.
/// Prevents unbounded memory growth from many short-lived TCP connections.
/// When exceeded, entries are evicted so the map stays bounded.
const MAX_CONN_ID_CACHE: usize = 1024;

/// Lock a `std::sync::Mutex`, recovering the lock guard when the lock comes
/// back poisoned (a panic happened while some other thread held it).
///
/// WHY not `.unwrap()`: `std::sync::Mutex` poisons on panic, so the FIRST
/// panic poisons and every later `lock().unwrap()` panics too — the service
/// goes permanently dead at that call site instead of failing one request.
/// Every caller here guards plain bookkeeping state whose worst-case stale
/// value is benign: rate-limit buckets (one IP's per-minute counter off by a
/// little), model concurrency limits (immutable after construction, so
/// provably unaffected), and the peer→connection cache (one peer re-keyed to
/// a fresh UUID on the next request). Continuing is strictly better than
/// dying, and the poison is logged so the originating panic stays visible in
/// the ACP logs rather than being swallowed.
///
/// Returns the guard itself (not the payload) — moving out of a `MutexGuard`
/// is not allowed, and the caller needs the guard alive for its whole scope.
fn recover_poisoned_lock<R>(
    locked: Result<R, std::sync::PoisonError<R>>,
    what: &str,
) -> R {
    match locked {
        Ok(held) => held,
        Err(poisoned) => {
            logger::error(
                "poisoned-lock",
                &format!("{} lock was poisoned by a panicking thread — recovering it and continuing", what),
            );
            poisoned.into_inner()
        }
    }
}

static KEEPALIVE_BYTES: axum::body::Bytes = axum::body::Bytes::from_static(b": keepalive\n\n");

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
        let mut buckets = recover_poisoned_lock(self.buckets.lock(), "rate-limiter bucket");
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

/// Per-model concurrency throttle. Prevents the upstream 429 storm observed
/// when multiple agents (codex-acp, claude-code-acp, kiro) all hit the
/// same backend model (e.g. zai-org/GLM-5.2-FP8) with large contexts at once.
/// `Semaphore::Permit`s are held for the DURATION of each upstream request
/// (including streaming), so excess requests queue instead of being forwarded
/// and rejected with 429.
///
/// A fixed `max_concurrent` is not enough: a single model on a shared vLLM can
/// only prefill/decode so many requests in parallel before queue pressure
/// spikes. Any higher limit just moves the 429 an mbps later.
pub struct ModelThrottle {
    default_limit: usize,
    limits: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    semaphores: tokio::sync::Mutex<std::collections::HashMap<String, Arc<tokio::sync::Semaphore>>>,
}

impl ModelThrottle {
    pub fn new(default_limit: usize, overrides: &std::collections::HashMap<String, usize>) -> Self {
        Self {
            default_limit,
            limits: std::sync::Mutex::new({
                let mut m = std::collections::HashMap::new();
                for (k, v) in overrides {
                    m.insert(k.clone(), *v);
                }
                m
            }),
            semaphores: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn limit_for(&self, model: &str) -> usize {
        let limits = recover_poisoned_lock(self.limits.lock(), "model-concurrency limits");
        limits.get(model).copied().unwrap_or(self.default_limit)
    }

    /// Acquire a permit for `model` on the provider `provider_idx`, waiting up
    /// to `timeout`. Returns None if the wait timed out (caller should respond
    /// 503 + Retry-After).  The semaphore is keyed by provider + model so two
    /// providers serving the same backend model ID get independent slots; the
    /// per-model limit override still looks up the bare model ID.
    pub async fn acquire(
        &self,
        model: &str,
        provider_idx: usize,
        timeout: Duration,
    ) -> Option<tokio::sync::OwnedSemaphorePermit> {
        let limit = self.limit_for(model);
        if limit == 0 {
            return None; // 0 = unlimited (config default is 3, but allow opt-out)
        }

        // Lazily create the semaphore for this provider+model pair.
        let sem_key = format!("{}/{}", provider_idx, model);
        let sem = {
            let mut sems = self.semaphores.lock().await;
            sems.entry(sem_key)
                .or_insert_with(|| Arc::new(tokio::sync::Semaphore::new(limit)))
                .clone()
        };

        let t0 = std::time::Instant::now();
        let res = tokio::time::timeout(timeout, sem.acquire_owned())
            .await
            .ok()
            .and_then(|r| r.ok());
        if let Some(waited_ms) = res.as_ref().map(|_| t0.elapsed().as_millis()) {
            if waited_ms > 250 {
                logger::info(
                    "throttle",
                    &format!(
                        "model {} waited {}ms for a free slot (limit={})",
                        model, waited_ms, limit
                    ),
                );
            }
        }
        res
    }
}

/// Throttle timeout before returning 503. Long enough that a streaming model
/// slot is unlikely to be held much longer; short enough that the client
/// retries rather than hanging. Sized above the longest observed upstream turn
/// (~296s on qwen3.6:35b) so a queued request never 503s behind a live one —
/// waiting IS the GPU gate, the 503 is only an escape hatch.
const MODEL_THROTTLE_TIMEOUT: Duration = Duration::from_secs(600);
/// HTTP 503 Retry-After suggested to the client after a throttle timeout.
const MODEL_THROTTLE_RETRY_AFTER: &str = "60";

/// Send an upstream request, retrying TRANSPORT-level failures only.
///
/// `build` is a closure that constructs a fresh `RequestBuilder` per attempt —
/// reqwest consumes the builder on `send()`, so a retry has to rebuild it.
///
/// WHY transport-only: `send()` fails when reqwest never received response
/// headers (connection dropped, DNS/TLS/socket blip, upstream mid-restart). An
/// HTTP 4xx/5xx from the backend is a real answer — re-sending it doubles the
/// GPU work for a model that already ran, and the throttle/overflow paths
/// already handle those deliberately.
///
/// Sized from this machine's logs: ~1% of upstream attempts fail this way, and
/// of those ~25% are followed within 2 minutes by an attempt that succeeds
/// (upstream briefly unreachable), while ~75% have no next attempt at all
/// (upstream genuinely down). Retrying rescues the first group and costs only
/// the delay below in the second, so it's a net win rather than a guess.
///
/// Returns the response (or the last transport error) plus how many attempts
/// were used, so callers can log the retry count without re-deriving it.
async fn send_upstream_retry<F>(
    retries: u32,
    build: F,
) -> (Result<reqwest::Response, reqwest::Error>, u32)
    where F: Fn() -> reqwest::RequestBuilder
{
    let attempts = retries.saturating_add(1);
    let mut last_err: Option<reqwest::Error> = None;
    for attempt in 1..=attempts {
        match build().send().await {
            Ok(resp) => return (Ok(resp), attempt),
            Err(e) => {
                if attempt < attempts {
                    // Linear backoff: the blip is either a dead socket or the
                    // upstream mid-restart, so a fixed short wait is as good as
                    // exponential here and keeps the worst case bounded (1+2+3
                    // = 6s added for retries=3).
                    let delay = Duration::from_secs(attempt as u64);
                    logger::warn(
                        "upstream-retry",
                        &format!(
                            "transport failure (attempt {}/{}): {} — retrying in {}s",
                            attempt, attempts, e, delay.as_secs()
                        ),
                    );
                    tokio::time::sleep(delay).await;
                } else {
                    last_err = Some(e);
                }
            }
        }
    }
    (Err(last_err.unwrap()), attempts)
}

/// Acquire a per-model permit for the given backend model. Logs when the
/// request had to wait, and returns `Some(permit)` to hold for the duration
/// of the upstream call, or `Some(None)` when throttling is disabled
/// (max=0), or `Err(503 Response)` when the wait timed out.
async fn throttle_model<'a>(
    state: &'a AppState,
    model: &str,
) -> Result<Option<tokio::sync::OwnedSemaphorePermit>, Response> {
    let limit = state.model_throttle.limit_for(model);
    if limit == 0 {
        return Ok(None); // unlimited
    }
    // Resolve which provider will serve this model so two providers carrying
    // the same backend model ID get independent semaphore slots.
    let provider_idx = state.cfg.upstream_for(&model).provider_idx;
    match state
        .model_throttle
        .acquire(model, provider_idx, MODEL_THROTTLE_TIMEOUT)
        .await
    {
        Some(permit) => Ok(Some(permit)),
        None => {
            logger::warn(
                "throttle",
                &format!(
                    "model {} saturated (limit={}), timed out after {}s — 503",
                    model,
                    limit,
                    MODEL_THROTTLE_TIMEOUT.as_secs()
                ),
            );
            Err((
                StatusCode::SERVICE_UNAVAILABLE,
                [("retry-after", MODEL_THROTTLE_RETRY_AFTER)],
                format!(
                    "Model {} is busy (limit={} concurrent). Retry after {}s",
                    model, limit, MODEL_THROTTLE_RETRY_AFTER
                ),
            )
                .into_response())
        }
    }
}

/// Look up a model's context window.
///
/// Two sources, in priority order:
/// 1. The provider block in `config.yaml` (`context_length`), which is what
///    `/v1/models` already advertises — so the guard and the advertisement
///    can never disagree.
/// 2. The Codex model catalog (`slug` + `context_window`), kept for the
///    Anthropic-hosted slugs Codex itself brings up.
///
/// Falls back to `DEFAULT_CONTEXT_WINDOW` when neither lists the model.
pub fn get_context_window(state: &AppState, model_alias: &str) -> u64 {
    if let Some(ctx) = state.cfg.model_context_window(model_alias) {
        return ctx;
    }
    if let Some(c) = state.model_catalog.as_ref() {
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
    DEFAULT_CONTEXT_WINDOW
}

/// Window used when neither `config.yaml` nor the Codex catalog declares a
/// model. Deliberately conservative: an unknown model gets the smallest window
/// we can defend rather than a generous one that lets an oversized request
/// reach a backend that then 400s or 503s it.
const DEFAULT_CONTEXT_WINDOW: u64 = 196_608;

/// The effective context window for a model, as advertised by `/v1/models`.
/// This branch (local/dev) has no `max_context_cap` concept, so it is whatever
/// `get_context_window` resolves — `config.yaml` first, then the Codex catalog.
/// (The EricAI `master` variant caps this at the shared vLLM's enforced
/// `max_model_len`; here each model's declared window is assumed to be what the
/// backend enforces.)
fn effective_context_window(state: &AppState, model_alias: &str) -> u64 {
    get_context_window(&state, &model_alias)
}

/// Merge the model's configured `logit_bias` (config.yaml, per provider model)
/// into an outgoing chat-completions payload. Configured entries win over any
/// client-sent ones: these are sampler guards, not preferences. The driving
/// case is the strata coder pack, where the ChatML role-boundary opener must
/// never be sampled — the model occasionally emits it as content, role-plays
/// a fake user turn ("... user\n... user\nContinue") and then hits its stop
/// token, which the client sees as a premature end_turn mid-task (observed
/// 2026-10-29, 8 output tokens, strata logged the stop; GitHub class:
/// claude-code#88211). A no-op for models without a configured bias.
fn apply_configured_logit_bias(payload: &mut Value, cfg: &config::Config, backend_model: &str) {
    let Some(bias) = cfg.effective_logit_bias(backend_model) else {
        return;
    };
    if bias.is_empty() {
        return;
    }
    let Some(obj) = payload.as_object_mut() else {
        return;
    };
    let entry = obj
        .entry("logit_bias")
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !entry.is_object() {
        // A client-sent logit_bias that is not an object is invalid per the
        // OpenAI schema; replace it rather than dropping the guard.
        *entry = Value::Object(serde_json::Map::new());
    }
    if let Some(map) = entry.as_object_mut() {
        for (token, b) in &bias {
            map.insert(token.clone(), json!(b));
        }
    }
}

/// Build an HTTP 400 response when a request's estimated token count exceeds
/// the model's effective context window. Returns `None` when the request fits
/// (or the window is unknown), so callers fall through to normal forwarding.
///
/// WHY reject here instead of forwarding: the proxy's `estimate_tokens` is
/// more accurate than codex's own `get_total_token_usage` for a single huge
/// injected message (e.g. a pasted config file) — codex's tracked usage
/// undercounts items after the last model-generated item, so its pre-turn
/// auto-compaction (`run_pre_sampling_compact`, threshold = 90% of the
/// hardcoded 272k fallback window) never fires and a multi-MB request goes
/// to the backend. The backend then either 400s (model window exceeded) or
/// 503s (shared-vLLM prefill saturated by the oversized request). Rejecting
/// here turns that into a clean client-visible context-length error the
/// client (codex/ericli) recognizes and compacts/retries against.
///
/// The message includes "context length" so `is_context_overflow_error`
/// (logger.rs) classifies it as an overflow, not a transient overload.
fn context_overflow_response(
    state: &AppState,
    model_alias: &str,
    estimated_tokens: usize,
) -> Option<Response> {
    let window = effective_context_window(state, model_alias);
    if estimated_tokens as u64 <= window {
        return None;
    }
    logger::warn(
        "context-overflow",
        &format!(
            "Rejecting before upstream: ~{} tokens > context_window {} for model {} — forwarding would fail with 400/503",
            estimated_tokens, window, model_alias
        ),
    );
    let msg = format!(
        "The input ({} estimated tokens) is longer than the model's context length ({} tokens). Compact the conversation or switch to a larger-context model.",
        estimated_tokens, window
    );
    Some((StatusCode::BAD_REQUEST, msg).into_response())
}

/// True when the model declares image input support.
///
/// Two sources, in priority order:
/// 1. `input_modalities` on the provider model entry in `config.yaml` — the
///    proxy's own models live here, and Codex's catalog doesn't know their
///    slugs, so without this every image block would be stripped unconditionally.
/// 2. The Codex model catalog (`input_modalities`), for Anthropic-hosted slugs.
///
/// Unknown entries are text-only, so image blocks get stripped before the
/// backend rejects the request with a hard 500 ("not a multimodal model").
pub fn get_model_supports_images(state: &AppState, model_alias: &str) -> bool {
    if let Some(mods) = state.cfg.model_input_modalities(model_alias) {
        return mods.iter().any(|v| v.as_str() == "image");
    }
    if let Some(c) = state.model_catalog.as_ref() {
        if let Some(arr) = c.get("models").and_then(|m| m.as_array()) {
            for m in arr {
                if m.get("slug").and_then(|s| s.as_str()) == Some(model_alias) {
                    if let Some(mods) = m.get("input_modalities").and_then(|v| v.as_array()) {
                        return mods.iter().any(|v| v.as_str() == Some("image"));
                    }
                }
            }
        }
    }
    false
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

    if cfg.provider().is_none() {
        logger::error("main", "No provider found in config");
        std::process::exit(1);
    }

    let rate_limiter = Arc::new(RateLimiter::new(cfg.rate_limit_per_minute));
    let model_throttle = Arc::new(ModelThrottle::new(
        cfg.max_concurrent_per_model,
        &cfg.model_concurrency,
    ));
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

    // Load the optional Codex-style model catalog once at startup — avoids
    // per-request disk reads. This is a SECONDARY metadata source: model
    // windows and modalities come from the provider blocks in config.yaml
    // first (see get_context_window / get_model_supports_images). The catalog
    // only has to exist for slugs the proxy doesn't declare itself, e.g. the
    // Anthropic-hosted ones Codex brings up, and its path is whatever Codex
    // configured (`model_catalog_json = "~/.codex/models.json"` in
    // ~/.codex/config.toml) — it is NOT `model_catalog.json`, so hardcoding
    // that name used to miss the file every time.
    let catalog_path = cfg.model_catalog.clone()
        .or_else(|| std::env::var("HOME").ok()
            .map(|h| format!("{}/.codex/model_catalog.json", h)));
    let model_catalog: ModelCatalog = Arc::new(
        catalog_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .as_ref().and_then(|s| serde_json::from_str::<Value>(s).ok()),
    );
    if model_catalog.is_some() {
        logger::info(
            "main",
            &format!("Model catalog loaded from {}", catalog_path.unwrap()),
        );
    } else if catalog_path.is_some() {
        logger::info(
            "main",
            &format!(
                "Model catalog not found at {} — model windows and modalities come from config.yaml (DEFAULT_CONTEXT_WINDOW {} only for slugs neither lists)",
                catalog_path.unwrap(), DEFAULT_CONTEXT_WINDOW
            ),
        );
    } else {
        logger::info(
            "main",
            &format!(
                "No model catalog configured — model windows and modalities come from config.yaml (DEFAULT_CONTEXT_WINDOW {})",
                DEFAULT_CONTEXT_WINDOW
            ),
        );
    }

    // Shared shutdown signal — cancelled on SIGINT/SIGTERM so all WebSocket
    // handlers can close their connections gracefully.
    let shutdown_token = tokio_util::sync::CancellationToken::new();

    let state = AppState {
        cfg: cfg.clone(),
        http_client,
        model_catalog,
        shutdown: shutdown_token.clone(),
        rate_limiter,
        model_throttle,
        conn_id_cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
    };

    let app = axum::Router::new()
        .route("/v1/models", get(handle_models))
        .route(
            "/v1/responses",
            post(handle_responses).get(handle_responses_ws),
        )
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/embeddings", post(handle_embeddings))
        .route("/v1/messages", post(handle_messages))
        // Codex can send very large request bodies (up to ~30MB for long
        // conversation contexts with 196K+ tokens). Axum's default body
        // limit is 2MB, which causes "413 Payload Too Large" errors that
        // manifest as "stream disconnected before completion" in codex.
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50 MB
        .layer(middleware::from_fn_with_state(state.clone(), conn_id_middleware))
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

    // ── IPv4-ONLY bind ──────────────────────────────────────────────────────
    // The proxy binds strictly to IPv4. The socket is created with
    // `Domain::IPV4`, so it can never accept IPv6 connections. The bind
    // address is parsed as an IPv4 literal (`Ipv4Addr`) — if `cfg.host` is a
    // hostname ("localhost") or an IPv6 literal ("::1"), `Ipv4Addr::from_str`
    // fails and we fall back to the loopback address `127.0.0.1` with a
    // warning, instead of panicking on startup.
    //
    // WHY: many clients (e.g. the ericli sidecar) resolve "localhost" to ::1
    // (IPv6) FIRST. If the proxy ever bound to IPv6, those clients would get
    // connection-refused because the proxy only listens on IPv4. Hard-coding
    // IPv4 here makes the proxy's address unambiguous and reachable by all
    // local clients regardless of their getaddrinfo() order.
    let bind_v4: std::net::SocketAddrV4 = match cfg.host.parse::<std::net::Ipv4Addr>() {
        Ok(ip) => std::net::SocketAddrV4::new(ip, cfg.port),
        Err(e) => {
            logger::warn(
                "main",
                &format!(
                    "config host {:?} is not a valid IPv4 literal ({}) — binding to 127.0.0.1:{} (IPv4 only)",
                    cfg.host, e, cfg.port
                ),
            );
            std::net::SocketAddrV4::new(std::net::Ipv4Addr::LOCALHOST, cfg.port)
        }
    };
    logger::info(
        "main",
        &format!("vllm-proxy listening on http://{}", bind_v4),
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
            .bind(&socket2::SockAddr::from(bind_v4))
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

// ─── Connection ID Middleware ───────────────────────────────────────────────
/// Generates a short 8-char UUID per HTTP request and stores it in a
/// task-local so all `logger::` calls within the request (including
/// `tokio::spawn`ed sub-tasks) share the same identifier in the log.
/// For WebSocket upgrades, the conn_id cannot propagate through
/// `on_upgrade` (it spawns a new task), so the WS handler sets it
/// manually inside the upgrade closure.
/// Resolve the stable per-connection UUID for a peer key (e.g. `SocketAddr`
/// string). Reuses the existing UUID if the peer is already cached, otherwise
/// generates a fresh one and caches it. Bounds the cache at
/// `MAX_CONN_ID_CACHE` entries, evicting arbitrary entries when over the cap
/// so short-lived connections can't grow the map without limit. The eviction
/// order is unspecified (std::collections::HashMap does not guarantee
/// insertion order) — only the bound is guaranteed.
///
/// This is a pure function (no I/O) so it can be unit-tested directly.
fn resolve_conn_id(
    cache: &mut std::collections::HashMap<String, String>,
    key: &str,
) -> String {
    if let Some(id) = cache.get(key) {
        return id.clone();
    }
    if cache.len() >= MAX_CONN_ID_CACHE {
        let drop_count = cache.len() - MAX_CONN_ID_CACHE + 1;
        let keys: Vec<String> = cache.keys().cloned().collect();
        for k in keys.iter().take(drop_count) {
            cache.remove(k);
        }
    }
    let id = logger::new_conn_id();
    cache.insert(key.to_string(), id.clone());
    id
}

/// Assigns a per-CONNECTION UUID (keyed by peer `SocketAddr`) to every HTTP
/// request and stores it in a task-local so all `logger::` calls within the
/// request (including `tokio::spawn`ed sub-tasks) share the same identifier.
///
/// Unlike the original per-REQUEST UUID, requests arriving over the SAME TCP
/// connection (e.g. a keep-alive burst of `/v1/chat/completions` POSTs) now
/// share ONE UUID, so a client's rapid request sequence is traceable as a
/// single logical session.
///
/// The peer `SocketAddr` (IP:port) uniquely identifies a TCP connection. A
/// new connection gets a new port → a new UUID. The cache is bounded by
/// `MAX_CONN_ID_CACHE` to avoid unbounded growth.
///
/// For WebSocket upgrades, the conn_id cannot propagate through `on_upgrade`
/// (it spawns a new task), so the WS handler sets it manually inside the
/// upgrade closure (same as before).
async fn conn_id_middleware(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<std::net::SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    let key = peer.to_string();
    let conn_id = {
        let mut cache = recover_poisoned_lock(state.conn_id_cache.lock(), "peer->connection-id cache");
        resolve_conn_id(&mut cache, &key)
    };
    logger::set_conn_id(&conn_id, next.run(request)).await
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
    let mut models: Vec<Value> = Vec::with_capacity(all_models.len());
    for (alias, _name) in &all_models {
            // Same lookup the overflow guard uses, so /v1/models and the
            // 400-on-overflow path can never advertise different windows.
            let context_length = state.cfg.model_context_window(alias);
            let mut entry = json!({
                "id": alias,
                "object": "model",
                "created": 1700000000,
                "owned_by": "vllm",
            });
            if let Some(ctx) = context_length {
                entry["context_length"] = json!(ctx);
            }
            if let Some(tags) = state.cfg.model_tags(alias) {
                entry["tags"] = json!(tags);
            }
            if let Some(mods) = state.cfg.model_input_modalities(alias) {
                entry["input_modalities"] = json!(mods);
            }
            models.push(entry);
    }
    Json(json!({"object": "list", "data": models}))
}

// ─── GET /health ────────────────────────────────────────────────────────────
pub async fn handle_health() -> impl IntoResponse {
    Json(json!({"status": "ok", "service": "vllm-proxy"}))
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

    // Per-model concurrency throttle — hold spot for the whole upstream call.
    let _permit = match throttle_model(&state, &backend_model).await {
        Ok(p) => p,
        Err(e) => return e,
    };

    let is_stream = request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // Translate request
    let mut cc_payload =
        translate_request::responses_to_chat_completions(&backend_model, &request_json, is_stream);
    cc_payload = translate_request::inject_thinking(cc_payload, &backend_model);
    apply_configured_logit_bias(&mut cc_payload, &state.cfg, &backend_model);

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
    //
    // NEW: codex's auto-compaction is INERT for proxy-routed models — its
    // pre-turn guard uses a hardcoded 272k fallback window and its tracked
    // usage undercounts a single huge injected message, so an oversized
    // request can still reach here. Hard-reject before forwarding: the proxy's
    // estimate is accurate enough to catch the egregious overflow, and a clean
    // 400 here beats a backend 400/503 storm on the shared vLLM.
    let context_window = get_context_window(&state, &model_alias);
    let estimated_tokens = estimate_tokens(&cc_payload);
    if estimated_tokens as u64 > context_window {
        logger::warn("responses", &format!(
           "Context size warning: ~{} tokens may exceed context_window {} for model {} — codex should auto-compact",
           estimated_tokens, context_window, model_alias
       ));
    }
    if let Some(resp) = context_overflow_response(&state, &model_alias, estimated_tokens) {
        return resp;
    }

    // Get auth token
        let upstream = state.cfg.upstream_for(&backend_model);
        let base_url = upstream.base_url;
    let url = format!("{}/chat/completions", base_url);

    logger::info(
        "responses",
        &format!("POST {} model={} stream={}", url, backend_model, is_stream),
    );

    let (resp_result, upstream_attempts) = send_upstream_retry(
        state.cfg.request_retry,
        || {
            let req = state
                .http_client
                .post(&url)

                .header("Accept", "text/event-stream");
                let req = if let Some(k) = upstream.api_key.clone() { req.header("Authorization", format!("Bearer {}", k)) } else { req };
                req
                .json(&cc_payload)
        },
    ).await;
        let resp = match resp_result
    {
        Ok(r) => r,
        Err(e) => {
            logger::error("responses", &format!("Upstream error after {} attempt(s): {}", upstream_attempts, e));
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

        // Everything the dropped-tool-call recovery needs to replay the request
        // non-streaming (see retry_non_streaming).
        let retry_noargs_tool_call = responses_request_has_zero_parameter_tool(&request_json)
            && needs_noargs_retry(&backend_model);
        let retry_client = state.http_client.clone();
        let retry_url = url.clone();
        let retry_api_key = upstream.api_key.clone();
            let retry_request = cc_payload.clone();

        // Use a channel so we can detect client disconnects.  When the
        // client (codex) drops the connection — e.g. when the user clicks
        // Stop — Axum stops polling the response body.  The channel sender
        // then fails on the next `send()`, and we abort the upstream stream.
        use tokio::sync::mpsc;
        let (tx, rx) = mpsc::channel::<Result<axum::body::Bytes, std::convert::Infallible>>(32);

        let cancel_token = tokio_util::sync::CancellationToken::new();
        let upstream_cancel = cancel_token.clone();
        let server_shutdown = state.shutdown.clone();

        // Capture conn_id so the spawned task can re-scope it (tokio
        // task-locals do not propagate into tokio::spawn).
        let conn_id = logger::current_conn_id();

        // Spawn the upstream reader task
        tokio::spawn(async move {
            let inner = async move {
            let mut stream = byte_stream;
            let mut buffer = String::new();
            // Read cursor: bytes before this offset have already been
            // processed. Instead of drain(..pos+1) which shifts all
            // remaining bytes left (O(n²) for n lines), we advance the
            // cursor and compact only once per chunk.
            let mut read_pos: usize = 0;
            // SSE keepalive: send a comment line every 15 seconds while
            // waiting for upstream data.  This prevents the HTTP connection
            // from timing out when the backend takes a long time to process
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
                                      // Defer `[DONE]`: process_line() turns it into
                                      // flush_and_complete(), which emits response.completed. That must
                                      // happen AFTER the dropped-tool-call recovery below, or the
                                      // recovered function_call lands after the response is complete.
                                      if buffer[read_pos..pos].trim() == "data: [DONE]" {
                                          read_pos = pos + 1;
                                          continue;
                                      }
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
               // `[DONE]` is deferred to the post-loop path so the
               // dropped-tool-call recovery runs before response.completed.
               let events = if buffer[read_pos..].trim() == "data: [DONE]" {
                   Vec::new()
               } else {
                   translator.process_line(&buffer[read_pos..])
               };
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

          // If the upstream stream ended without a finish_reason (silent
          // stall, connection drop), emit an error event instead of
          // response.completed. Without this, the proxy synthesizes a
          // completed response with end_turn=true, and Codex treats the
          // truncated stream as a complete turn.
          if !translator.has_finish_reason() {
              logger::error("responses", "Upstream stream ended without finish_reason — sending error instead of response.completed");
              let err_payload = json!({
                  "type": "error",
                  "status": 502,
                  "error": {
                      "message": "Upstream stream ended without completion signal — response may be truncated"
                  }
              });
              let err_event = format!("event: error\ndata: {}\n\n", err_payload);
              if tx.send(Ok(axum::body::Bytes::from(err_event.into_bytes()))).await.is_err() {
                  return;
              }
              return;
          }

          // Same dropped-tool-call recovery as /v1/messages: GLM-5.2 silently
          // discards streamed tool calls with empty arguments (codex's
          // `get_goal`), so re-run the request non-streaming and splice the
          // tool call in before completing the response.
          if retry_noargs_tool_call
              && translator.should_retry_non_streaming(NOARGS_RETRY_MAX_TEXT)
          {
              if let Some(message) =
                  retry_non_streaming(
                      &retry_client,
                      &retry_url,
                      &retry_request,
                      retry_api_key.as_deref(),
                  )
                      .await
              {
                  let events = translator.inject_tool_calls_from_message(&message);
                  if !events.is_empty() {
                      logger::info(
                          "responses",
                          "Recovered a dropped tool call via a non-streaming re-run",
                      );
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
              }
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
            };
            // Re-scope the conn_id so log lines in the spawned task
            // share the same UUID as the parent request handler.
            if let Some(id) = conn_id {
                logger::set_conn_id(&id, inner).await;
            } else {
                inner.await;
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
        // instances using the proxy simultaneously. Stored in task-local so
        // all log calls (including spawned sub-tasks) share it automatically.
        let conn_short = logger::new_conn_id();
        logger::set_conn_id(&conn_short, async {
        logger::info("ws", "WebSocket connection opened for /v1/responses");

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
                    logger::info("ws", "WebSocket closed by client");
                    break 'connection;
                }
                Err(_) => {
                    logger::info("ws", "WebSocket idle timeout (300s between turns), closing");
                    let _ = socket.close().await;
                    break 'connection;
                }
            };

            // Check if the server is shutting down (SIGINT/SIGTERM received).
            if state.shutdown.is_cancelled() {
                logger::info("ws", "Server shutting down, closing WebSocket");
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

            // Per-model concurrency throttle — hold spot for the duration of
            // this turn's upstream call. Multiple codex/ci instances
            // share the proxy; without this they all forward to the backend
            // simultaneously and GLM-5.2 returns 429 (queue full).
            let _permit = match throttle_model(&state, &backend_model).await {
                Ok(p) => p,
                Err(_err_resp) => {
                    let _ = socket.send(axum::extract::ws::Message::Text(
                        format!(
                            "{{\"type\":\"error\",\"status\":{},\"error\":{{\"message\":\"{}\"}}}}",
                            503,
                            "Model busy — proxy throttle timeout. Try again."
                        )
                    )).await;
                    continue 'connection; // keep connection alive for retry
                }
            };

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
                        "Incremental request: previous_response_id={}, merged {} cached + {} new = {} total items",
                        prev_id, cached_count, new_count, total_count
                    ));
                } else {
                    // Cache miss: previous_response_id was evicted (FIFO
                    // at 50 entries) or belongs to a different connection.
                    // Sending only the delta items (1-3 new items) would
                    // give the LLM an incomplete conversation → hallucination.
                    // Instead, send an explicit 409 error so codex retries
                    // with a full request (no previous_response_id).
                    logger::error("ws", &format!(
                        "previous_response_id={} not found in cache — sending 409 to force full retry",
                        prev_id
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
            apply_configured_logit_bias(&mut cc_payload, &state.cfg, &backend_model);
            if logger::is_enabled(logger::LogLevel::Debug) {
                logger::debug("ws", &format!("cc_payload keys: {} | max_tokens: {} | reasoning_effort: {} | chat_template_kwargs: {} | messages: {}", cc_payload.as_object().map(|o| o.keys().map(|k| k.as_str()).collect::<Vec<_>>().join(",")).unwrap_or_default(), cc_payload.get("max_tokens").map(|v| v.to_string()).unwrap_or("none".to_string()), cc_payload.get("reasoning_effort").map(|v| v.to_string()).unwrap_or("none".to_string()), cc_payload.get("chat_template_kwargs").map(|v| v.to_string()).unwrap_or("none".to_string()), cc_payload.get("messages").and_then(|m| m.as_array()).map(|a| a.len()).unwrap_or(0)));

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
                    logger::debug("ws", &format!("cc_payload messages:\n{}", msg_summary.join("\n")));
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
                                "msg_diff: PURE APPEND | prefix={} new={} total={} | +{}",
                                shared, new_count, curr_len, new_items.join(", ")
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
                            "msg_diff: PREFIX CHANGED (KV-cache break!) | shared={} removed={} added={} total={}",
                            shared, removed.len(), added.len(), curr_len
                        ));
                        logger::warn("ws", &format!(
                            "  removed: [{}]", removed.join(", ")
                        ));
                        logger::warn("ws", &format!(
                            "  added:   [{}]", added.join(", ")
                        ));
                    } else if prev_len == 0 {
                        // First turn (debug only)
                        if debug_on {
                            logger::debug("ws", &format!(
                                "msg_diff: FIRST TURN | total={}", curr_len
                            ));
                        }
                    } else if debug_on {
                        // shared == prev_len, new_count == 0 (no change?) or edge case
                        logger::debug("ws", &format!(
                            "msg_diff: no change | prev={} curr={} shared={}",
                            prev_len, curr_len, shared
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
            //
            // NEW: codex's auto-compaction is INERT for proxy-routed models (hardcoded
            // 272k fallback window + undercounted usage for a single huge message), so
            // an oversized request can reach here. Hard-reject before forwarding: send
            // an error event codex's parse_wrapped_websocket_error_event understands
            // (same shape as the upstream-error path below) and keep the connection
            // alive so codex can compact/retry on the next turn.
            let context_window = get_context_window(&state, &model_alias);
            let estimated_tokens = estimate_tokens(&cc_payload);
            if estimated_tokens as u64 > context_window {
                logger::warn("ws", &format!(
                    "Context size warning: ~{} tokens may exceed context_window {} for model {} — codex should auto-compact",
                    estimated_tokens, context_window, model_alias
                ));
            }
            if let Some(window) = { let w = effective_context_window(&state, &model_alias); (estimated_tokens as u64 > w).then_some(w) } {
                logger::warn(
                    "context-overflow",
                    &format!(
                        "Rejecting before upstream: ~{} tokens > context_window {} for model {} — forwarding would fail with 400/503",
                        estimated_tokens, window, model_alias
                    ),
                );
                let err_payload = json!({
                    "type": "error",
                    "status": 400,
                    "error": {
                        "message": format!("The input ({} estimated tokens) is longer than the model's context length ({} tokens). Compact the conversation or switch to a larger-context model.", estimated_tokens, window)
                    }
                });
                let _ = socket.send(axum::extract::ws::Message::Text(
                    err_payload.to_string()
                )).await;
                continue 'connection; // keep connection alive for compact/retry
            }

                       let upstream = state.cfg.upstream_for(&backend_model);
                       let base_url = upstream.base_url;
           let url = format!("{}/chat/completions", base_url);

           logger::info("ws", &format!("WS POST {} model={} stream={}", url, backend_model, is_stream));
            logger::info("ws", &format!("WS POST {} model={} stream={} input_items={}",
                url, backend_model, is_stream,
                effective_request_arc.get("input").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0)));

            // Send the request to the backend.  While waiting for the upstream
            // response headers (which can take 60+ seconds for large resumed
            // sessions), send WebSocket pings every 15 seconds to keep the
            // connection alive and prevent "Connection reset by peer".
            let req = state.http_client
                .post(&url)

                .header("Accept", "text/event-stream");
                let req = if let Some(k) = upstream.api_key.clone() { req.header("Authorization", format!("Bearer {}", k)) } else { req };
                let send_fut = req
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
            // when the upstream takes a long time to process large contexts
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
                                    // Defer `[DONE]`: process_line() turns it into
                                    // flush_and_complete(), which emits response.completed. That must
                                    // happen AFTER the dropped-tool-call recovery below, or the
                                    // recovered function_call lands after the response is complete.
                                    if buffer[read_pos..pos].trim() == "data: [DONE]" {
                                        read_pos = pos + 1;
                                        continue;
                                    }
                                    let events = translator.process_line(&buffer[read_pos..pos]);
                                    // Hot path: only allocate the preview string when
                                    // debug logging is actually on. With multiple codex
                                    // clients streaming, this saves an O(n) chars().collect()
                                    // per upstream SSE line.
                                    if logger::is_enabled(logger::LogLevel::Debug) {
                                        let raw_line: String = buffer[read_pos..pos].chars().take(300).collect();
                                        logger::debug("ws", &format!("← upstream line: {}", raw_line));
                                    }
                                    read_pos = pos + 1;
                                   for event in events {
                                       stream_event_count += 1;
                                       if logger::is_enabled(logger::LogLevel::Debug) {
                                           let preview: String = event.chars().take(500).collect();
                                           logger::debug("ws", &format!("→ codex #{} (stream): {}", stream_event_count, preview));
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
                    // Server shutdown: send a Close frame and exit the stream loop.
                    // Without this, axum 0.7's with_graceful_shutdown waits forever
                    // for upgraded WebSocket connections that never close on their own.
                    _ = state.shutdown.cancelled() => {
                        logger::info("ws", "Server shutting down, closing WebSocket during stream");
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
               let _ = socket.send(axum::extract::ws::Message::Text(
                   err_payload.to_string()
               )).await;
               continue 'connection;
           }

           // Flush remaining buffer
          if read_pos < buffer.len() && !buffer[read_pos..].trim().is_empty() {
               // `[DONE]` is deferred to the post-loop path so the
               // dropped-tool-call recovery runs before response.completed.
               let events = if buffer[read_pos..].trim() == "data: [DONE]" {
                   Vec::new()
               } else {
                   translator.process_line(&buffer[read_pos..])
               };
               for event in events {
                   stream_event_count += 1;
                   if logger::is_enabled(logger::LogLevel::Debug) {
                       let preview: String = event.chars().take(500).collect();
                       logger::debug("ws", &format!("→ codex #{} (flush): {}", stream_event_count, preview));
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

           // If the upstream stream ended without a finish_reason (silent
           // stall, connection drop), emit an error event instead of
           // response.completed. Without this, the proxy synthesizes a
           // completed response with end_turn=true, and Codex treats the
           // truncated stream as a complete turn.
           if !translator.has_finish_reason() {
               logger::error("ws", "Upstream stream ended without finish_reason — sending error instead of response.completed");
               let err_payload = json!({
                   "type": "error",
                   "status": 502,
                   "error": {
                       "message": "Upstream stream ended without completion signal — response may be truncated"
                   }
               });
               let _ = socket.send(axum::extract::ws::Message::Text(
                   err_payload.to_string()
               )).await;
               continue 'connection;
           }

           // Same dropped-tool-call recovery as /v1/messages and the SSE
           // responses path: GLM-5.2 silently discards streamed tool calls with
           // empty arguments (codex's `get_goal`), so re-run the request
           // non-streaming and splice the tool call in before completing.
           if responses_request_has_zero_parameter_tool(&effective_request_arc)
               && needs_noargs_retry(&backend_model)
               && translator.should_retry_non_streaming(NOARGS_RETRY_MAX_TEXT)
           {
               if let Some(message) = retry_non_streaming(
                   &state.http_client,
                   &url,
                   &cc_payload,
                   upstream.api_key.as_deref(),
               )
               .await
               {
                   let events = translator.inject_tool_calls_from_message(&message);
                   if !events.is_empty() {
                      logger::info(
                          "ws",
                          "Recovered a dropped tool call via a non-streaming re-run",
                      );
                       for event in events {
                           if socket.send(axum::extract::ws::Message::Text(event)).await.is_err() {
                               break 'connection;
                           }
                       }
                   }
               }
           }

           // Properly flush the translator by passing [DONE], which emits
           // the full response.completed event with output/usage data.
           // This replaces the previous manual response.completed that was
           // missing output, usage, model, and other fields.
           let done_events = translator.process_line("data: [DONE]");
            for event in done_events {
                if logger::is_enabled(logger::LogLevel::Debug) {
                    let preview: String = event.chars().take(500).collect();
                    logger::debug("ws", &format!("→ codex (DONE flush): {}", preview));
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
                logger::debug("ws", &format!("get_output_items: {} items — types: [{}]", output_items.len(), output_items.iter().map(|item| item.get("type").and_then(|v| v.as_str()).unwrap_or("?")).collect::<Vec<_>>().join(", ")));
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
                            "Appended {} assistant output items to cached request (input: {} → {})",
                            input_arr.len() - before_len, before_len, input_arr.len()
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
            logger::debug("ws", &format!("Turn complete ({} events sent), looping back to recv() next request", stream_event_count));
        }

        logger::info("ws", "WebSocket connection closed");
        }).await
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

/// Per-SSE-line classification used by the thinking-only guard on the
/// /v1/chat/completions passthrough.
///
/// The guard must inspect each `data:` line BEFORE `rewrite_reasoning_to_content`
/// runs, because the rewrite moves `delta.reasoning` into `delta.content` —
/// after the rewrite we can no longer tell a genuine answer from a thinking
/// trace that was promoted to look like one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ChatDeltaFlags {
    /// A non-empty `delta.content` field was present (real answer text).
    has_content: bool,
    /// A non-empty `delta.reasoning` / `delta.reasoning_content` was present.
    has_reasoning: bool,
    /// A non-empty `delta.tool_calls` array was present (tool turn).
    has_tool_calls: bool,
}

/// Inspect a single SSE `data:` line from the upstream chat/completions
/// stream. Returns `None` for non-JSON lines (keepalives, `[DONE]`, empty).
fn inspect_chat_delta(line: &str) -> Option<ChatDeltaFlags> {
    let trimmed = line.trim();
    if !trimmed.starts_with("data: ") || trimmed == "data: [DONE]" || trimmed.len() <= 6 {
        return None;
    }
    let v: Value = serde_json::from_str(&trimmed[6..]).ok()?;
    let mut flags = ChatDeltaFlags::default();
    let Some(choices) = v.get("choices").and_then(|c| c.as_array()) else {
        return Some(flags);
    };
    for choice in choices {
        let Some(delta) = choice.get("delta").and_then(|d| d.as_object()) else {
            continue;
        };
        if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
            if !c.trim().is_empty() {
                flags.has_content = true;
            }
        }
        let rc = delta
            .get("reasoning")
            .and_then(|v| v.as_str())
            .or_else(|| delta.get("reasoning_content").and_then(|v| v.as_str()));
        if let Some(rc) = rc {
            if !rc.trim().is_empty() {
                flags.has_reasoning = true;
            }
        }
        if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            if !tcs.is_empty() {
                flags.has_tool_calls = true;
            }
        }
    }
    Some(flags)
}

/// Compose the SSE error frame used by the thinking-only guard.
fn thinking_only_error_event() -> String {
    let err_payload = json!({
        "error": {
            "message": "Model produced only reasoning (thinking) with no final answer — retrying with different settings",
            "code": 502,
        }
    });
    format!("data: {}\n\n", err_payload)
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

    // Per-model concurrency throttle — hold spot for the whole upstream call.
    let _permit = match throttle_model(&state, &backend_model).await {
        Ok(p) => p,
        Err(e) => return e,
    };

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

    apply_configured_logit_bias(&mut request_json, &state.cfg, &backend_model);

        let upstream = state.cfg.upstream_for(&backend_model);
        let base_url = upstream.base_url;
    let url = format!("{}/chat/completions", base_url);

    logger::info(
        "chat",
        &format!("Passthrough POST {} model={}", url, backend_model),
    );

    // Ollama mac provider: its OpenAI-compat /v1/chat/completions ignores
    // enable_thinking and burns reasoning tokens (see tdai-gateway.mac.yaml).
    // For non-streaming calls, round-trip through Ollama-native /api/chat with
    // think:false and translate the response back to OpenAI shape — mirrors the
    // memory gateway's own ollama-openai-fetch.ts so both routes suppress
    // thinking. Streaming stays on the passthrough path (native translation is
    // out of scope, same as the gateway's fetch wrapper).
    let ollama_native = base_url.contains(":11434");
    let nonstream = !request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if ollama_native && nonstream {
        let root = if base_url.ends_with("/v1") {
            base_url[..base_url.len() - 3].to_string()
        } else {
            base_url
        };
        let native_url = format!("{}/api/chat", root);

        // Ollama-native /api/chat expects tool-call `function.arguments` as an
        // OBJECT; the OpenAI-compat request the SDK replays carries it as a JSON
        // STRING (OpenAI format). On the extractor's multi-turn loop the step-1
        // tool call is sent back as history, and Ollama's parser then fails:
        // "Value looks like object, but can't find closing '}' symbol" → 400 →
        // skill-extract dead-letters. Mirror of the response-path normalization
        // (which stringifies object→string); here parse string→object.
        let raw_messages = request_json
            .get("messages")
            .cloned()
            .unwrap_or(json!([]));
        let native_messages: Value = if let Some(arr) = raw_messages.as_array() {
            let out: Vec<Value> = arr.iter().map(|msg| {
                let mut m = msg.clone();
                if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
                    let normalized: Vec<Value> = tcs.iter().map(|tc| {
                        let e = tc.clone();
                        if let Some(fnv) = e.clone().get("function").and_then(|f| f.as_object()) {
                            if let Some(args) = fnv.get("arguments") {
                                if let Some(s) = args.as_str() {
                                    let parsed = serde_json::from_str(s).unwrap_or(json!(null));
                                    if parsed.is_object() {
                                        let mut ee = e;
                                        ee["function"] = json!({
                                            "name": fnv.get("name").cloned().unwrap_or(json!("")),
                                            "arguments": parsed,
                                        });
                                        return ee;
                                    }
                                }
                            }
                        }
                        e
                    }).collect();
                    m["tool_calls"] = json!(normalized);
                }
                m
            }).collect();
            json!(out)
        } else {
            raw_messages
        };

        let mut native_body = json!({
            "model": request_json.get("model").cloned().unwrap_or(json!(backend_model)),
            "messages": native_messages,
            "stream": false,
            "think": thinking_enabled,
        });
        if let Some(tools) = request_json.get("tools") {
            native_body["tools"] = tools.clone();
        }
        if let Some(tc) = request_json.get("tool_choice") {
            native_body["tool_choice"] = tc.clone();
        }
        // Forward OpenAI response_format to Ollama-native `format` so JSON
        // extraction gets constrained decoding (raw JSON, no markdown fences).
        // Without this, newer models (qwen3.6) wrap JSON in ```json fences and
        // the gateway's strict JSON.parse fails → skill-extract dead-letters.
        if let Some(rf) = request_json.get("response_format") {
            let rtype = rf.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if rtype == "json_object" {
                native_body["format"] = json!("json");
            } else if rtype == "json_schema" {
                if let Some(schema) = rf.get("json_schema").and_then(|js| js.get("schema")) {
                    native_body["format"] = schema.clone();
                }
            }
        }
        let mut opts = json!({});
        if let Some(mt) = request_json
            .get("max_completion_tokens")
            .or_else(|| request_json.get("max_tokens"))
        {
            opts["num_predict"] = mt.clone();
        }
        if let Some(t) = request_json.get("temperature") {
            opts["temperature"] = t.clone();
        }
        if let Some(p) = request_json.get("top_p") {
            opts["top_p"] = p.clone();
        }
        if !opts.as_object().map(|o| o.is_empty()).unwrap_or(false) {
            native_body["options"] = opts;
        }

        logger::info(
            "chat",
            &format!("Ollama native POST {} model={} think={}", native_url, backend_model, thinking_enabled),
        );

        let (nresp, _native_attempts) = send_upstream_retry(
            state.cfg.request_retry,
            || {
                let nreq = state.http_client.post(&native_url).json(&native_body);
                if let Some(k) = upstream.api_key.clone() {
                    nreq.header("Authorization", format!("Bearer {}", k))
                } else {
                    nreq
                }
            },
        ).await;
        match nresp {
            Ok(r) => {
                let status = r.status();
                if !status.is_success() {
                    let eb = r.bytes().await.unwrap_or_default();
                    return (status, eb).into_response();
                }
                let raw = r.json::<Value>().await;
                match raw {
                    Ok(nv) => {
                        let msg = nv.get("message").cloned().unwrap_or(json!({}));
                        let mut out_msg = json!({
                            "role": msg.get("role").cloned().unwrap_or(json!("assistant")),
                            "content": msg.get("content").cloned().unwrap_or(json!("")),
                        });
                        if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                            if !tcs.is_empty() {
                                // Ollama-native tool_calls omit the OpenAI-required
                                // "type":"function" field; the SDK's schema-validated
                                // response handler rejects them → "Invalid JSON response"
                                // (kills the agentic skill extractor). Normalize each entry.
                                let normalized: Vec<Value> = tcs.iter().map(|tc| {
                                    let mut e = tc.clone();
                                    match e.get("type") {
                                        None => e["type"] = json!("function"),
                                        Some(t) if t.is_null() => e["type"] = json!("function"),
                                        Some(_) => {}
                                    }
                                    // OpenAI schema requires function.arguments to be a
                                    // JSON-encoded STRING; Ollama-native returns it as an
                                    // object → SDK schema validation rejects → "Invalid JSON
                                    // response". Serialize when it's an object/array.
                                    if let Some(fnv) = e.clone().get("function") {
                                        if let Some(args) = fnv.get("arguments") {
                                            if !args.is_null() && !args.is_string() {
                                                let mut ee = e;
                                                ee["function"] = json!({
                                                    "name": fnv.get("name").cloned().unwrap_or(json!("")),
                                                    "arguments": json!(serde_json::to_string(args).unwrap_or_default()),
                                                });
                                                return ee;
                                            }
                                        }
                                    }
                                    e
                                }).collect();
                                out_msg["tool_calls"] = json!(normalized);
                            }
                        }
                        let openai = json!({
                            "id": "chatcmpl-ollama-native",
                            "object": "chat.completion",
                            "created": 0,
                            "model": backend_model,
                            "choices": json!([json!({
                                "index": 0,
                                "message": out_msg,
                                "finish_reason": "stop",
                            })]),
                            "usage": json!({
                                "prompt_tokens": nv.get("prompt_eval_count").cloned().unwrap_or(json!(0)),
                                "completion_tokens": nv.get("eval_count").cloned().unwrap_or(json!(0)),
                                "total_tokens": 0,
                            }),
                        });
                        let body_bytes = serde_json::to_vec(&openai).unwrap_or_default();
                        return (
                            StatusCode::OK,
                            axum::body::Bytes::from(body_bytes),
                        )
                            .into_response();
                    }
                    Err(e) => {
                        logger::error("chat", &format!("Ollama native response unparseable: {}", e));
                        return (
                            StatusCode::BAD_GATEWAY,
                            format!("Ollama native parse error: {}", e),
                        )
                            .into_response();
                    }
                }
            }
            Err(e) => {
                logger::error("chat", &format!("Ollama native POST failed: {}", e));
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("Ollama native error: {}", e),
                )
                    .into_response();
            }
        }
    }

    let (resp, _cc_attempts) = send_upstream_retry(
        state.cfg.request_retry,
        || {
            let req = state
                .http_client
                .post(&url)

                .header("Accept", "text/event-stream")
                .json(&request_json);
                let req = if let Some(k) = upstream.api_key.clone() { req.header("Authorization", format!("Bearer {}", k)) } else { req };
                req
        },
    ).await;

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

                let conn_id = logger::current_conn_id();
                tokio::spawn(async move {
                    let inner = async {
                   let mut stream = stream;
                   let mut buffer = String::new();
                   let mut read_pos: usize = 0;
                   let mut sse_keepalive = tokio::time::interval(Duration::from_secs(15));
                   sse_keepalive.tick().await; // consume immediate first tick
                   // In-stream error detection: vLLM/TGI backends sometimes return
                   // error objects (e.g. {"error":{"message":"...","code":429}}) as
                   // `data:` lines with no `choices` key. Without this, the
                   // passthrough forwards them verbatim and the client
                   // silently skips them, then treats the stream end as a successful
                   // empty response → premature end_turn. When detected, we inject
                   // an SSE error event so the client can retry/fallback.
                   let mut upstream_error: Option<(u16, String)> = None;
                   // Thinking-only guard state: GLM-5.x with thinking enabled
                   // streams the ENTIRE response via `delta.reasoning` and emits
                   // no `delta.content` (or only whitespace). The rewrite below
                   // promotes that reasoning to `content`, so to the client the
                   // turn looks like a (short) real answer followed by
                   // finish_reason=stop → end_turn. Detect the "reasoning but
                   // never any real content/tool call" case and, at stream end,
                   // inject an error event instead of a clean [DONE] so the
                   // client retries/falls back instead of ending the turn on a
                   // thinking trace. See also the non-streaming branch below.
                   let mut saw_reasoning = false;
                   let mut saw_real_content = false;
                   let mut saw_tool_calls = false;
                   let mut pending_done = false;

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
                                          // detection so the client (which uses this
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
                                          if !pending_done {
                                              // Defer `[DONE]` until the end of the
                                              // stream so the empty-stream guard can
                                              // inject an error before the client sees
                                              // a clean completion. This applies to
                                              // BOTH thinking and non-thinking paths —
                                              // vLLM sometimes returns a completely
                                              // empty stream (just [DONE]) on
                                              // high-context requests, and without
                                              // deferring [DONE] the client treats it
                                              // as a successful empty response.
                                              if output.trim() == "data: [DONE]" {
                                                  pending_done = true;
                                                  continue;
                                              }
                                              // Classify the raw line (for both
                                              // thinking and non-thinking paths) so
                                              // the end-of-stream guard knows whether
                                              // any real content/tool_calls were sent.
                                              if let Some(flags) = inspect_chat_delta(line) {
                                                  if flags.has_reasoning {
                                                      saw_reasoning = true;
                                                  }
                                                  if flags.has_content {
                                                      saw_real_content = true;
                                                  }
                                                  if flags.has_tool_calls {
                                                      saw_tool_calls = true;
                                                  }
                                              }
                                          }
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
                         if !pending_done && output.trim() == "data: [DONE]" {
                             pending_done = true;
                         } else {
                           let framed = format!("{}\n", output);
                           let _ = tx.send(Ok(axum::body::Bytes::from(framed))).await;
                          }
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
                 } else if thinking_enabled && saw_reasoning && !saw_real_content && !saw_tool_calls {
                      // Thinking-only stream: reasoning deltas were already
                      // rewritten to content and forwarded, but the model never
                      // produced a real answer or tool call. The client would
                      // otherwise see a clean `[DONE]` and treat the turn as
                      // complete (end_turn on a thinking trace). Inject an error
                      // event instead so the client/CCR retry: the client throws on a
                      // `data:` line with an `error` key and falls back to
                      // directChat, whose non-streaming re-run hits the guard
                      // below and returns 502 — triggering model fallback instead
                      // of ending the turn.
                      logger::error(
                          "chat",
                          "Thinking-only stream detected (reasoning, no content, no tool calls) — injecting error instead of [DONE]",
                      );
                      let _ = tx.send(Ok(axum::body::Bytes::from(
                          thinking_only_error_event().into_bytes(),
                      )))
                     .await;
                 } else if !saw_real_content && !saw_tool_calls && !saw_reasoning {
                     // Completely empty stream: vLLM returned no content deltas,
                     // no reasoning, no tool_calls — just [DONE]. This happens
                     // on high-context requests (150K+ tokens) when the model
                     // fails to generate any output. Without this guard, the
                     // client sees a clean stream end and treats it as a
                     // successful empty response → premature end_turn.
                     // Inject an error so the client retries/falls back.
                     logger::error(
                         "chat",
                         "Empty stream detected (no content, no reasoning, no tool_calls) — injecting error instead of [DONE]",
                     );
                     let _ = tx.send(Ok(axum::body::Bytes::from(
                         thinking_only_error_event().into_bytes(),
                     )))
                     .await;
                 } else if pending_done {
                      let _ = tx.send(Ok(axum::body::Bytes::from(
                          "data: [DONE]\n\n".to_string().into_bytes(),
                      )))
                      .await;
                 }
                    };
                    if let Some(id) = conn_id {
                        logger::set_conn_id(&id, inner).await;
                    } else {
                        inner.await;
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
                // Track whether ANY choice had only reasoning promoted to
                // content (i.e. no real answer text ever, no tool call) so a
                // thinking-only response returns 502 instead of a "successful"
                // short answer that the client turns into end_turn. Mirrors
                // the streaming thinking-only guard above.
                let mut thinking_only_nonstream = false;
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
                                            let has_tool_calls = msg
                                                .get("tool_calls")
                                                .and_then(|t| t.as_array())
                                                .map(|a| !a.is_empty())
                                                .unwrap_or(false);
                                            if !has_tool_calls {
                                                thinking_only_nonstream = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if thinking_only_nonstream {
                            logger::error(
                                "chat",
                                "Non-streaming thinking-only response detected (reasoning, no content, no tool calls) — returning 502",
                            );
                        }
                        serde_json::to_vec(&v).unwrap_or_else(|_| body_bytes.to_vec())
                    } else {
                        body_bytes.to_vec()
                    }
                } else {
                    body_bytes.to_vec()
                };
                let final_status = if thinking_only_nonstream {
                    StatusCode::BAD_GATEWAY
                } else {
                    status
                };
                let final_body = if thinking_only_nonstream {
                    serde_json::to_vec(&json!({
                        "error": {
                            "message": "Model produced only reasoning (thinking) with no final answer — retrying with different settings",
                            "code": 502,
                        }
                    }))
                    .unwrap_or_default()
                } else {
                    body_bytes
                };
                let mut response = Response::builder().status(final_status);
                for (k, v) in &headers {
                    if k != "content-encoding" && k != "transfer-encoding" {
                        response = response.header(k, v);
                    }
                }
                response.body(Body::from(final_body)).unwrap()
            }
        }
        Err(e) => {
            logger::error("chat", &format!("Upstream error: {}", e));
            (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e)).into_response()
        }
    }
}

// ─── POST /v1/embeddings (passthrough) ─────────────────────────────────────
/// Thin non-streaming passthrough for OpenAI-compatible `/v1/embeddings`.
///
/// The upstream provider already serves embedding models (e.g.
/// `BAAI/bge-m3`, `Qwen/Qwen3-Embedding-8B`, `intfloat/e5-mistral-7b-instruct`,
/// `nomic-ai/nomic-embed-text-v1.5`); this route just exposes them through the
/// proxy's auth/rate-limit layer so downstream clients (e.g. the TencentDB
/// Agent Memory gateway) can point their embedding `baseUrl` at the proxy
/// instead of the raw upstream. The request and response are forwarded
/// verbatim — only the `model` alias is resolved to its backend id, matching
/// `handle_chat_completions`.
pub async fn handle_embeddings(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let mut request_json: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            logger::error("embeddings", &format!("Invalid JSON: {}", e));
            return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)).into_response();
        }
    };

    let model_alias = request_json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    // Resolve alias → backend model id (e.g. "bge-m3" → "BAAI/bge-m3/...").
    let backend_model = state.cfg.resolve_model(&model_alias);
    request_json["model"] = json!(backend_model);

        let upstream = state.cfg.upstream_for(&backend_model);
        let base_url = upstream.base_url;
    let url = format!("{}/embeddings", base_url);

    logger::info(
        "embeddings",
        &format!("Passthrough POST {} model={}", url, backend_model),
    );

    let (resp, _embed_attempts) = send_upstream_retry(
        state.cfg.request_retry,
        || {
            let req = state
                .http_client
                .post(&url);
                let req = if let Some(k) = upstream.api_key.clone() { req.header("Authorization", format!("Bearer {}", k)) } else { req };
                req

                .json(&request_json)
        },
    ).await;

    match resp {
        Ok(r) => {
            let status = r.status();
            let headers = r.headers().clone();
            // Embedding responses are small JSON blobs — never streamed. Forward
            // the body verbatim (no reasoning/content rewrite needed).
            let body_bytes = r.bytes().await.unwrap_or_default();
            let mut response = Response::builder().status(status);
            for (k, v) in &headers {
                if k != "content-encoding" && k != "transfer-encoding" {
                    response = response.header(k, v);
                }
            }
            response.body(Body::from(body_bytes)).unwrap()
        }
        Err(e) => {
            logger::error("embeddings", &format!("Upstream error: {}", e));
            (StatusCode::BAD_GATEWAY, format!("Upstream error: {}", e)).into_response()
        }
    }
}

// ─── POST /v1/messages (Anthropic Messages API → OpenAI Chat Completions) ───
/// Longest already-streamed assistant text that still counts as a stalled turn
/// worth re-running non-streaming. Measured stalls carried 0–110 characters; a
/// genuine final answer runs into the hundreds or thousands and is left alone.
const NOARGS_RETRY_MAX_TEXT: usize = 400;

/// True if any tool in an Anthropic request declares no parameters. Only those
/// trip the backend's streaming tool-call defect, so a request without one
/// never pays for the recovery path.
fn has_zero_parameter_tool(request_json: &Value) -> bool {
    request_json
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|tools| {
            tools.iter().any(|t| {
                !t.get("input_schema")
                    .and_then(|s| s.get("properties"))
                    .and_then(|p| p.as_object())
                    .map(|p| !p.is_empty())
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

/// True for models whose streaming tool-call parser drops empty-argument calls.
/// Scoped to GLM: DeepSeek-V4 streams them correctly (verified).
fn needs_noargs_retry(backend_model: &str) -> bool {
    backend_model.to_lowercase().contains("glm")
}

/// True if any tool in a Responses-API request declares no parameters.
///
/// Responses-API tools are `{"type":"function","name":...,"parameters":{...}}`
/// (flat, unlike Anthropic's nested `input_schema`), and `type":"namespace"`
/// entries nest their own `tools` array. Custom/freeform tools (`apply_patch`,
/// `exec`) take raw text rather than a JSON object, so they are not affected and
/// are skipped.
fn responses_request_has_zero_parameter_tool(request_json: &Value) -> bool {
    fn schema_is_empty(tool: &Value) -> bool {
        // Only plain function tools use a JSON-object argument schema.
        if tool.get("type").and_then(|t| t.as_str()) != Some("function") {
            return false;
        }
        !tool
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
            .map(|p| !p.is_empty())
            .unwrap_or(false)
    }
    let Some(tools) = request_json.get("tools").and_then(|t| t.as_array()) else {
        return false;
    };
    tools.iter().any(|tool| {
        if tool.get("type").and_then(|t| t.as_str()) == Some("namespace") {
            return tool
                .get("tools")
                .and_then(|t| t.as_array())
                .map(|subs| subs.iter().any(schema_is_empty))
                .unwrap_or(false);
        }
        schema_is_empty(tool)
    })
}

/// Replay a chat-completions request with `stream: false` and return
/// `choices[0].message`, or None if the re-run fails or carries no message.
///
/// Used only to recover a tool call the streaming response dropped, so failures
/// are logged and swallowed: the client still gets the original turn.
async fn retry_non_streaming(
    client: &reqwest::Client,
    url: &str,
    cc_request: &Value,
    api_key: Option<&str>,
) -> Option<Value> {
    let mut body = cc_request.clone();
    if let Some(obj) = body.as_object_mut() {
        obj.insert("stream".to_string(), json!(false));
        obj.remove("stream_options");
    }
    let req = client.post(url);
    let req = if let Some(k) = api_key {
        req.header("Authorization", format!("Bearer {}", k))
    } else {
        req
    };
    let resp = req
        .json(&body)
        .send()
        .await;
    let resp = match resp {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            logger::warn(
                "messages",
                &format!("Non-streaming tool-call re-run returned {}", r.status()),
            );
            return None;
        }
        Err(e) => {
            logger::warn(
                "messages",
                &format!("Non-streaming tool-call re-run failed: {}", e),
            );
            return None;
        }
    };
    let parsed: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            logger::warn(
                "messages",
                &format!("Non-streaming tool-call re-run returned unparseable JSON: {}", e),
            );
            return None;
        }
    };
    parsed
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .cloned()
}

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

    // Per-model concurrency throttle — hold spot for the whole upstream call.
    let _permit = match throttle_model(&state, &backend_model).await {
        Ok(p) => p,
        Err(e) => return e,
    };

    let is_stream = request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Vision capability: a model whose catalog entry declares `image` input
    // modality can consume image_url content parts; text-only models must
    // have image blocks stripped before translation or the backend returns a
    // hard 500 ("not a multimodal model").
    let vision_capable = get_model_supports_images(&state, &backend_model);

    // Translate Anthropic → OpenAI Chat Completions
    let mut cc_request = translate_anthropic::anthropic_to_chat_completions_with_vision(
        &backend_model,
        &request_json,
        is_stream,
        vision_capable,
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
    apply_configured_logit_bias(&mut cc_request, &state.cfg, &backend_model);

        let upstream = state.cfg.upstream_for(&backend_model);
        let base_url = upstream.base_url;
    let url = format!("{}/chat/completions", base_url);

    logger::info(
        "messages",
        &format!("POST {} model={} stream={}", url, backend_model, is_stream),
    );

    // Log which tools declare no parameters. Those are the ones GLM-5.2 drops
    // in streaming mode (see NOARGS_PLACEHOLDER), so knowing whether a client
    // actually sends any is the difference between the workaround mattering and
    // being dead weight.
    if let Some(tools) = request_json.get("tools").and_then(|t| t.as_array()) {
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
            .collect();
        let noargs: Vec<&str> = tools
            .iter()
            .filter(|t| {
                !t.get("input_schema")
                    .and_then(|s| s.get("properties"))
                    .and_then(|p| p.as_object())
                    .map(|p| !p.is_empty())
                    .unwrap_or(false)
            })
            .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
            .collect();
        logger::debug(
            "messages",
            &format!(
                "[tools] {} offered: [{}] | zero-parameter: [{}]",
                names.len(),
                names.join(","),
                noargs.join(",")
            ),
        );
        if !noargs.is_empty() {
            logger::info(
                "messages",
                &format!(
                    "[tools] {} of {} tools declare no parameters: [{}]",
                    noargs.len(),
                    tools.len(),
                    noargs.join(",")
                ),
            );
        }
    }

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
        let context_window = get_context_window(&state, &model);
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

    let (resp, _msg_attempts) = send_upstream_retry(
        state.cfg.request_retry,
        || {
            let req = state
                .http_client
                .post(&url)

                .header("Accept", "text/event-stream");
                let req = if let Some(k) = upstream.api_key.clone() { req.header("Authorization", format!("Bearer {}", k)) } else { req };
                req
                .json(&cc_request)
        },
    ).await;

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

                // Everything the dropped-tool-call recovery needs to replay the
                // request non-streaming (see retry_non_streaming).
                let retry_noargs_tool_call =
                    has_zero_parameter_tool(&request_json) && needs_noargs_retry(&backend_model);
                let retry_client = state.http_client.clone();
                let retry_url = url.clone();
                let retry_api_key = upstream.api_key.clone();
                            let retry_request = cc_request.clone();

                let model_for_translator = backend_model.clone();
                let conn_id = logger::current_conn_id();
                tokio::spawn(async move {
                    let inner = async {
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
                                            // Defer `[DONE]` to the post-loop
                                            // path. translate_line() turns it
                                            // straight into finish(), which
                                            // emits message_delta/message_stop —
                                            // that must happen AFTER the
                                            // dropped-tool-call recovery below,
                                            // or the recovered tool_use blocks
                                            // land after the message has already
                                            // been closed.
                                            if line.trim() == "data: [DONE]" {
                                                continue;
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
                        // Same `[DONE]` deferral as the main loop: finish() must
                        // run after the dropped-tool-call recovery, not here.
                        if !remaining.trim().is_empty()
                            && remaining.trim() != "data: [DONE]"
                            && upstream_error.is_none()
                        {
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
                        // If the upstream stream ended without a finish_reason
                        // (silent stall, connection drop), the translator's
                        // stop_reason is empty. finish() would default to
                        // "end_turn", making Claude Code think the model chose
                        // to stop — causing premature end_turn on short
                        // responses. Emit an error event instead so the client
                        // can retry.
                        if !translator.has_finish_reason() {
                            logger::error("messages", "Upstream stream ended without finish_reason — sending error instead of end_turn");
                            for event in translator.abort_for_error() {
                                let _ = tx.send(Ok(axum::body::Bytes::from(event))).await;
                            }
                            let err_payload = json!({
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": "Upstream stream ended without completion signal — response may be truncated"
                                }
                            });
                            let sse = format!("event: error\ndata: {}\n\n", err_payload);
                            let _ = tx.send(Ok(axum::body::Bytes::from(sse))).await;
                        } else {
                           // GLM-5.2-FP8 drops streamed tool calls whose
                           // arguments object is empty (verified against the
                           // backend with no proxy in the path): the turn comes
                           // back as finish_reason "stop" with no tool_calls and
                           // at most a short preamble, so the client renders a
                           // turn that announced an action and did nothing. The
                           // same request non-streaming returns the tool call
                           // every time, so re-run it once and splice the tool
                           // call in. Gated to requests that actually offer a
                           // zero-parameter tool, and to turns whose streamed
                           // text is short, so a real final answer never pays
                           // for the extra call.
                           if retry_noargs_tool_call
                               && translator.should_retry_non_streaming(NOARGS_RETRY_MAX_TEXT)
                           {
                               if let Some(message) = retry_non_streaming(
                                   &retry_client,
                                   &retry_url,
                                       &retry_request,
                                   retry_api_key.as_deref(),
                               )
                               .await
                               {
                                   let events =
                                       translator.inject_tool_calls_from_message(&message);
                                   if !events.is_empty() {
                                       logger::info(
                                           "messages",
                                           "Recovered a dropped tool call via a non-streaming re-run",
                                       );
                                       for event in events {
                                           let _ = tx
                                               .send(Ok(axum::body::Bytes::from(event)))
                                               .await;
                                       }
                                   }
                               }
                           }
                           let finish_events = translator.translate_line("data: [DONE]");
                           for event in finish_events {
                               let _ = tx.send(Ok(axum::body::Bytes::from(event))).await;
                           }
                       }
                   }
                   };
                    if let Some(id) = conn_id {
                        logger::set_conn_id(&id, inner).await;
                    } else {
                        inner.await;
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
    apply_configured_logit_bias(&mut cc_payload, &state.cfg, &backend_model);

        let upstream = state.cfg.upstream_for(&backend_model);
        let base_url = upstream.base_url;
    let url = format!("{}/chat/completions", base_url);

    // Context size warning: estimate tokens and log if exceeds model's context window.
    // We do NOT trim here — codex has its own context management (auto-compaction
    // at 90% of context_window, tool_output_token_limit, etc.). The proxy's
    // trimming was broken: it only counted `content` as a string (missing array
    // content, tool_calls, tool results, and the tools schema entirely), and the
    // `total > 4` guard allowed huge single messages through. Removing it lets
    // codex's proven context management handle this correctly.
    //
    // NEW: hard-reject before forwarding when the estimated tokens exceed the
    // model's effective window (see context_overflow_response). A compact
    // request that is itself over-window can't be compacted further by the
    // backend — reject it cleanly instead of letting the shared vLLM 400/503.
    let context_window = get_context_window(&state, &model_alias);
    let estimated_tokens = estimate_tokens(&cc_payload);
    if estimated_tokens as u64 > context_window {
        logger::warn("compact", &format!(
            "Context size warning: ~{} tokens may exceed context_window {} — codex should auto-compact",
            estimated_tokens, context_window
        ));
    }
    if let Some(resp) = context_overflow_response(&state, &model_alias, estimated_tokens) {
        return resp;
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

    let (resp_result, upstream_attempts) = send_upstream_retry(
        state.cfg.request_retry,
        || {
            let req = state
                .http_client
                .post(&url)

                .header("Accept", "text/event-stream");
                let req = if let Some(k) = upstream.api_key.clone() { req.header("Authorization", format!("Bearer {}", k)) } else { req };
                req
                .json(&cc_payload)
        },
    ).await;
        let resp = match resp_result
    {
        Ok(r) => r,
        Err(e) => {
            logger::error("compact", &format!("Upstream error after {} attempt(s): {}", upstream_attempts, e));
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
//   1. Query the backend /models endpoint for models with >= 1M
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
    // Client may override the model (e.g. "qwen38-27b"). Resolved via alias,
    // then falls back to the existing auto-selection strategy.
    let requested_model = request_json
        .get("model")
        .and_then(|m| m.as_str())
        .filter(|m| !m.is_empty());

    let summary_model = match requested_model {
        Some(alias) => {
            let resolved = state.cfg.resolve_model(alias);
            logger::info(
                "summarize",
                &format!("Using client-requested model: {} → {}", alias, resolved),
            );
            resolved
        }
        None => select_summary_model(&state).await,
    };
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

    // ── Send to backend ───────────────────────────────────────────────────
        let upstream = state.cfg.upstream_for(&summary_model);
        let base_url = upstream.base_url;
    let url = format!("{}/chat/completions", base_url);

    logger::info(
        "summarize",
        &format!("POST {} model={}", url, summary_model),
    );

    let (resp_result, upstream_attempts) = send_upstream_retry(
        state.cfg.request_retry,
        || {
            let req = state
                .http_client
                .post(&url)

                .header("Accept", "text/event-stream");
                let req = if let Some(k) = upstream.api_key.clone() { req.header("Authorization", format!("Bearer {}", k)) } else { req };
                req
                .json(&cc_payload)
        },
    ).await;
        let resp = match resp_result
    {
        Ok(r) => r,
        Err(e) => {
            logger::error("summarize", &format!("Upstream error after {} attempt(s): {}", upstream_attempts, e));
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
/// 1. Query backend /models for 1M+ context models that are loaded.
/// 2. Fall back to the largest context_window model from the catalog.
/// 3. Fall back to the first model in config.
async fn select_summary_model(state: &AppState) -> String {
    // ── Strategy 1: Query backend for 1M models ───────────────────────────
        // `provider()` is Some for a running service — main() exits when it is
        // None — but with no provider there is no backend to probe, so fall
        // through to the catalog/first-model fallbacks instead of unwrapping.
        let base_url = match state.cfg.provider() {
            Some(p) => p.base_url.clone(),
            None => {
                logger::info(
                    "summarize",
                    "No provider configured — skipping the backend 1M-model probe",
                );
                return select_summary_model_from_catalog(state);
            },
        };
    let models_url = format!("{}/models", base_url);

    let resp = state
        .http_client
        .get(&models_url)

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

#[cfg(test)]
mod tests {
    use super::*;

    // ─── poison-tolerant lock recovery ───────────────────────────────────
    // A panic while a std::sync::Mutex is held poisons it. Without recovery
    // every later lock().unwrap() panics too, so the FIRST bad request kills
    // that call site forever. These pin that recovery hands back the payload,
    // and keeps doing so on every subsequent request (poison is permanent).

    #[test]
    fn poisoned_lock_still_yields_its_payload() {
        // Arc so the poisoning thread can own a handle with 'static lifetime.
        let mutex = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        {
            let mut held = mutex.lock().unwrap();
            held.insert("key".to_string(), 1);
        }
        // Poison by panicking while the guard is alive, in a child thread so
        // the test process itself survives.
        let shared = Arc::clone(&mutex);
        let child = std::thread::spawn(move || {
            let mut held = shared.lock().unwrap();
            held.insert("child".to_string(), 2);
            panic!("simulated panic while holding the lock");
        });
        assert!(child.join().is_err(), "the child panicked and poisoned the lock");

        // Every later lock() now returns Err(PoisonError) — permanently. The
        // helper must recover the payload each time, not just the first time.
        let first = recover_poisoned_lock(mutex.lock(), "test mutex");
        assert_eq!(first.get("key").copied(), Some(1));
        drop(first);
        let second = recover_poisoned_lock(mutex.lock(), "test mutex");
        assert_eq!(second.get("key").copied(), Some(1));
    }

    // ─── select_summary_model must not trust provider() being Some ────────

    #[tokio::test]
    async fn summarize_skips_probe_when_no_provider_configured() {
        // test_state() builds a config with NO providers — the exact shape that
        // used to make provider().unwrap() panic. With no backend there is
        // nothing to probe, so the fallback chain must answer instead.
        let state = test_state(None);
        assert_eq!(
            select_summary_model(&state).await,
            "Qwen/Qwen2.5-14B-Instruct-1M".to_string(),
        );
    }

    // ─── inspect_chat_delta / thinking-only guard tests ───────────────────

    #[test]
    fn inspect_delta_content_only() {
        let flags = inspect_chat_delta("data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}").unwrap();
        assert!(flags.has_content);
        assert!(!flags.has_reasoning);
        assert!(!flags.has_tool_calls);
    }

    #[test]
    fn inspect_delta_reasoning_only() {
        let flags = inspect_chat_delta("data: {\"choices\":[{\"delta\":{\"reasoning\":\"thinking...\"}}]}").unwrap();
        assert!(!flags.has_content);
        assert!(flags.has_reasoning);
        assert!(!flags.has_tool_calls);
    }

    #[test]
    fn inspect_delta_reasoning_and_content() {
        let flags = inspect_chat_delta("data: {\"choices\":[{\"delta\":{\"reasoning\":\"think\",\"content\":\"answer\"}}]}").unwrap();
        assert!(flags.has_content);
        assert!(flags.has_reasoning);
        assert!(!flags.has_tool_calls);
    }

    #[test]
    fn inspect_delta_tool_calls() {
        let flags = inspect_chat_delta("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"x\"}]}}]}").unwrap();
        assert!(!flags.has_content);
        assert!(!flags.has_reasoning);
        assert!(flags.has_tool_calls);
    }

    #[test]
    fn inspect_delta_whitespace_content_is_not_real() {
        let flags = inspect_chat_delta("data: {\"choices\":[{\"delta\":{\"content\":\"  \"}}]}").unwrap();
        assert!(!flags.has_content);
    }

    #[test]
    fn inspect_delta_done_and_junk_return_none() {
        assert!(inspect_chat_delta("data: [DONE]").is_none());
        assert!(inspect_chat_delta(": keepalive").is_none());
        assert!(inspect_chat_delta("not json").is_none());
    }

    #[test]
    fn inspect_delta_usage_chunk_no_flags() {
        let flags = inspect_chat_delta("data: {\"choices\":[],\"usage\":{\"total_tokens\":5}}").unwrap();
        assert!(!flags.has_content);
        assert!(!flags.has_reasoning);
        assert!(!flags.has_tool_calls);
    }

    #[test]
    fn thinking_only_error_event_has_error_key() {
        let ev = thinking_only_error_event();
        assert!(ev.starts_with("data: {"));
        let parsed: Value = serde_json::from_str(ev.trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(parsed["error"]["code"], 502);
    }

    #[test]
    fn rewrite_reasoning_moves_reasoning_to_content() {
        let line = "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think text\"}}]}";
        let out = rewrite_reasoning_to_content(line);
        assert!(out.contains("\"content\":\"think text\""));
        assert!(!out.contains("\"reasoning\""));
    }

    #[test]
    fn rewrite_reasoning_keeps_real_content() {
        let line = "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think\",\"content\":\"answer\"}}]}";
        let out = rewrite_reasoning_to_content(line);
        assert!(out.contains("\"content\":\"answerthink\""));
    }

    #[test]
    fn rewrite_reasoning_passthrough_done() {
        assert_eq!(
            rewrite_reasoning_to_content("data: [DONE]"),
            "data: [DONE]"
        );
    }

    // --- resolve_conn_id: per-connection UUID ---
    // The fix for "same connection should have the same UUID" in the
    // /v1/chat/completions passthrough. `resolve_conn_id` reuses the cached
    // UUID for the same peer key, so a burst of requests over one TCP
    // connection shares a single identifier.

    #[test]
    fn resolve_conn_id_same_key_reuses_uuid() {
        let mut cache = std::collections::HashMap::new();
        let key = "127.0.0.1:52341";
        let first = resolve_conn_id(&mut cache, key);
        let second = resolve_conn_id(&mut cache, key);
        // Same connection -> same UUID (the fix).
        assert_eq!(first, second);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn resolve_conn_id_returns_8_hex_chars() {
        let mut cache = std::collections::HashMap::new();
        let id = resolve_conn_id(&mut cache, "10.0.0.1:8080");
        assert_eq!(id.len(), 8);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()), "id not hex: {}", id);
    }

    #[test]
    fn resolve_conn_id_distinct_keys_get_distinct_uuids() {
        let mut cache = std::collections::HashMap::new();
        let a = resolve_conn_id(&mut cache, "127.0.0.1:1000");
        let b = resolve_conn_id(&mut cache, "127.0.0.1:2000");
        // Different TCP connections -> different UUIDs.
        assert_ne!(a, b);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn resolve_conn_id_bounds_cache_at_max() {
        let mut cache = std::collections::HashMap::new();
        // Fill past the cap so eviction kicks in.
        for i in 0..(MAX_CONN_ID_CACHE + 10) {
            resolve_conn_id(&mut cache, &format!("10.0.0.{}:{}", i % 250, 1000 + i));
        }
        // The cache must never exceed the cap.
        assert!(cache.len() <= MAX_CONN_ID_CACHE, "cache grew to {}", cache.len());
        // And it should sit right at the cap after eviction+insert.
        assert_eq!(cache.len(), MAX_CONN_ID_CACHE);
    }

    #[test]
    fn resolve_conn_id_evicts_when_over_cap() {
        // std::collections::HashMap does NOT guarantee insertion order, so we
        // can't assert which specific key is evicted — only that the cache
        // stays bounded and the newest key is always present.
        let mut cache = std::collections::HashMap::new();
        // Fill past the cap so eviction kicks in.
        for i in 0..(MAX_CONN_ID_CACHE + 5) {
            resolve_conn_id(&mut cache, &format!("10.0.2.{}:{}", i % 250, 3000 + i));
        }
        // The cache must never exceed the cap.
        assert!(cache.len() <= MAX_CONN_ID_CACHE);
        // The most recently inserted key is always retrievable (it was just
        // added, so it must be present regardless of which entry was evicted).
        let newest = format!("10.0.2.{}:{}", (MAX_CONN_ID_CACHE + 4) % 250, 3000 + MAX_CONN_ID_CACHE + 4);
        assert!(cache.contains_key(&newest));
    }

    // ─── get_model_supports_images tests ──────────────────────────────────

    fn catalog_with_modalities(model: &str, modalities: &[&str]) -> Value {
        let mods: Vec<Value> = modalities.iter().map(|m| json!(m)).collect();
        json!({
            "models": [
                {
                    "slug": model,
                    "input_modalities": mods,
                }
            ]
        })
    }

    #[test]
    fn supports_images_when_catalog_declares_image_modality() {
        let catalog = catalog_with_modalities("qwen3.8-27b", &["text", "image"]);
        let state = test_state(Some(catalog));
        assert!(get_model_supports_images(&state, "qwen3.8-27b"));
    }

    #[test]
    fn does_not_support_images_when_text_only() {
        let catalog = catalog_with_modalities("llama-3.1-8b", &["text"]);
        let state = test_state(Some(catalog));
        assert!(!get_model_supports_images(&state, "llama-3.1-8b"));
    }

    #[test]
    fn does_not_support_images_when_catalog_entry_missing() {
        let catalog = catalog_with_modalities("known", &["text"]);
        let state = test_state(Some(catalog));
        assert!(!get_model_supports_images(&state, "unknown-model"));
    }

    #[test]
    fn does_not_support_images_when_catalog_absent() {
        let state = test_state(None);
        assert!(!get_model_supports_images(&state, "qwen3.8-27b"));
    }

    // ── config.yaml is the primary metadata source ───────────────────────
    // The proxy's own models are declared in the provider blocks, not in Codex's
    // catalog, so a declared modality must win even with no catalog on disk —
    // otherwise every image block is stripped unconditionally and a vision
    // backend (qwen3.6:35b carries a CLIP projector) never sees the image.

    /// Config with two providers: one declares image modalities, one is text-only.
    fn state_with_provider_models(catalog: Option<Value>) -> AppState {
        let yaml = r#"
port: 8317
auth-dir: "/tmp"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "http://192.168.124.14:8080/v1"
    models:
      - name: qwen-big
        alias: qwen-big
        context_length: 262144
        input_modalities: [text]
  - name: "mac"
    prefix: "mac"
    base-url: "http://127.0.0.1:11434/v1"
    models:
      - name: "qwen3.6:35b"
        alias: qwen36-35b
        context_length: 32768
        input_modalities: [text, image]
"#;
        let cfg: config::Config = serde_yaml::from_str(yaml).expect("test config");
        AppState {
            cfg: Arc::new(cfg),
            http_client: reqwest::Client::new(),
            model_catalog: Arc::new(catalog),
            shutdown: tokio_util::sync::CancellationToken::new(),
            rate_limiter: Arc::new(RateLimiter::new(0)),
            model_throttle: Arc::new(ModelThrottle::new(3, &std::collections::HashMap::new())),
            conn_id_cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    #[test]
    fn supports_images_from_config_declared_modalities_without_catalog() {
        let state = state_with_provider_models(None);
        assert!(get_model_supports_images(&state, "qwen36-35b"));
        assert!(!get_model_supports_images(&state, "qwen-big"));
    }

    #[test]
    fn config_modalities_override_the_catalog() {
        // Catalog claims the mac alias is text-only; config.yaml says it takes
        // images. config.yaml wins — it is the proxy's own model list.
        let catalog = catalog_with_modalities("qwen36-35b", &["text"]);
        let state = state_with_provider_models(Some(catalog));
        assert!(get_model_supports_images(&state, "qwen36-35b"));
    }

    #[test]
    fn context_window_comes_from_config_when_catalog_is_absent() {
        // /v1/models advertises 262144 for the strata slug; the overflow guard
        // must use the same number, not the DEFAULT_CONTEXT_WINDOW fallback.
        let state = state_with_provider_models(None);
        assert_eq!(get_context_window(&state, "qwen-big"), 262144);
        assert_eq!(effective_context_window(&state, "qwen-big"), 262144);
        assert_eq!(get_context_window(&state, "mac/qwen36-35b"), 32768);
    }

    fn cfg_with_logit_bias() -> config::Config {
        let yaml = r#"
port: 8317
auth-dir: "/tmp"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "http://192.168.124.14:8080/v1"
    logit-bias:
      "248044": -100
      "248045": -100
    models:
      - name: coder-pack
        alias: coder-pack
  - name: "mac"
    prefix: "mac"
    base-url: "http://127.0.0.1:11434/v1"
    models:
      - name: plain
        alias: plain
      - name: tuned
        alias: tuned
        logit_bias:
          "248045": -50
"#;
        serde_yaml::from_str(yaml).expect("test config")
    }

    #[test]
    fn provider_logit_bias_is_merged_into_the_payload() {
        let cfg = cfg_with_logit_bias();
        let mut payload = json!({"model": "coder-pack", "messages": []});
        apply_configured_logit_bias(&mut payload, &cfg, "coder-pack");
        let bias = payload.get("logit_bias").expect("bias merged");
        assert_eq!(bias.get("248044").and_then(|v| v.as_f64()), Some(-100.0));
        assert_eq!(bias.get("248045").and_then(|v| v.as_f64()), Some(-100.0));
    }

    #[test]
    fn provider_logit_bias_reaches_undeclared_model_names() {
        // The strata server is called "strata-ubuntu" / "strata-proxy" on the
        // wire — names no model entry declares. They route to the default
        // provider with the key echoed as model ID, and the provider's guard
        // must still apply (this is the exact shape of the 2026-10-29
        // premature-end_turn traffic).
        let cfg = cfg_with_logit_bias();
        assert!(cfg.upstream_target("strata-ubuntu").is_none());
        let mut payload = json!({"model": "strata-ubuntu", "messages": []});
        apply_configured_logit_bias(&mut payload, &cfg, "strata-ubuntu");
        let bias = payload.get("logit_bias").expect("provider bias applied");
        assert_eq!(bias.get("248045").and_then(|v| v.as_f64()), Some(-100.0));
    }

    #[test]
    fn configured_logit_bias_wins_over_a_client_sent_conflict() {
        let cfg = cfg_with_logit_bias();
        let mut payload = json!({
            "model": "coder-pack",
            "logit_bias": {"248045": 5, "100": 2}
        });
        apply_configured_logit_bias(&mut payload, &cfg, "coder-pack");
        let bias = payload.get("logit_bias").expect("bias merged");
        // The guard is not a preference: a client cannot un-ban the boundary
        // token, but its own unrelated entries survive the merge.
        assert_eq!(bias.get("248045").and_then(|v| v.as_f64()), Some(-100.0));
        assert_eq!(bias.get("100").and_then(|v| v.as_f64()), Some(2.0));
    }

    #[test]
    fn model_logit_bias_overrides_single_tokens_of_the_provider_map() {
        let cfg = cfg_with_logit_bias();
        let bias = cfg.effective_logit_bias("tuned").expect("merged bias");
        // The mac provider declares nothing, so the model map stands alone…
        assert_eq!(bias.get("248045"), Some(&-50.0));
        // …while the default provider's model inherits every provider token.
        let cfg_yaml: config::Config = serde_yaml::from_str(r#"
port: 8317
auth-dir: "/tmp"
openai-compatibility:
  - name: "default"
    prefix: "default"
    base-url: "http://192.168.124.14:8080/v1"
    logit-bias:
      "1": -100
      "2": -100
    models:
      - name: m
        alias: m
        logit_bias:
          "2": -10
"#).expect("test config");
        let merged = cfg_yaml.effective_logit_bias("m").expect("merged bias");
        assert_eq!(merged.get("1"), Some(&-100.0));
        assert_eq!(merged.get("2"), Some(&-10.0));
    }

    #[test]
    fn models_without_configured_bias_are_left_untouched() {
        let cfg = cfg_with_logit_bias();
        let mut payload = json!({"model": "plain", "messages": []});
        apply_configured_logit_bias(&mut payload, &cfg, "plain");
        assert!(payload.get("logit_bias").is_none());
    }

    #[test]
    fn context_window_falls_back_to_default_for_unknown_slug() {
        let state = state_with_provider_models(None);
        assert_eq!(
            get_context_window(&state, "slug-nobody-declares"),
            DEFAULT_CONTEXT_WINDOW,
        );
    }

    #[test]
    fn catalog_window_still_used_when_config_does_not_declare_it() {
        // A slug only the Codex catalog knows about (Anthropic-hosted) must
        // still resolve through the catalog fallback.
        let state = state_with_provider_models(Some(json!({ "models": [
            {"slug": "deepseek-v4-flash", "context_window": 1_048_576}
        ]})));
        assert_eq!(get_context_window(&state, "deepseek-v4-flash"), 1_048_576);
    }

    // ── Context-overflow hard-reject regression tests ────────────────────
    // These guard the fix for dropped/503'd requests when a single huge
    // injected message (e.g. a pasted config file) exceeds the model's
    // effective context window. The proxy must reject BEFORE forwarding so
    // the shared vLLM doesn't 400/503 the oversized request.

    /// Build a minimal AppState with an optional model catalog. Used only
    /// by tests. This branch (local/dev) has no token_mgr / max_context_cap.
    fn test_state(catalog: Option<Value>) -> AppState {
        let cfg: config::Config = serde_yaml::from_str(
            "port: 8317\nauth-dir: \"/tmp\"\n",
        )
        .expect("test config");
        AppState {
            cfg: Arc::new(cfg),
            http_client: reqwest::Client::new(),
            model_catalog: Arc::new(catalog),
            shutdown: tokio_util::sync::CancellationToken::new(),
            rate_limiter: Arc::new(RateLimiter::new(0)),
            model_throttle: Arc::new(ModelThrottle::new(3, &std::collections::HashMap::new())),
            conn_id_cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    #[test]
    fn context_overflow_rejects_over_window_request() {
        // Model catalog: deepseek-v4-flash has window 262144.
        let catalog = json!({ "models": [
            {"slug": "deepseek-v4-flash", "context_window": 262144}
        ]});
        let state = test_state(Some(catalog));
        // 1,035,488 estimated tokens (the observed oversized request) >> 262144.
        let resp = context_overflow_response(&state, "deepseek-v4-flash", 1_035_488);
        assert!(resp.is_some(), "oversized request must be rejected");
        assert_eq!(
            resp.unwrap().status(),
            axum::http::StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn context_overflow_allows_in_window_request() {
        let catalog = json!({ "models": [
            {"slug": "deepseek-v4-flash", "context_window": 262144}
        ]});
        let state = test_state(Some(catalog));
        // 150K estimated tokens fits well within the 262144 window.
        assert!(context_overflow_response(&state, "deepseek-v4-flash", 150_000).is_none());
    }

    #[test]
    fn effective_window_uses_catalog_window() {
        // local/dev has no max_context_cap, so the effective window is the
        // catalog window verbatim.
        let catalog = json!({ "models": [
            {"slug": "qwen3.8-27b-1m", "context_window": 1010000}
        ]});
        let state = test_state(Some(catalog));
        assert_eq!(effective_context_window(&state, "qwen3.8-27b-1m"), 1010000);
    }
}
