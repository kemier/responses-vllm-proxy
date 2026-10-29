// Integration tests for /v1/messages when the upstream fails mid-stream.
//
// The backend rejects an oversized request with HTTP 200 and an SSE
// error event rather than a 4xx status:
//
//   event: error
//   data: {"message":"upstream error","code":503}
//   data: [DONE]
//
// Because the status is 200 and the error line carries no `choices`, the
// translator used to skip it and finish() synthesized an empty assistant text
// block. Claude Code CLI treated that as a turn that produced no output, nudged
// once, got another empty message, and ended the conversation with no error
// shown to the user.
//
// These tests drive the real handler against a mock upstream, which is what it
// takes to catch the trailing `data: [DONE]` the backend sends *after* the
// error — a unit test on the translator cannot see that.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use vllm_proxy::main::config::Config;
use vllm_proxy::main::{handle_messages, AppState};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower::util::ServiceExt;

static TEST_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn unique_tmp_dir(prefix: &str) -> std::path::PathBuf {
    let n = TEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "{}-{}-{}-{}",
        prefix,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        n,
    ))
}

/// Serve `upstream_sse` verbatim as an HTTP 200 event stream, then run
/// `request_body` through handle_messages and return the raw SSE the client
/// would receive.
///
/// `upstream_sse` is written exactly as given — no trailing `data: [DONE]` is
/// appended — so a test can reproduce the backend's real framing.
async fn run_messages(upstream_sse: &str, request_body: Value) -> String {
    run_messages_full(upstream_sse, request_body, None).await.1
}

/// Like `run_messages`, but also returns the HTTP status and lets the caller
/// control the upstream status line.
async fn run_messages_full(
    upstream_sse: &str,
    request_body: Value,
    upstream_status: Option<&str>,
) -> (StatusCode, String) {
    let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream_port = upstream_listener.local_addr().unwrap().port();

    let body_to_serve = upstream_sse.to_string();
    let status_line = upstream_status.unwrap_or("200 OK").to_string();
    let _upstream = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let accept = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            upstream_listener.accept(),
        )
        .await;
        let (mut socket, _) = match accept {
            Ok(Ok(conn)) => conn,
            Ok(Err(e)) => panic!("upstream accept failed: {e}"),
            Err(_) => return,
        };
        let mut buf = vec![0u8; 262144];
        let _ = socket.read(&mut buf).await.unwrap();
        let response = format!(
            "HTTP/1.1 {}\r\nContent-Type: text/event-stream\r\n\r\n{}",
            status_line, body_to_serve
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });

    let tmp_dir = unique_tmp_dir("vllm-messages-test");
    let auth_dir = tmp_dir.join("auth");
    std::fs::create_dir_all(&auth_dir).unwrap();
    let expires = chrono::Utc::now() + chrono::Duration::hours(1);
    std::fs::write(
        auth_dir.join("vllm-user.json"),
        json!({
            "access_token": "test-token",
            "refresh_token": "test-refresh",
            "expires_at": expires.to_rfc3339(),
        })
        .to_string(),
    )
    .unwrap();

    let config_yaml = format!(
        r#"
port: 0
host: "127.0.0.1"
auth-dir: "{}"
debug: false
logging-to-file: false
log-level: "error"
openai-compatibility:
  - name: default
    prefix: "default/"
    base-url: "http://127.0.0.1:{}"
    models:
      - name: "zai-org/GLM-5.2-FP8"
        alias: "glm-5.2"
"#,
        auth_dir.display(),
        upstream_port,
    );
    let config_path = tmp_dir.join("config.yaml");
    std::fs::write(&config_path, &config_yaml).unwrap();

    let cfg = Arc::new(Config::load(config_path.to_str().unwrap()).unwrap());
    let state = AppState {
        cfg,
        http_client: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap(),
        model_catalog: std::sync::Arc::new(None),
        shutdown: tokio_util::sync::CancellationToken::new(),
        rate_limiter: Arc::new(vllm_proxy::main::RateLimiter::new(0)),
        model_throttle: Arc::new(vllm_proxy::main::ModelThrottle::new(
            0,
            &std::collections::HashMap::new(),
        )),
        conn_id_cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        vision_caption_cache: vllm_proxy::main::new_vision_caption_cache(),
    };

    use axum::routing::post;
    let app = axum::Router::new()
        .route("/v1/messages", post(handle_messages))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

fn streaming_request() -> Value {
    json!({
        "model": "glm-5.2",
        "max_tokens": 32000,
        "stream": true,
        "messages": [{"role": "user", "content": "summarize this"}]
    })
}

/// The backend's real framing for an oversized request: an error event followed
/// by [DONE], all under HTTP 200.
const UPSTREAM_ERROR_STREAM: &str =
    "event: error\ndata: {\"message\":\"upstream error\",\"code\":503}\n\ndata: [DONE]\n";

#[tokio::test]
async fn instream_error_surfaces_as_an_error_event() {
    let sse = run_messages(UPSTREAM_ERROR_STREAM, streaming_request()).await;
    assert!(
        sse.contains("event: error"),
        "client must receive an error event, got:\n{sse}"
    );
    assert!(sse.contains("503"), "error should carry the upstream code:\n{sse}");
    assert!(
        sse.contains("api_error"),
        "error should be typed for the client:\n{sse}"
    );
}

#[tokio::test]
async fn instream_error_does_not_synthesize_an_empty_assistant_message() {
    // The regression that silently ended turns. The trailing `data: [DONE]`
    // must not reach the translator, or finish() emits an empty text block and
    // a normal-looking end_turn.
    let sse = run_messages(UPSTREAM_ERROR_STREAM, streaming_request()).await;
    assert!(
        !sse.contains("content_block_start"),
        "must not synthesize a content block:\n{sse}"
    );
    assert!(
        !sse.contains("\"stop_reason\":\"end_turn\""),
        "a failed turn must not look like a clean end_turn:\n{sse}"
    );
    assert!(
        !sse.contains("message_stop"),
        "must not report a completed message:\n{sse}"
    );
}

#[tokio::test]
async fn partial_output_then_error_closes_the_block_and_reports() {
    // Text streamed before the failure must be preserved and its block closed,
    // with the error still reported.
    let sse = run_messages(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial answer\"}}]}\n\
         \nevent: error\ndata: {\"message\":\"upstream error\",\"code\":503}\n\ndata: [DONE]\n",
        streaming_request(),
    )
    .await;
    assert!(sse.contains("partial answer"), "partial text must survive:\n{sse}");
    assert!(sse.contains("content_block_stop"), "open block must close:\n{sse}");
    assert!(sse.contains("event: error"), "error must still surface:\n{sse}");
    assert!(
        !sse.contains("message_stop"),
        "an aborted turn must not report completion:\n{sse}"
    );
}

#[tokio::test]
async fn healthy_stream_is_unaffected() {
    // Control: no error plumbing side effects on a normal response.
    let sse = run_messages(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\
         \ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
         \ndata: [DONE]\n",
        streaming_request(),
    )
    .await;
    assert!(!sse.contains("event: error"), "no error expected:\n{sse}");
    assert!(sse.contains("hello"), "content must pass through:\n{sse}");
    assert!(sse.contains("\"stop_reason\":\"end_turn\""));
    assert!(sse.contains("message_stop"));
}

#[tokio::test]
async fn non_streaming_error_object_in_200_response_is_reported() {
    // Same trap on the non-streaming branch: an error object instead of choices.
    let (status, body) = run_messages_full(
        "{\"message\":\"upstream error\",\"code\":503}",
        json!({
            "model": "glm-5.2",
            "max_tokens": 32000,
            "stream": false,
            "messages": [{"role": "user", "content": "summarize this"}]
        }),
        None,
    )
    .await;
    // Must NOT be 200. A 200 with an error body makes the Anthropic SDK report
    // "API returned an empty or malformed response (HTTP 200)".
    assert_ne!(status, StatusCode::OK, "got: {body}");
    assert_eq!(status.as_u16(), 503, "should preserve the upstream code: {body}");
    let parsed: Value = serde_json::from_str(body.trim()).expect("should be JSON");
    assert_eq!(parsed["type"], "error", "got: {body}");
    assert_eq!(parsed["error"]["type"], "api_error");
    assert!(
        parsed["error"]["message"].as_str().unwrap().contains("503"),
        "got: {body}"
    );
    // Must not be translated into an assistant message with empty content.
    assert!(parsed.get("content").is_none(), "got: {body}");
}

/// The backend's verbatim wording for an over-budget request, captured from a
/// real failure: 164609 input + 32000 output = 196609 against a 196608 window.
const VLLM_CONTEXT_ERROR: &str = "{\"object\":\"error\",\"message\":\"This model's maximum context length is 196608 tokens. However, you requested 32000 output tokens and your prompt contains at least 164609 input tokens, for a total of at least 196609 tokens. Please reduce the length of the input prompt or the number of requested output tokens.\",\"type\":\"BadRequestError\",\"code\":400}";

#[tokio::test]
async fn context_overflow_becomes_a_recoverable_prompt_too_long_error() {
    // Claude Code CLI compacts and retries when it recognizes Anthropic's
    // wording, which it matches with
    // /prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)/i. Passing the
    // backend's own phrasing through would leave the session dead instead.
    let (status, body) = run_messages_full(
        VLLM_CONTEXT_ERROR,
        streaming_request(),
        Some("500 Internal Server Error"),
    )
    .await;

    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "an over-long prompt is a 400 regardless of the backend's status: {body}"
    );
    let parsed: Value = serde_json::from_str(body.trim()).expect("should be JSON");
    assert_eq!(parsed["error"]["type"], "invalid_request_error", "got: {body}");

    let msg = parsed["error"]["message"].as_str().unwrap();
    assert_eq!(msg, "prompt is too long: 196609 tokens > 196608 maximum");

    // Verify against Claude Code's actual extraction regex semantics: the two
    // numbers it needs must be present and in the right order.
    let lower = msg.to_lowercase();
    assert!(lower.starts_with("prompt is too long"));
    let nums: Vec<u64> = lower
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(nums, vec![196609, 196608], "actual then limit: {msg}");
    assert!(nums[0] > nums[1], "must report actual exceeding the limit");
}

#[tokio::test]
async fn non_context_upstream_failure_keeps_its_own_status_and_type() {
    // Control: an unrelated 500 must not be mislabeled as prompt-too-long,
    // or the client would compact for no reason.
    let (status, body) = run_messages_full(
        "{\"object\":\"error\",\"message\":\"CUDA out of memory\",\"code\":500}",
        streaming_request(),
        Some("500 Internal Server Error"),
    )
    .await;
    assert_eq!(status.as_u16(), 500, "got: {body}");
    let parsed: Value = serde_json::from_str(body.trim()).expect("should be JSON");
    assert_eq!(parsed["error"]["type"], "api_error");
    assert!(
        !parsed["error"]["message"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("prompt is too long"),
        "must not claim prompt-too-long: {body}"
    );
}

#[tokio::test]
async fn rate_limit_is_typed_for_the_client() {
    let (status, body) = run_messages_full(
        "{\"message\":\"Too many requests\",\"code\":429}",
        streaming_request(),
        Some("429 Too Many Requests"),
    )
    .await;
    assert_eq!(status.as_u16(), 429, "got: {body}");
    let parsed: Value = serde_json::from_str(body.trim()).expect("should be JSON");
    assert_eq!(parsed["error"]["type"], "rate_limit_error", "got: {body}");
}
