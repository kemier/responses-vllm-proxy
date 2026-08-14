// Integration tests for the /v1/responses/compact endpoint.
//
// These tests spin up a mock upstream HTTP server that returns canned SSE
// chunks, then build the proxy's axum Router and call handle_compact
// directly via tower::ServiceExt::oneshot.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use vllm_proxy::main::config::Config;
use vllm_proxy::main::{handle_compact, AppState};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower::util::ServiceExt;

// ─── Helpers ────────────────────────────────────────────────────────────────

fn sse_delta(content: &str) -> String {
    format!(
        r#"data: {{"choices":[{{"delta":{{"content":"{}"}},"finish_reason":null}}]}}"#,
        content
    )
}

fn sse_stop() -> String {
    r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#.to_string()
}

/// Monotonic counter for unique temp-dir names. Avoids collisions when
/// multiple tests start in the same millisecond (parallel test execution).
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

async fn run_compact_test(upstream_sse: Vec<String>, request_body: Value) -> (StatusCode, Value) {
    let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream_port = upstream_listener.local_addr().unwrap().port();

    // Keep ownership of the listener outside the spawned task so it stays
    // alive until the proxy connects (or the timeout fires). The task
    // accepts one connection, writes the canned SSE response, and exits.
    let upstream = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // Timeout: if the proxy never connects (e.g. reqwest fails before
        // reaching upstream), don't block the test forever.
        let accept_result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            upstream_listener.accept(),
        )
        .await;
        let (mut socket, _) = match accept_result {
            Ok(Ok(conn)) => conn,
            Ok(Err(e)) => panic!("upstream accept failed: {e}"),
            Err(_) => return, // timed out — proxy never connected
        };
        let mut buf = vec![0u8; 65536];
        let _ = socket.read(&mut buf).await.unwrap();

        let mut response =
            String::from("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
        for chunk in &upstream_sse {
            response.push_str(chunk);
            response.push('\n');
        }
        response.push_str("data: [DONE]\n");
        socket.write_all(response.as_bytes()).await.unwrap();
    });

    let tmp_dir = unique_tmp_dir("vllm-compact-test");
    let auth_dir = tmp_dir.join("auth");
    std::fs::create_dir_all(&auth_dir).unwrap();

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
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();

    let state = AppState {
        cfg: cfg.clone(),
        http_client,
        model_catalog: std::sync::Arc::new(None),
        shutdown: tokio_util::sync::CancellationToken::new(),
        rate_limiter: Arc::new(vllm_proxy::main::RateLimiter::new(0)),
        model_throttle: Arc::new(vllm_proxy::main::ModelThrottle::new(
            0,
            &std::collections::HashMap::new(),
        )),
        conn_id_cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
    };

    use axum::routing::post;
    let app = axum::Router::new()
        .route("/v1/responses/compact", post(handle_compact))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses/compact")
                .header("content-type", "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body_bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body_json: Value = serde_json::from_slice(&body_bytes)
        .unwrap_or_else(|_| json!({"_raw": String::from_utf8_lossy(&body_bytes).to_string()}));

    // Don't block forever if the mock upstream task is still waiting for
    // a connection that will never arrive. A short timeout is enough —
    // the proxy either connected (task finished) or failed (still waiting).
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), upstream).await;
    let _ = std::fs::remove_dir_all(&tmp_dir);

    (status, body_json)
}

async fn build_app() -> (axum::Router, std::path::PathBuf) {
    let tmp_dir = unique_tmp_dir("vllm-compact-test-noup");
    let auth_dir = tmp_dir.join("auth");
    std::fs::create_dir_all(&auth_dir).unwrap();
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
    base-url: "http://127.0.0.1:9999"
    models:
      - name: "zai-org/GLM-5.2-FP8"
        alias: "glm-5.2"
"#,
        auth_dir.display(),
    );
    let config_path = tmp_dir.join("config.yaml");
    std::fs::write(&config_path, &config_yaml).unwrap();

    let cfg = Arc::new(Config::load(config_path.to_str().unwrap()).unwrap());
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();

    let state = AppState {
        cfg,
        http_client,
        model_catalog: std::sync::Arc::new(None),
        shutdown: tokio_util::sync::CancellationToken::new(),
        rate_limiter: Arc::new(vllm_proxy::main::RateLimiter::new(0)),
        model_throttle: Arc::new(vllm_proxy::main::ModelThrottle::new(
            0,
            &std::collections::HashMap::new(),
        )),
        conn_id_cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
    };

    use axum::routing::post;
    let app = axum::Router::new()
        .route("/v1/responses/compact", post(handle_compact))
        .with_state(state);

    (app, tmp_dir)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_compact_basic_streaming_response() {
    let sse_chunks = vec![
        sse_delta("Summary of "),
        sse_delta("the conversation."),
        sse_stop(),
    ];

    let request = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Hi there!"},
        ]
    });

    let (status, body) = run_compact_test(sse_chunks, request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "response.compaction");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["model"], "glm-5.2");

    let output = body["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "compaction");
    let summary = output[0]["summary_text"].as_str().unwrap();
    assert!(summary.contains("Summary of"));
    assert!(summary.contains("the conversation."));
    assert_eq!(output[0]["compaction_trigger"], "auto");

    assert!(body["usage"]["input_tokens"].is_u64());
    assert!(body["usage"]["output_tokens"].is_u64());
}

#[tokio::test]
async fn test_compact_empty_stream_returns_default_summary() {
    let sse_chunks = vec![sse_stop()];

    let request = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": "test"},
        ]
    });

    let (status, body) = run_compact_test(sse_chunks, request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "response.compaction");
    let summary = body["output"][0]["summary_text"].as_str().unwrap();
    assert!(!summary.is_empty());
}

#[tokio::test]
async fn test_compact_multi_chunk_delta_assembly() {
    let sse_chunks = vec![
        sse_delta("Part1 "),
        sse_delta("Part2 "),
        sse_delta("Part3"),
        sse_stop(),
    ];

    let request = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": "summarize"},
        ]
    });

    let (status, body) = run_compact_test(sse_chunks, request).await;

    assert_eq!(status, StatusCode::OK);
    let summary = body["output"][0]["summary_text"].as_str().unwrap();
    assert_eq!(summary, "Part1 Part2 Part3");
}

#[tokio::test]
async fn test_compact_response_has_id() {
    let sse_chunks = vec![sse_delta("ok"), sse_stop()];

    let request = json!({
        "model": "glm-5.2",
        "input": "test"
    });

    let (status, body) = run_compact_test(sse_chunks, request).await;

    assert_eq!(status, StatusCode::OK);
    let id = body["id"].as_str().unwrap();
    assert!(id.starts_with("resp_compact_"));
}

#[tokio::test]
async fn test_compact_invalid_json_returns_400() {
    let (app, tmp_dir) = build_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses/compact")
                .header("content-type", "application/json")
                .body(Body::from("{invalid json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[tokio::test]
async fn test_compact_stream_field_removed_from_request() {
    let sse_chunks = vec![sse_delta("ok"), sse_stop()];

    let request = json!({
        "model": "glm-5.2",
        "input": "test",
        "stream": false
    });

    let (status, body) = run_compact_test(sse_chunks, request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "response.compaction");
}

#[tokio::test]
async fn test_compact_usage_fields_present() {
    let sse_chunks = vec![sse_delta("summary"), sse_stop()];

    let request = json!({
        "model": "glm-5.2",
        "input": "test"
    });

    let (status, body) = run_compact_test(sse_chunks, request).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body["usage"].is_object());
    assert!(body["usage"]["input_tokens"].is_u64());
    assert!(body["usage"]["output_tokens"].is_u64());
    assert!(body["usage"]["total_tokens"].is_u64());
}
