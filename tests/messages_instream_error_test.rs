// Tests for in-stream upstream error handling on the Anthropic path.
//
// The backend rejects an oversized request with HTTP 200 plus an SSE
// error event rather than a 4xx:
//
//   event: error
//   data: {"message":"upstream error","code":503}
//   data: [DONE]
//
// Those lines carry no `choices`, so the translator skips them. Before the fix,
// finish() then synthesized an empty assistant text block, which Claude Code
// CLI accepted as a turn that simply produced no output — it nudged once, got
// another empty message, and ended the conversation with no error shown.
//
// abort_for_error() closes the stream without that fallback so the caller can
// emit a real Anthropic `error` event instead.

use vllm_proxy::main::translate_anthropic::AnthropicStreamTranslator;
use vllm_proxy::main::translate_response::extract_upstream_error;
use serde_json::json;

/// The exact error payload the backend returns for an oversized request.
const UPSTREAM_ERR: &str = r#"{"message":"upstream error","code":503}"#;

#[test]
fn recognizes_the_backend_oversized_request_error() {
    let parsed: serde_json::Value = serde_json::from_str(UPSTREAM_ERR).unwrap();
    let (code, msg) = extract_upstream_error(&parsed).expect("error should be recognized");
    assert_eq!(code, 503);
    assert_eq!(msg, "upstream error");
}

#[test]
fn error_line_alone_yields_no_content_blocks() {
    // The translator must not invent content from an error line.
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    let events = t.translate_line(&format!("data: {}", UPSTREAM_ERR));
    assert!(
        !events.iter().any(|e| e.contains("content_block")),
        "error line must not produce a content block, got: {:?}",
        events
    );
}

#[test]
fn abort_for_error_emits_no_empty_text_block() {
    // This is the regression that silently ended turns: finish() would
    // synthesize `[{"type":"text","text":""}]` here.
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    t.translate_line(&format!("data: {}", UPSTREAM_ERR));
    let events = t.abort_for_error();
    assert!(
        !events.iter().any(|e| e.contains("content_block_start")),
        "must not synthesize a text block, got: {:?}",
        events
    );
    // The turn did not complete, so it must not look like a normal end_turn.
    assert!(!events.iter().any(|e| e.contains("message_delta")));
    assert!(!events.iter().any(|e| e.contains("message_stop")));
}

#[test]
fn abort_for_error_closes_a_block_left_open_by_partial_output() {
    // Partial text then an error: the open block must be closed so the
    // emitted stream stays well-formed.
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    let started =
        t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"partial answer"}}]}"#);
    assert!(started.iter().any(|e| e.contains("content_block_start")));

    let events = t.abort_for_error();
    assert!(
        events.iter().any(|e| e.contains("content_block_stop")),
        "open text block must be closed, got: {:?}",
        events
    );
    assert!(!events.iter().any(|e| e.contains("content_block_start")));
    assert!(!events.iter().any(|e| e.contains("message_stop")));
}

#[test]
fn abort_for_error_is_idempotent() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}"#);
    let first = t.abort_for_error();
    assert!(first.iter().any(|e| e.contains("content_block_stop")));
    // A second call must not double-close.
    assert!(t.abort_for_error().is_empty());
}

#[test]
fn normal_stream_still_completes_with_content_and_stop() {
    // Control: a healthy stream is unaffected by the error plumbing.
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}"#);
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#);
    let done = t.translate_line("data: [DONE]");
    assert!(done.iter().any(|e| e.contains("\"stop_reason\":\"end_turn\"")));
    assert!(done.iter().any(|e| e.contains("message_stop")));
}

#[test]
fn non_error_json_lines_are_not_mistaken_for_errors() {
    // A normal delta must not trip the error detector, or every healthy
    // response would be reported as a failure.
    let delta = json!({"choices":[{"index":0,"delta":{"content":"some text"}}]});
    assert!(extract_upstream_error(&delta).is_none());

    let usage = json!({"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}});
    assert!(extract_upstream_error(&usage).is_none());
}

// ─── Context-length error parsing ───────────────────────────────────────────

use vllm_proxy::main::translate_response::parse_context_length_error;

#[test]
fn parses_the_real_backend_context_error() {
    let body = "This model's maximum context length is 196608 tokens. However, you \
                requested 32000 output tokens and your prompt contains at least 164609 \
                input tokens, for a total of at least 196609 tokens. Please reduce the \
                length of the input prompt or the number of requested output tokens.";
    assert_eq!(parse_context_length_error(body), Some((196609, 196608)));
}

#[test]
fn falls_back_to_summing_output_and_input_when_no_total_is_given() {
    let body = "This model's maximum context length is 196608 tokens. However, you \
                requested 32000 output tokens and your prompt contains 164609 input tokens.";
    assert_eq!(parse_context_length_error(body), Some((196609, 196608)));
}

#[test]
fn ignores_unrelated_errors() {
    // Must not fire on failures that have nothing to do with sizing, or the
    // client would compact for no reason.
    assert_eq!(parse_context_length_error("CUDA out of memory"), None);
    assert_eq!(parse_context_length_error("upstream error"), None);
    assert_eq!(parse_context_length_error("Too many requests"), None);
    assert_eq!(parse_context_length_error(""), None);
}

#[test]
fn requires_both_numbers_before_claiming_overflow() {
    // Mentions the phrase but carries no usable figures.
    assert_eq!(
        parse_context_length_error("This model's maximum context length is unknown"),
        None
    );
}

#[test]
fn parsing_is_case_insensitive() {
    let body = "THIS MODEL'S MAXIMUM CONTEXT LENGTH IS 196608 TOKENS. HOWEVER, YOU \
                REQUESTED 32000 OUTPUT TOKENS AND YOUR PROMPT CONTAINS AT LEAST 164609 \
                INPUT TOKENS, FOR A TOTAL OF AT LEAST 196609 TOKENS.";
    assert_eq!(parse_context_length_error(body), Some((196609, 196608)));
}
