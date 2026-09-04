// Unit tests for src/translate_anthropic.rs
//
// Covers:
//   - anthropic_to_chat_completions (request translation)
//   - AnthropicStreamTranslator (streaming SSE translation)
//   - chat_completions_to_anthropic (non-streaming response translation)

use vllm_proxy::main::translate_anthropic::{
    anthropic_to_chat_completions, anthropic_to_chat_completions_with_vision,
    chat_completions_to_anthropic, AnthropicStreamTranslator,
};
use serde_json::{json, Value};

// ─── anthropic_to_chat_completions (request) ───────────────────────────────

#[test]
fn test_anthropic_simple_user_message() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    });
    let out = anthropic_to_chat_completions("glm-5.2", &body, false);
    assert_eq!(out["model"], "glm-5.2");
    assert_eq!(out["stream"], false);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "Hello");
    assert_eq!(out["max_tokens"], 1024);
    // No stream_options when stream=false
    assert!(out.get("stream_options").is_none());
}

#[test]
fn test_anthropic_stream_adds_stream_options() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = anthropic_to_chat_completions("glm-5.2", &body, true);
    assert_eq!(out["stream"], true);
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn test_anthropic_system_string_becomes_system_message() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "You are helpful.");
    assert_eq!(messages[1]["role"], "user");
}

#[test]
fn test_anthropic_system_array_becomes_system_message() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "system": [{"type": "text", "text": "Rule 1"}, {"type": "text", "text": "Rule 2"}],
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "Rule 1\nRule 2");
}

#[test]
fn test_anthropic_mid_array_system_message_merged_into_leading_system() {
    // Claude Code CLI sometimes emits a `system`-role message inside `messages`
    // (e.g. the "Available agent types for the Agent tool" block injected
    // mid-conversation). vLLM rejects any request where the first message is
    // not system ("System message must be at the beginning."), so stray system
    // messages must be merged into the leading system message, never emitted
    // in place after a user message.
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hi there"},
            {"role": "system", "content": "Available agent types for the Agent tool:\n- claude: Catch-all"},
            {"role": "user", "content": "Use the agent tool now"}
        ]
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    let messages = out["messages"].as_array().unwrap();
    // No non-first system message may survive.
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, vec!["system", "user", "user"]);
    // The leading system message absorbed the mid-array system text.
    assert!(messages[0]["content"]
        .as_str()
        .unwrap()
        .contains("Available agent types"));
    assert!(messages[0]["content"]
        .as_str()
        .unwrap()
        .starts_with("You are helpful."));
    assert_eq!(messages[1]["content"], "Hi there");
    assert_eq!(messages[2]["content"], "Use the agent tool now");
}

#[test]
fn test_anthropic_system_message_inside_messages_hoisted_when_no_top_level_system() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": "You are the system."},
            {"role": "user", "content": "Hello"}
        ]
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    let messages = out["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, vec!["system", "user"]);
    assert_eq!(messages[0]["content"], "You are the system.");
    assert_eq!(messages[1]["content"], "Hello");
}

#[test]
fn test_anthropic_assistant_with_tool_use() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Stockholm"}}
            ]}
        ]
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"], "Let me check.");
    let tool_calls = messages[1]["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls[0]["id"], "toolu_1");
    assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
    let args: Value =
        serde_json::from_str(tool_calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Stockholm");
}

#[test]
fn test_anthropic_user_with_tool_result() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "STO"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny 22C"}
            ]}
        ]
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    let messages = out["messages"].as_array().unwrap();
    // user, assistant(tool_calls), tool
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "toolu_1");
    assert_eq!(messages[2]["content"], "Sunny 22C");
}

#[test]
fn test_anthropic_tools_converted_to_openai_function() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
        }]
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    let tools = out["tools"].as_array().unwrap();
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "get_weather");
    assert_eq!(
        tools[0]["function"]["parameters"]["properties"]["city"]["type"],
        "string"
    );
}

#[test]
fn test_anthropic_tool_choice_auto() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "auto"
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    assert_eq!(out["tool_choice"], "auto");
}

#[test]
fn test_anthropic_tool_choice_any_becomes_required() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "any"
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    assert_eq!(out["tool_choice"], "required");
}

#[test]
fn test_anthropic_tool_choice_specific_tool() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "tool", "name": "get_weather"}
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    assert_eq!(out["tool_choice"]["type"], "function");
    assert_eq!(out["tool_choice"]["function"]["name"], "get_weather");
}

#[test]
fn test_anthropic_stop_sequences_become_stop() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "stop_sequences": ["END", "STOP"]
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    let stop = out["stop"].as_array().unwrap();
    assert_eq!(stop.len(), 2);
    assert!(stop.iter().any(|s| s == "END"));
    assert!(stop.iter().any(|s| s == "STOP"));
}

#[test]
fn test_anthropic_temperature_top_p_passed_through() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["top_p"], 0.9);
}

#[test]
fn test_anthropic_empty_messages() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": []
    });
    let out = anthropic_to_chat_completions("glm", &body, false);
    assert_eq!(out["messages"].as_array().unwrap().len(), 0);
}

// ─── AnthropicStreamTranslator (streaming) ─────────────────────────────────

#[test]
fn test_anthropic_stream_simple_text() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    let e1 = t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#);
    // Should emit message_start, content_block_start, content_block_delta
    assert!(e1.iter().any(|e| e.contains("message_start")));
    assert!(e1
        .iter()
        .any(|e| e.contains("content_block_start") && e.contains("\"type\":\"text\"")));
    assert!(e1
        .iter()
        .any(|e| e.contains("content_block_delta") && e.contains("Hello")));

    let done = t.translate_line("data: [DONE]");
    assert!(done.iter().any(|e| e.contains("content_block_stop")));
    assert!(done.iter().any(|e| e.contains("message_delta")));
    assert!(done.iter().any(|e| e.contains("message_stop")));
}

#[test]
fn test_anthropic_stream_reasoning_then_text() {
    let mut t = AnthropicStreamTranslator::new("deepseek-v4");
    let e1 =
        t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"reasoning":"Thinking..."}}]}"#);
    assert!(e1
        .iter()
        .any(|e| e.contains("content_block_start") && e.contains("\"type\":\"thinking\"")));
    assert!(e1
        .iter()
        .any(|e| e.contains("thinking_delta") && e.contains("Thinking...")));

    let e2 = t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Answer"}}]}"#);
    // Should close thinking block, open text block
    assert!(e2.iter().any(|e| e.contains("content_block_stop")));
    assert!(e2
        .iter()
        .any(|e| e.contains("content_block_start") && e.contains("\"type\":\"text\"")));
    assert!(e2
        .iter()
        .any(|e| e.contains("text_delta") && e.contains("Answer")));
}

#[test]
fn test_anthropic_stream_tool_use() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    // Tool call with id and name
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]}}]}"#);
    // Argument fragment
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"STO\"}"}}]}}]}"#);
    // finish_reason triggers tool_use block emission
    let finish = t.translate_line(
        r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
    );
    assert!(finish
        .iter()
        .any(|e| e.contains("content_block_start") && e.contains("\"type\":\"tool_use\"")));
    assert!(finish.iter().any(|e| e.contains("input_json_delta")));
    // message_delta (which carries stop_reason) is deliberately deferred to
    // [DONE]: vLLM sends the usage chunk *after* finish_reason, so emitting it
    // here would always report 0 tokens.
    assert!(!finish.iter().any(|e| e.contains("message_delta")));

    let done = t.translate_line("data: [DONE]");
    assert!(done
        .iter()
        .any(|e| e.contains("\"stop_reason\":\"tool_use\"")));
    assert!(done.iter().any(|e| e.contains("message_stop")));
    // The tool_use block must not be duplicated by the empty-response fallback.
    assert!(!done.iter().any(|e| e.contains("content_block_start")));
}

#[test]
fn test_anthropic_stream_finish_reason_stop() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#);
    let finish =
        t.translate_line(r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#);
    assert!(finish.iter().any(|e| e.contains("content_block_stop")));
    let done = t.translate_line("data: [DONE]");
    assert!(done
        .iter()
        .any(|e| e.contains("\"stop_reason\":\"end_turn\"")));
}

#[test]
fn test_anthropic_stream_finish_reason_length() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"..."}}]}"#);
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#);
    let done = t.translate_line("data: [DONE]");
    assert!(done
        .iter()
        .any(|e| e.contains("\"stop_reason\":\"max_tokens\"")));
}

#[test]
fn test_anthropic_stream_usage_extracted() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#);
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":42,"completion_tokens":7}}"#);
    let done = t.translate_line("data: [DONE]");
    let msg_delta = done.iter().find(|e| e.contains("message_delta")).unwrap();
    assert!(msg_delta.contains("\"input_tokens\":42"));
    assert!(msg_delta.contains("\"output_tokens\":7"));
}

#[test]
fn test_anthropic_stream_ignores_non_data_lines() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    assert_eq!(t.translate_line("").len(), 0);
    assert_eq!(t.translate_line("   ").len(), 0);
    assert_eq!(t.translate_line(": comment").len(), 0);
}

#[test]
fn test_anthropic_stream_ignores_invalid_json() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    assert_eq!(t.translate_line("data: not json").len(), 0);
    assert_eq!(t.translate_line("data: {broken").len(), 0);
}

#[test]
fn test_anthropic_stream_empty_content_ignored() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    // Empty content delta should not open a text block
    let events = t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":""}}]}"#);
    // Only message_start (if first chunk), no content_block_start for text
    assert!(!events.iter().any(|e| e.contains("\"type\":\"text\"")));
}

#[test]
fn test_anthropic_stream_done_without_finish_defaults_end_turn() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#);
    let done = t.translate_line("data: [DONE]");
    assert!(done
        .iter()
        .any(|e| e.contains("\"stop_reason\":\"end_turn\"")));
}

#[test]
fn test_anthropic_stream_double_done_idempotent() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#);
    let first = t.translate_line("data: [DONE]");
    assert!(!first.is_empty());
    // Second [DONE] should produce no events (already finished)
    let second = t.translate_line("data: [DONE]");
    assert!(second.is_empty());
}

// ─── chat_completions_to_anthropic (non-streaming) ─────────────────────────

#[test]
fn test_non_streaming_anthropic_simple_text() {
    let raw = json!({
        "id": "chatcmpl_1",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    });
    let resp = chat_completions_to_anthropic("glm-5.2", &raw);
    assert_eq!(resp["type"], "message");
    assert_eq!(resp["role"], "assistant");
    assert_eq!(resp["stop_reason"], "end_turn");
    let content = resp["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "Hello!");
    assert_eq!(resp["usage"]["input_tokens"], 5);
    assert_eq!(resp["usage"]["output_tokens"], 3);
}

#[test]
fn test_non_streaming_anthropic_with_reasoning() {
    let raw = json!({
        "id": "chatcmpl_2",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer.",
                "reasoning": "Let me think..."
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    });
    let resp = chat_completions_to_anthropic("deepseek-v4", &raw);
    let content = resp["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "Let me think...");
    assert_eq!(content[1]["type"], "text");
    assert_eq!(content[1]["text"], "The answer.");
}

#[test]
fn test_non_streaming_anthropic_with_tool_calls() {
    let raw = json!({
        "id": "chatcmpl_3",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"STO\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });
    let resp = chat_completions_to_anthropic("glm-5.2", &raw);
    assert_eq!(resp["stop_reason"], "tool_use");
    let content = resp["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "tool_use");
    assert_eq!(content[0]["id"], "call_1");
    assert_eq!(content[0]["name"], "get_weather");
    assert_eq!(content[0]["input"]["city"], "STO");
}

#[test]
fn test_non_streaming_anthropic_empty_response() {
    let raw = json!({
        "id": "chatcmpl_empty",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop"
        }]
    });
    let resp = chat_completions_to_anthropic("glm-5.2", &raw);
    let content = resp["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "");
    assert_eq!(resp["usage"]["input_tokens"], 0);
    assert_eq!(resp["usage"]["output_tokens"], 0);
}

#[test]
fn test_non_streaming_anthropic_no_choices() {
    let raw = json!({"id": "x", "choices": []});
    let resp = chat_completions_to_anthropic("glm-5.2", &raw);
    assert_eq!(resp["type"], "message");
    assert_eq!(resp["stop_reason"], "end_turn");
    let content = resp["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "");
}

#[test]
fn test_non_streaming_anthropic_finish_reason_length() {
    let raw = json!({
        "id": "chatcmpl_len",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "truncated"},
            "finish_reason": "length"
        }]
    });
    let resp = chat_completions_to_anthropic("glm-5.2", &raw);
    assert_eq!(resp["stop_reason"], "max_tokens");
}

// ─── Reasoning-only fallback ───────────────────────────────────────────────
// GLM-5.2 with thinking enabled may emit the entire response via
// delta.reasoning / message.reasoning with empty content. Without a text
// content block, Claude Code CLI's /compact fails with "no assistant message
// in summarization response". These tests verify the proxy promotes
// reasoning-only output to a text block in both the streaming and
// non-streaming Anthropic paths.

#[test]
fn test_stream_reasoning_only_synthesizes_text_block() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    // Reasoning-only stream: content never arrives.
    t.translate_line(
        r#"data: {"choices":[{"index":0,"delta":{"reasoning":"thinking step 1. "}}]}"#,
    );
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"reasoning":"thinking step 2."}}]}"#);
    t.translate_line(r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#);
    let done = t.translate_line("data: [DONE]");

    // A text content block must have been synthesized from the reasoning.
    let joined = done.join("\n");
    assert!(
        joined.contains("\"type\":\"text_delta\""),
        "expected a synthesized text_delta, got: {}",
        joined
    );
    // The synthesized text must contain BOTH reasoning fragments (accumulated).
    assert!(
        joined.contains("thinking step 1. "),
        "missing first reasoning fragment"
    );
    assert!(
        joined.contains("thinking step 2."),
        "missing second reasoning fragment"
    );
    // And the text block must be properly closed.
    assert!(joined.contains("content_block_stop"));
    assert!(joined.contains("message_stop"));
}

#[test]
fn test_stream_reasoning_plus_content_does_not_duplicate() {
    let mut t = AnthropicStreamTranslator::new("glm-5.2");
    // Both reasoning and real content arrive — text block comes from content,
    // reasoning must NOT be synthesized again at finish().
    let mut all = Vec::new();
    all.extend(t.translate_line(
        r#"data: {"choices":[{"index":0,"delta":{"reasoning":"private thoughts"}}]}"#,
    ));
    all.extend(
        t.translate_line(r#"data: {"choices":[{"index":0,"delta":{"content":"final answer"}}]}"#),
    );
    all.extend(
        t.translate_line(r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
    );
    all.extend(t.translate_line("data: [DONE]"));
    let joined = all.join("\n");

    // The real content is present (emitted as a text_delta during streaming).
    assert!(joined.contains("final answer"));
    // The reasoning must NOT have been promoted to a text block at finish
    // (no synthesized text_delta containing the reasoning text).
    assert!(
        !joined.contains("\"text\":\"private thoughts\""),
        "reasoning was incorrectly synthesized as text when content was present"
    );
}

#[test]
fn test_non_streaming_reasoning_only_promotes_to_text() {
    let raw = json!({
        "id": "chatcmpl_ro",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "", "reasoning": "the real summary"},
            "finish_reason": "stop"
        }]
    });
    let resp = chat_completions_to_anthropic("glm-5.2", &raw);
    let content = resp["content"].as_array().unwrap();
    // A thinking block AND a text block (promoted from reasoning).
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "the real summary");
    assert_eq!(content[1]["type"], "text");
    assert_eq!(content[1]["text"], "the real summary");
}

#[test]
fn test_non_streaming_content_present_not_promoted() {
    let raw = json!({
        "id": "chatcmpl_cp",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "real answer", "reasoning": "thoughts"},
            "finish_reason": "stop"
        }]
    });
    let resp = chat_completions_to_anthropic("glm-5.2", &raw);
    let content = resp["content"].as_array().unwrap();
    // thinking + text (from content), no promoted duplicate.
    assert_eq!(content.len(), 2);
    assert_eq!(content[1]["type"], "text");
    assert_eq!(content[1]["text"], "real answer");
}

// ─── Image block forwarding (user message) ────────────────────────────────

#[test]
fn test_anthropic_user_image_block_becomes_image_url_content() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aGVsbG8="
                }},
                {"type": "text", "text": "What is in this image?"}
            ]
        }]
    });
    let out = anthropic_to_chat_completions("qwen3.8-27b", &body, false);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    let content = messages[0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    // text first, then image_url
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "What is in this image?");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(
        content[1]["image_url"]["url"],
        "data:image/png;base64,aGVsbG8="
    );
    assert_eq!(messages[0]["role"], "user");
}

#[test]
fn test_anthropic_user_image_only_no_empty_message() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "Zm9v"
                }}
            ]
        }]
    });
    let out = anthropic_to_chat_completions("qwen3.8-27b", &body, false);
    let messages = out["messages"].as_array().unwrap();
    // Exactly one message: the image content array. No spurious empty string.
    assert_eq!(messages.len(), 1);
    let content = messages[0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "image_url");
    assert_eq!(
        content[0]["image_url"]["url"],
        "data:image/jpeg;base64,Zm9v"
    );
}

#[test]
fn test_anthropic_user_non_base64_image_skipped() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://x.com/i.png"}},
                {"type": "text", "text": "hi"}
            ]
        }]
    });
    let out = anthropic_to_chat_completions("qwen3.8-27b", &body, false);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    // Non-base64 image source is dropped; plain string content preserved.
    assert_eq!(messages[0]["content"], "hi");
}

#[test]
fn test_anthropic_image_stripped_for_non_vision_model() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aGVsbG8="
                }},
                {"type": "text", "text": "Describe the image"}
            ]
        }]
    });
    // glm-5.2 is text-only — image must be stripped, text preserved as plain string.
    let out = anthropic_to_chat_completions_with_vision("zai-org/GLM-5.2-FP8", &body, false, false);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "Describe the image");
}

#[test]
fn test_anthropic_image_only_stripped_for_non_vision_model() {
    let body = json!({
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "Zm9v"
                }}
            ]
        }]
    });
    // Image-only message on a text-only model → keep a plain empty-ish message
    // so the request stays valid (the client still gets a turn).
    let out = anthropic_to_chat_completions_with_vision("zai-org/GLM-5.2-FP8", &body, false, false);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["content"], "");
}

// ─── effort → reasoning_effort mapping ─────────────────────────────────────

#[test]
fn test_effort_output_config_mapping() {
    let cases: [(&str, Option<&str>); 8] = [
        ("minimal", Some("none")),
        ("low", Some("low")),
        ("medium", Some("medium")),
        ("high", Some("xhigh")),
        ("xhigh", Some("xhigh")),
        ("max", Some("xhigh")),
        ("off", Some("none")),
        ("bogus", None),
    ];
    for (effort, expected) in cases {
        let body = json!({
            "model": "qwen38-27b",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": effort},
        });
        let out = anthropic_to_chat_completions("qwen38-27b", &body, false);
        match expected {
            Some(e) => assert_eq!(out.get("reasoning_effort").and_then(|v| v.as_str()), Some(e), "effort={effort}"),
            None => assert!(out.get("reasoning_effort").is_none(), "effort={effort}"),
        }
    }
}

#[test]
fn test_effort_thinking_budget_mapping() {
    let cases: [(i64, &str); 5] = [
        (0, "none"),
        (1024, "low"),
        (2048, "low"),
        (8192, "medium"),
        (16384, "xhigh"),
    ];
    for (budget, expected) in cases {
        let body = json!({
            "model": "qwen38-27b",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": budget},
        });
        let out = anthropic_to_chat_completions("qwen38-27b", &body, false);
        assert_eq!(
            out.get("reasoning_effort").and_then(|v| v.as_str()),
            Some(expected),
            "budget={budget}"
        );
    }
    let body = json!({
        "model": "qwen38-27b",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
    });
    let out = anthropic_to_chat_completions("qwen38-27b", &body, false);
    assert_eq!(out.get("reasoning_effort").and_then(|v| v.as_str()), Some("none"));
}

#[test]
fn test_effort_absent_keeps_template_default() {
    let body = json!({
        "model": "qwen38-27b",
        "messages": [{"role": "user", "content": "hi"}],
    });
    let out = anthropic_to_chat_completions("qwen38-27b", &body, false);
    assert!(out.get("reasoning_effort").is_none());
}
