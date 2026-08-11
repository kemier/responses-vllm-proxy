use ericai_proxy::translate_request::inject_thinking;
use ericai_proxy::translate_request::responses_to_chat_completions;
use serde_json::{json, Value};

#[test]
fn test_simple_text_input() {
    let req = json!({
        "model": "glm-5.2",
        "input": "Hello, world!"
    });
    let out = responses_to_chat_completions("zai-org/GLM-5.2-FP8", &req, true);
    assert_eq!(out["model"], "zai-org/GLM-5.2-FP8");
    assert_eq!(out["stream"], true);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "Hello, world!");
}

#[test]
fn test_instructions_become_system_message() {
    let req = json!({
        "model": "glm-5.2",
        "instructions": "You are a helpful assistant.",
        "input": "Hi"
    });
    let out = responses_to_chat_completions("glm", &req, false);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "You are a helpful assistant.");
    assert_eq!(messages[1]["role"], "user");
}

#[test]
fn test_input_array_with_message() {
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Hello"}
            ]}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "Hello");
}

#[test]
fn test_string_content_in_message() {
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": "Simple text"}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages[0]["content"], "Simple text");
}

#[test]
fn test_multi_content_parts_joined() {
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Part 1"},
                {"type": "input_text", "text": "Part 2"}
            ]}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages[0]["content"], "Part 1\nPart 2");
}

#[test]
fn test_function_call_and_output() {
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": "What's the weather?"},
            {"type": "function_call", "call_id": "call_123", "name": "get_weather", "arguments": "{\"city\":\"Stockholm\"}"},
            {"type": "function_call_output", "call_id": "call_123", "output": "Sunny, 22C"}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    // user + assistant(tool_calls) + tool
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    assert!(messages[1]["tool_calls"].is_array());
    assert_eq!(
        messages[1]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call_123");
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "call_123");
    assert_eq!(messages[2]["content"], "Sunny, 22C");
}

#[test]
fn test_function_call_empty_arguments_normalized_to_empty_json_object() {
    // A cached `function_call` history item (e.g. from a tool called with no
    // parameters, or an older cache entry predating the empty-args fix) may
    // have an empty or missing `arguments` field. Replaying "" as the
    // assistant tool_calls arguments is not valid JSON and vLLM rejects the
    // whole request with "Assistant tool call function.arguments must be
    // valid JSON." The proxy must default to "{}" on replay.
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": "wait for it"},
            {"type": "function_call", "call_id": "call_noargs", "name": "wait", "arguments": ""},
            {"type": "function_call_output", "call_id": "call_noargs", "output": "done"}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(
        messages[1]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );

    // Missing `arguments` field entirely should also default to "{}".
    let req2 = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": "wait for it"},
            {"type": "function_call", "call_id": "call_missing", "name": "wait"},
            {"type": "function_call_output", "call_id": "call_missing", "output": "done"}
        ]
    });
    let out2 = responses_to_chat_completions("glm", &req2, true);
    let messages2 = out2["messages"].as_array().unwrap();
    assert_eq!(
        messages2[1]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );
}

#[test]
fn test_function_call_truncated_json_arguments_repaired() {
    // Root-cause case for the recurring upstream HTTP 400
    // ("Assistant tool call function.arguments must be valid JSON."):
    // the model hit max_tokens mid-tool-call, so the accumulated arguments
    // are a TRUNCATED JSON object. That item lives in the conversation
    // history, so without repair every subsequent turn resends it and fails
    // identically — the session stays wedged even across proxy restarts.
    let truncated = r#"{"command":["bash","-lc","cat fi"#;
    assert!(
        serde_json::from_str::<serde_json::Value>(truncated).is_err(),
        "fixture must actually be invalid JSON"
    );

    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": "read the file"},
            {"type": "function_call", "call_id": "call_trunc", "name": "exec_command", "arguments": truncated},
            {"type": "function_call_output", "call_id": "call_trunc", "output": "error: truncated"}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();

    let args = messages[1]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .expect("arguments must be a string");
    // The repaired arguments MUST be valid JSON — that is the whole point.
    serde_json::from_str::<serde_json::Value>(args)
        .expect("repaired arguments must be valid JSON");
    // The raw attempted text is preserved so the model still sees it.
    assert!(args.contains("bash"));
}

#[test]
fn test_function_call_valid_arguments_passed_through_unchanged() {
    // Valid JSON must pass through byte-identical — the repair path must not
    // alter the overwhelmingly common case.
    let valid = r#"{"city":"Stockholm"}"#;
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": valid},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(
        messages[0]["tool_calls"][0]["function"]["arguments"],
        valid
    );
}

#[test]
fn test_function_call_unquoted_value_recovers_parameter_name() {
    // Exact failure shape captured from production logs (DeepSeek-V4-Flash,
    // tool='exec_command', 37 consecutive failing turns on one poisoned
    // history item): the model names the parameter correctly but writes the
    // value as RAW TEXT instead of a JSON string literal — no quotes, literal
    // newlines, unescaped inner quotes.
    let malformed = "{\"command\": cd /workspace/git/ezennin/ericai-proxy && python3 -c \"\ncontent = open('src/config.rs').read()\n";
    assert!(
        serde_json::from_str::<serde_json::Value>(malformed).is_err(),
        "fixture must actually be invalid JSON"
    );

    let req = json!({
        "model": "deepseek-v4-flash",
        "input": [
            {"type": "function_call", "call_id": "chatcmpl-tool-b48fa86b", "name": "exec_command", "arguments": malformed},
            {"type": "function_call_output", "call_id": "chatcmpl-tool-b48fa86b", "output": "err"}
        ]
    });
    let out = responses_to_chat_completions("deepseek", &req, true);
    let messages = out["messages"].as_array().unwrap();

    let args = messages[0]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .expect("arguments must be a string");

    // Must now be valid JSON — this is what prevents the upstream 400.
    let parsed: serde_json::Value =
        serde_json::from_str(args).expect("repaired arguments must be valid JSON");

    // The original parameter name is preserved (not replaced by a generic
    // "input" wrapper), and the intended command text is retained.
    let command = parsed
        .get("command")
        .and_then(|v| v.as_str())
        .expect("expected the original 'command' parameter to be recovered");
    assert!(command.contains("python3 -c"));
    assert!(command.contains("src/config.rs"));
}

#[test]
fn test_developer_role_becomes_user() {
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "developer", "content": "Do things"}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "user");
}

#[test]
fn test_assistant_role_preserved() {
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "assistant", "content": "I can help."}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "assistant");
}

#[test]
fn test_apply_patch_tool_conversion() {
    let req = json!({
        "model": "glm-5.2",
        "input": "test",
        "tools": [
            {"type": "custom", "name": "apply_patch", "description": "Edit files using a patch format."}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let tools = out["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "apply_patch");
    assert!(tools[0]["function"]["parameters"]["properties"]["input"].is_object());
}

#[test]
fn test_function_tool_conversion() {
    let req = json!({
        "model": "glm-5.2",
        "input": "test",
        "tools": [
            {"type": "function", "name": "get_weather", "description": "Get weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let tools = out["tools"].as_array().unwrap();
    assert_eq!(tools[0]["function"]["name"], "get_weather");
    assert_eq!(
        tools[0]["function"]["parameters"]["properties"]["city"]["type"],
        "string"
    );
}

#[test]
fn test_builtin_tools_skipped() {
    let req = json!({
        "model": "glm-5.2",
        "input": "test",
        "tools": [
            {"type": "web_search", "name": "web_search"},
            {"type": "function", "name": "my_tool", "description": "test",
             "parameters": {"type": "object", "properties": {}}}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let tools = out["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], "my_tool");
}

#[test]
fn test_reasoning_effort_passed_through() {
    let req = json!({
        "model": "glm-5.2",
        "input": "test",
        "reasoning": {"effort": "high"}
    });
    let out = responses_to_chat_completions("glm", &req, true);
    assert_eq!(out["reasoning_effort"], "high");
}

#[test]
fn test_max_output_tokens_mapped() {
    let req = json!({
        "model": "glm-5.2",
        "input": "test",
        "max_output_tokens": 4096
    });
    let out = responses_to_chat_completions("glm", &req, true);
    assert_eq!(out["max_tokens"], 4096);
}

#[test]
fn test_parallel_tool_calls_passed() {
    let req = json!({
        "model": "glm-5.2",
        "input": "test",
        "parallel_tool_calls": true
    });
    let out = responses_to_chat_completions("glm", &req, true);
    assert_eq!(out["parallel_tool_calls"], true);
}

#[test]
fn test_tool_choice_passed() {
    let req = json!({
        "model": "glm-5.2",
        "input": "test",
        "tool_choice": "auto"
    });
    let out = responses_to_chat_completions("glm", &req, true);
    assert_eq!(out["tool_choice"], "auto");
}

// ─── apply_patch round-trip tests ───────────────────────────────────────────

#[test]
fn test_custom_tool_call_input_wrapped_as_json_arguments() {
    // When codex replays a prior apply_patch custom_tool_call in conversation
    // history, the proxy must wrap the raw patch text as JSON {"input": "..."}
    // in the assistant tool_calls.arguments field. Passing raw patch text as
    // arguments causes GLM-5.2 to return an empty response on follow-up turns.
    let patch = "*** Begin Patch\n*** Add File: /tmp/test.txt\n+hello\n*** End Patch";
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": "create a file"},
            {"type": "custom_tool_call", "call_id": "call_abc", "name": "apply_patch", "input": patch},
            {"type": "custom_tool_call_output", "call_id": "call_abc", "output": "patch applied"}
        ]
    });
    let out = responses_to_chat_completions("zai-org/GLM-5.2-FP8", &req, true);
    let messages = out["messages"].as_array().unwrap();
    // user + assistant(tool_calls) + tool
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1]["role"], "assistant");
    let tool_calls = messages[1]["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls[0]["function"]["name"], "apply_patch");
    assert_eq!(tool_calls[0]["id"], "call_abc");
    // arguments MUST be valid JSON with an "input" field containing the patch
    let args_str = tool_calls[0]["function"]["arguments"].as_str().unwrap();
    let args: Value = serde_json::from_str(args_str).expect("arguments must be valid JSON");
    assert_eq!(args["input"], patch);
    // The tool result is emitted as a role=tool message
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "call_abc");
    assert_eq!(messages[2]["content"], "patch applied");
}

#[test]
fn test_custom_tool_call_empty_input_still_valid_json() {
    // Edge case: empty input should still produce valid JSON {"input": ""}
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "custom_tool_call", "call_id": "call_x", "name": "apply_patch", "input": ""},
            {"type": "custom_tool_call_output", "call_id": "call_x", "output": ""}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    let args_str = messages[0]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    let args: Value = serde_json::from_str(args_str).expect("arguments must be valid JSON");
    assert_eq!(args["input"], "");
}

#[test]
fn test_inject_thinking_deepseek() {
    let payload = json!({
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "reasoning_effort": "high",
        "messages": [],
        "stream": true
    });
    let out = inject_thinking(payload, "deepseek-ai/DeepSeek-V4-Pro");
    assert_eq!(out["chat_template_kwargs"]["thinking"], true);
    assert_eq!(out["chat_template_kwargs"]["reasoning_effort"], "high");
    assert_eq!(out["stream_options"]["include_usage"], true);
    // reasoning object should be removed
    assert!(out.get("reasoning").is_none());
}

#[test]
fn test_inject_thinking_deepseek_xhigh() {
    let payload = json!({
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "reasoning_effort": "xhigh",
        "stream": true
    });
    let out = inject_thinking(payload, "deepseek-ai/DeepSeek-V4-Pro");
    assert_eq!(out["chat_template_kwargs"]["reasoning_effort"], "max");
    // xhigh normalized to "high" for reasoning_effort field
    assert_eq!(out["reasoning_effort"], "high");
}

#[test]
fn test_inject_thinking_glm_enable() {
    let payload = json!({
        "model": "zai-org/GLM-5.2-FP8",
        "reasoning_effort": "high",
        "stream": true
    });
    let out = inject_thinking(payload, "zai-org/GLM-5.2-FP8");
    assert_eq!(out["chat_template_kwargs"]["enable_thinking"], true);
}

#[test]
fn test_inject_thinking_glm_low_disables() {
    let payload = json!({
        "model": "zai-org/GLM-5.2-FP8",
        "reasoning_effort": "low",
        "stream": true
    });
    let out = inject_thinking(payload, "zai-org/GLM-5.2-FP8");
    assert_eq!(out["chat_template_kwargs"]["enable_thinking"], false);
}

#[test]
fn test_inject_thinking_non_reasoning_model() {
    let payload = json!({
        "model": "mimo-v2.5",
        "reasoning_effort": "high",
        "stream": true
    });
    let out = inject_thinking(payload, "XiaomiMiMo/MiMo-V2.5-Pro");
    // No chat_template_kwargs for non-deepseek/glm models
    assert!(out.get("chat_template_kwargs").is_none());
    // reasoning_effort still normalized
    assert_eq!(out["reasoning_effort"], "high");
}

#[test]
fn test_inject_thinking_no_stream_no_stream_options() {
    let payload = json!({
        "model": "glm-5.2",
        "reasoning_effort": "high",
        "stream": false
    });
    let out = inject_thinking(payload, "zai-org/GLM-5.2-FP8");
    // stream_options should NOT be present when stream=false
    assert!(out.get("stream_options").is_none());
}

#[test]
fn test_inject_thinking_removes_reasoning_object() {
    let payload = json!({
        "model": "glm-5.2",
        "reasoning": {"effort": "high"},
        "reasoning_effort": "high",
        "stream": true
    });
    let out = inject_thinking(payload, "zai-org/GLM-5.2-FP8");
    assert!(out.get("reasoning").is_none());
}

#[test]
fn test_empty_input_array() {
    let req = json!({
        "model": "glm-5.2",
        "input": []
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 0);
}

#[test]
fn test_input_image_content() {
    let req = json!({
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": "https://example.com/img.png"},
                {"type": "input_text", "text": "What's this?"}
            ]}
        ]
    });
    let out = responses_to_chat_completions("glm", &req, true);
    let messages = out["messages"].as_array().unwrap();
    let content = messages[0]["content"].as_str().unwrap();
    assert!(content.contains("[image:"));
    assert!(content.contains("What's this?"));
}
