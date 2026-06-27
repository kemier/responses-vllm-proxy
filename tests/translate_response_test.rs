use ericai_proxy::translate_response::{ResponseTranslator, chat_completion_to_response};
use serde_json::json;

// ─── ResponseTranslator (streaming) tests ───────────────────────────────────

#[test]
fn test_simple_text_streaming() {
    let mut t = ResponseTranslator::new("resp_test1", None);

    // First chunk: content delta
    let events = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#);
    // Should emit: response.created, response.in_progress, output_item.added, content_part.added, output_text.delta
    assert!(events.len() >= 5);
    assert!(events[0].contains("response.created"));
    assert!(events[1].contains("response.in_progress"));
    assert!(events.iter().any(|e| e.contains("response.output_item.added")));
    assert!(events.iter().any(|e| e.contains("response.content_part.added")));
    assert!(events.iter().any(|e| e.contains("response.output_text.delta") && e.contains("Hello")));
}

#[test]
fn test_multi_chunk_text_streaming() {
    let mut t = ResponseTranslator::new("resp_test2", None);

    let e1 = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#);
    let e2 = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":", world"}}]}"#);
    let e3 = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":"!"}}]}"#);

    // First chunk has setup events, subsequent chunks just have deltas
    assert!(e1.iter().any(|e| e.contains("response.output_text.delta")));
    assert!(e2.iter().any(|e| e.contains("response.output_text.delta") && e.contains(", world")));
    assert!(e3.iter().any(|e| e.contains("response.output_text.delta") && e.contains("!")));
}

#[test]
fn test_done_emits_completion_events() {
    let mut t = ResponseTranslator::new("resp_test3", None);

    t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#);
    let done_events = t.process_line("data: [DONE]");

    // Should emit: output_text.done, content_part.done, output_item.done, response.completed
    assert!(done_events.iter().any(|e| e.contains("response.output_text.done")));
    assert!(done_events.iter().any(|e| e.contains("response.content_part.done")));
    assert!(done_events.iter().any(|e| e.contains("response.output_item.done")));
    assert!(done_events.iter().any(|e| e.contains("response.completed")));

    // The completed event should contain the full text
    let completed = done_events.iter().find(|e| e.contains("response.completed")).unwrap();
    assert!(completed.contains("Hi"));
}

#[test]
fn test_finish_reason_closes_message() {
    let mut t = ResponseTranslator::new("resp_test4", None);

    t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Done"}}]}"#);
    let finish_events = t.process_line(r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#);

    // Should close the message: output_text.done, content_part.done, output_item.done
    assert!(finish_events.iter().any(|e| e.contains("response.output_text.done")));
    assert!(finish_events.iter().any(|e| e.contains("response.output_item.done")));
}

#[test]
fn test_tool_call_streaming() {
    let mut t = ResponseTranslator::new("resp_test5", None);

    // First chunk: tool call with id and name
    let e1 = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"get_weather","arguments":""}}]}}]}"#);
    assert!(e1.iter().any(|e| e.contains("response.output_item.added") && e.contains("function_call")));
    assert!(e1.iter().any(|e| e.contains("get_weather")));

    // Second chunk: argument delta
    let e2 = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}"#);
    assert!(e2.iter().any(|e| e.contains("response.function_call_arguments.delta")));

    // Third chunk: more arguments
    let e3 = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Stockholm\"}"}}]}}]}"#);
    assert!(e3.iter().any(|e| e.contains("response.function_call_arguments.delta")));

    // Done
    let done = t.process_line("data: [DONE]");
    assert!(done.iter().any(|e| e.contains("response.function_call_arguments.done")));
    assert!(done.iter().any(|e| e.contains("response.output_item.done") && e.contains("function_call")));
    assert!(done.iter().any(|e| e.contains("response.completed")));

    // Verify accumulated arguments in the done event
    let args_done = done.iter().find(|e| e.contains("response.function_call_arguments.done")).unwrap();
    assert!(args_done.contains("Stockholm"));
}

#[test]
fn test_reasoning_content_streaming() {
    let mut t = ResponseTranslator::new("resp_test6", None);

    let e1 = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"reasoning_content":"Let me think..."}}]}"#);
    assert!(e1.iter().any(|e| e.contains("response.output_item.added") && e.contains("reasoning")));
    assert!(e1.iter().any(|e| e.contains("response.reasoning_summary_part.added")));
    assert!(e1.iter().any(|e| e.contains("response.reasoning_summary_text.delta") && e.contains("Let me think")));
}

#[test]
fn test_reasoning_then_text() {
    let mut t = ResponseTranslator::new("resp_test7", None);

    // Reasoning first
    t.process_line(r#"data: {"choices":[{"index":0,"delta":{"reasoning_content":"Thinking..."}}]}"#);
    // Then text content
    let text_events = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Answer"}}]}"#);

    // Should stop reasoning (emit done events) then start message
    assert!(text_events.iter().any(|e| e.contains("response.reasoning_summary_text.done")));
    assert!(text_events.iter().any(|e| e.contains("response.output_item.done")));
    assert!(text_events.iter().any(|e| e.contains("response.output_text.delta") && e.contains("Answer")));
}

#[test]
fn test_usage_in_stream() {
    let mut t = ResponseTranslator::new("resp_test8", None);

    t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#);
    // finish_reason + usage chunk closes the message but doesn't emit completed
    t.process_line(r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#);
    // [DONE] triggers the completed event with usage
    let done = t.process_line("data: [DONE]");

    let completed = done.iter().find(|e| e.contains("response.completed")).unwrap();
    assert!(completed.contains("input_tokens"));
    assert!(completed.contains("\"input_tokens\":10"));
}

#[test]
fn test_empty_lines_ignored() {
    let mut t = ResponseTranslator::new("resp_test9", None);
    assert_eq!(t.process_line("").len(), 0);
    assert_eq!(t.process_line("   ").len(), 0);
    assert_eq!(t.process_line(": comment").len(), 0);
}

#[test]
fn test_invalid_json_ignored() {
    let mut t = ResponseTranslator::new("resp_test10", None);
    assert_eq!(t.process_line("data: not json").len(), 0);
    assert_eq!(t.process_line("data: {broken").len(), 0);
}

#[test]
fn test_think_tag_stripped_to_reasoning() {
    let mut t = ResponseTranslator::new("resp_test11", None);

    // Content with think tag
    let events = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":"<think>Internal reasoning</think>Actual answer"}}]}"#);

    // Should route think content to reasoning, then output the actual answer
    assert!(events.iter().any(|e| e.contains("reasoning_summary_text.delta") && e.contains("Internal reasoning")));
    assert!(events.iter().any(|e| e.contains("response.output_text.delta") && e.contains("Actual answer")));
}

#[test]
fn test_multiple_tool_calls_same_choice() {
    let mut t = ResponseTranslator::new("resp_test12", None);

    // First tool call
    t.process_line(r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"tool_a","arguments":"{}"}}]}}]}"#);
    // Second tool call in same choice
    let e2 = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"tool_b","arguments":"{}"}}]}}]}"#);

    assert!(e2.iter().any(|e| e.contains("response.output_item.added") && e.contains("tool_b")));
}

// ─── apply_patch (custom tool) streaming tests ─────────────────────────────
// Regression tests for the double-emission bug: apply_patch argument fragments
// must NOT be emitted as streaming custom_tool_call_input.delta events (they
// are raw JSON fragments like `{"in` / `put": "*** ` which corrupt Codex's
// StreamingPatchParser). Instead, the full extracted patch text must be emitted
// as a single clean delta at close time.

#[test]
fn test_apply_patch_no_streaming_deltas_for_arguments() {
    let mut t = ResponseTranslator::new("resp_patch1", None);

    // First chunk: apply_patch tool call with id and name
    let e1 = t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_patch1","function":{"name":"apply_patch","arguments":""}}]}}]}"#,
    );
    // Should emit the output_item.added as custom_tool_call
    assert!(e1.iter().any(|e| e.contains("response.output_item.added") && e.contains("custom_tool_call")));
    assert!(e1.iter().any(|e| e.contains("apply_patch")));

    // Second chunk: argument fragment (raw JSON, incomplete)
    let e2 = t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"in"}}]}}]}"#,
    );
    // CRITICAL: must NOT emit any custom_tool_call_input.delta for this fragment
    assert!(
        !e2.iter().any(|e| e.contains("response.custom_tool_call_input.delta")),
        "apply_patch argument fragments must not be streamed as deltas"
    );

    // Third chunk: more argument fragments
    let e3 = t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"put\": \"*** Begin Patch\\n"}}]}}]}"#,
    );
    assert!(
        !e3.iter().any(|e| e.contains("response.custom_tool_call_input.delta")),
        "apply_patch argument fragments must not be streamed as deltas"
    );

    // Fourth chunk: rest of the patch
    let e4 = t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"*** End Patch\"}"}}]}}]}"#,
    );
    assert!(
        !e4.iter().any(|e| e.contains("response.custom_tool_call_input.delta")),
        "apply_patch argument fragments must not be streamed as deltas"
    );
}

#[test]
fn test_apply_patch_emits_clean_delta_on_done() {
    let mut t = ResponseTranslator::new("resp_patch2", None);

    // Stream an apply_patch call with JSON-wrapped arguments split across chunks
    t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_patch2","function":{"name":"apply_patch","arguments":""}}]}}]}"#,
    );
    t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"input\": \"*** Begin Patch\\n"}}]}}]}"#,
    );
    t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"*** Update File: foo.txt\\n@@\\n+hello\\n*** End Patch\"}"}}]}}]}"#,
    );

    // Trigger close via [DONE]
    let done = t.process_line("data: [DONE]");

    // Should emit exactly ONE custom_tool_call_input.delta with the clean patch text
    let deltas: Vec<&String> = done
        .iter()
        .filter(|e| e.contains("response.custom_tool_call_input.delta"))
        .collect();
    assert_eq!(
        deltas.len(),
        1,
        "expected exactly one clean delta at close time, got {}",
        deltas.len()
    );

    // The delta should contain the extracted patch text, NOT raw JSON
    let delta = deltas[0];
    assert!(delta.contains("*** Begin Patch"));
    assert!(delta.contains("*** Update File: foo.txt"));
    assert!(delta.contains("*** End Patch"));
    // Should NOT contain JSON wrapper artifacts
    assert!(!delta.contains("\"input\""));
    assert!(!delta.contains("{\\\""));

    // Should emit the done event with the clean input
    assert!(done.iter().any(|e| e.contains("response.custom_tool_call_input.done")));
    let done_evt = done
        .iter()
        .find(|e| e.contains("response.custom_tool_call_input.done"))
        .unwrap();
    assert!(done_evt.contains("*** Begin Patch"));
    assert!(done_evt.contains("*** End Patch"));

    // Should emit output_item.done with custom_tool_call type
    assert!(done.iter().any(|e| e.contains("response.output_item.done") && e.contains("custom_tool_call")));

    // Should emit response.completed
    assert!(done.iter().any(|e| e.contains("response.completed")));
}

#[test]
fn test_apply_patch_finish_reason_closes_cleanly() {
    let mut t = ResponseTranslator::new("resp_patch3", None);

    // Stream apply_patch with finish_reason instead of [DONE]
    t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_patch3","function":{"name":"apply_patch","arguments":"{\"input\": \"*** Begin Patch\\n*** End Patch\"}"}}]}}]}"#,
    );
    let finish_events = t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
    );

    // finish_reason should trigger close_function_call
    assert!(finish_events.iter().any(|e| e.contains("response.custom_tool_call_input.delta")));
    assert!(finish_events.iter().any(|e| e.contains("response.custom_tool_call_input.done")));
    assert!(finish_events.iter().any(|e| e.contains("response.output_item.done") && e.contains("custom_tool_call")));

    // The done event should contain the extracted patch text
    let done_evt = finish_events
        .iter()
        .find(|e| e.contains("response.custom_tool_call_input.done"))
        .unwrap();
    assert!(done_evt.contains("*** Begin Patch"));
    assert!(done_evt.contains("*** End Patch"));
}

#[test]
fn test_apply_patch_non_json_fallback() {
    // If the model sends apply_patch arguments that are NOT valid JSON
    // (e.g. raw patch text without JSON wrapper), the fallback in
    // extract_custom_tool_input should use the raw string as input.
    let mut t = ResponseTranslator::new("resp_patch4", None);

    t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_patch4","function":{"name":"apply_patch","arguments":"*** Begin Patch\\n*** End Patch"}}]}}]}"#,
    );
    let done = t.process_line("data: [DONE]");

    let done_evt = done
        .iter()
        .find(|e| e.contains("response.custom_tool_call_input.done"))
        .unwrap();
    assert!(done_evt.contains("*** Begin Patch"));
    assert!(done_evt.contains("*** End Patch"));
}

#[test]
fn test_regular_function_call_still_streams_deltas() {
    // Ensure the fix doesn't break regular function calls — they should still
    // stream argument deltas as before.
    let mut t = ResponseTranslator::new("resp_fn1", None);

    t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_fn1","function":{"name":"get_weather","arguments":""}}]}}]}"#,
    );
    let e2 = t.process_line(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}"#,
    );
    assert!(e2.iter().any(|e| e.contains("response.function_call_arguments.delta")));
}

// ─── chat_completion_to_response (non-streaming) tests ──────────────────────

#[test]
fn test_non_streaming_simple_response() {
    let raw = json!({
        "id": "chatcmpl_abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-5.2",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    });

    let resp = chat_completion_to_response(&raw, None);
    assert_eq!(resp["object"], "response");
    assert_eq!(resp["status"], "completed");
    assert_eq!(resp["id"], "resp_abc123");

    let output = resp["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"][0]["text"], "Hello there!");
}

#[test]
fn test_non_streaming_with_tool_calls() {
    let raw = json!({
        "id": "chatcmpl_xyz",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-5.2",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_123",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Stockholm\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });

    let resp = chat_completion_to_response(&raw, None);
    let output = resp["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(output[0]["name"], "get_weather");
    assert_eq!(output[0]["call_id"], "call_123");
    assert!(output[0]["arguments"].as_str().unwrap().contains("Stockholm"));
}

#[test]
fn test_non_streaming_with_reasoning_content() {
    let raw = json!({
        "id": "chatcmpl_rc",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "deepseek-v4-pro",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "I need to calculate..."
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    });

    let resp = chat_completion_to_response(&raw, None);
    let output = resp["output"].as_array().unwrap();
    // Should have reasoning item + message item
    assert_eq!(output.len(), 2);
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["summary"][0]["text"], "I need to calculate...");
    assert_eq!(output[1]["type"], "message");
}

#[test]
fn test_non_streaming_usage_mapping() {
    let raw = json!({
        "id": "chatcmpl_usage",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-5.2",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 5}
        }
    });

    let resp = chat_completion_to_response(&raw, None);
    assert_eq!(resp["usage"]["input_tokens"], 100);
    assert_eq!(resp["usage"]["output_tokens"], 50);
    assert_eq!(resp["usage"]["total_tokens"], 150);
    assert_eq!(resp["usage"]["input_tokens_details"]["cached_tokens"], 20);
    assert_eq!(resp["usage"]["output_tokens_details"]["reasoning_tokens"], 5);
}

#[test]
fn test_non_streaming_id_prefix_conversion() {
    let raw = json!({
        "id": "chatcmpl_abc",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-5.2",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}]
    });
    let resp = chat_completion_to_response(&raw, None);
    assert_eq!(resp["id"], "resp_abc");
}

#[test]
fn test_non_streaming_already_resp_id() {
    let raw = json!({
        "id": "resp_already",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-5.2",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}]
    });
    let resp = chat_completion_to_response(&raw, None);
    assert_eq!(resp["id"], "resp_already");
}

#[test]
fn test_non_streaming_request_fields_injected() {
    let raw = json!({
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-5.2",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}]
    });
    let req = json!({
        "model": "glm-5.2",
        "instructions": "Be helpful",
        "max_output_tokens": 4096,
        "temperature": 0.7,
        "tool_choice": "auto"
    });
    let resp = chat_completion_to_response(&raw, Some(&req));
    assert_eq!(resp["model"], "glm-5.2");
    assert_eq!(resp["instructions"], "Be helpful");
    assert_eq!(resp["max_output_tokens"], 4096);
    assert_eq!(resp["temperature"], 0.7);
    assert_eq!(resp["tool_choice"], "auto");
}

#[test]
fn test_non_streaming_empty_content() {
    let raw = json!({
        "id": "chatcmpl_empty",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-5.2",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop"
        }]
    });
    let resp = chat_completion_to_response(&raw, None);
    let output = resp["output"].as_array().unwrap();
    // Empty content should not produce a message item
    assert_eq!(output.len(), 0);
}

#[test]
fn test_raw_json_mode_no_sse_envelope() {
    let mut t = ResponseTranslator::new_raw_json("resp_raw1", None);

    let events = t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#);

    // Should have multiple events (created, in_progress, output_item.added, etc.)
    assert!(events.len() >= 5);

    // In raw JSON mode, each event should be a JSON object (starts with '{')
    // and NOT contain SSE envelope prefixes like "event:" or "data:"
    for event in &events {
        assert!(event.starts_with('{'), "Expected JSON object, got: {}", &event[..20.min(event.len())]);
        assert!(!event.contains("event: "), "Should not contain SSE envelope");
        assert!(!event.contains("data: "), "Should not contain SSE data prefix");
    }

    // First event should be response.created
    assert!(events[0].contains("\"response.created\""));
}

#[test]
fn test_raw_json_mode_completed_event() {
    let mut t = ResponseTranslator::new_raw_json("resp_raw2", None);

    t.process_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#);
    let done_events = t.process_line("data: [DONE]");

    // The last event should be response.completed
    let last = done_events.last().unwrap();
    assert!(last.contains("\"response.completed\""));
    assert!(last.starts_with('{')); // raw JSON, not SSE
}
