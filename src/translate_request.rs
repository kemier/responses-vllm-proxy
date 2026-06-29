use serde_json::{json, Value};

/// Convert OpenAI Responses API request → Chat Completions request.
/// Ported from the Go translator in CLIProxyAPI.
pub fn responses_to_chat_completions(model: &str, input: &Value, stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "messages": [],
        "stream": stream,
    });

    // max_output_tokens → max_tokens
    if let Some(max) = input.get("max_output_tokens").and_then(|v| v.as_i64()) {
        out["max_tokens"] = json!(max);
    }

    // parallel_tool_calls
    if let Some(ptc) = input.get("parallel_tool_calls") {
        out["parallel_tool_calls"] = ptc.clone();
    }

    // instructions → system message
    if let Some(instructions) = input.get("instructions").and_then(|v| v.as_str()) {
        out["messages"].as_array_mut().unwrap().push(json!({
            "role": "system",
            "content": instructions,
        }));
    }

    // input array → messages
    if let Some(input_arr) = input.get("input").and_then(|v| v.as_array()) {
        // Track which call_ids have function_call_output
        let mut output_call_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for item in input_arr {
            if item.get("type").and_then(|v| v.as_str()) == Some("function_call_output") {
                if let Some(cid) = item.get("call_id").and_then(|v| v.as_str()) {
                    let cid = cid.trim().to_string();
                    if !cid.is_empty() {
                        output_call_ids.insert(cid);
                    }
                }
            }
        }

        let mut pending_tool_calls: Vec<Value> = Vec::new();
        let mut pending_tool_call_ids: Vec<String> = Vec::new();
        let mut awaiting_tool_outputs: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut deferred_messages: Vec<Value> = Vec::new();

        let messages = out["messages"].as_array_mut().unwrap();

        let flush_pending = |pending: &mut Vec<Value>, pending_ids: &mut Vec<String>, messages: &mut Vec<Value>, awaiting: &mut std::collections::HashSet<String>| {
           if pending.is_empty() {
               return;
           }
            // Merge tool_calls into the previous assistant message if it has
            // content but no tool_calls. The Chat Completions API expects a
            // single assistant message with both content and tool_calls —
            // two consecutive assistant messages confuse some models (e.g.
            // qwen3 emits EOS after the text preamble instead of continuing
            // to a tool call on subsequent turns).
            let should_merge = messages.last()
                .and_then(|m| m.get("role").and_then(|r| r.as_str()))
                .map(|r| r == "assistant")
                .unwrap_or(false)
                && messages.last()
                    .map(|m| m.get("tool_calls").is_none())
                    .unwrap_or(false);

            if should_merge {
                if let Some(last) = messages.last_mut() {
                    last["tool_calls"] = json!(pending.clone());
                }
            } else {
                messages.push(json!({"role": "assistant", "tool_calls": pending.clone()}));
            }
           for id in pending_ids.drain(..) {
               if !id.trim().is_empty() {
                   awaiting.insert(id);
               }
           }
           pending.clear();
       };

        let flush_deferred = |deferred: &mut Vec<Value>, messages: &mut Vec<Value>| {
            for msg in deferred.drain(..) {
                messages.push(msg);
            }
        };

        for item in input_arr {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let item_type = if item_type.is_empty() && item.get("role").is_some() {
                "message"
            } else {
                item_type
            };

            if item_type != "function_call" {
                flush_pending(&mut pending_tool_calls, &mut pending_tool_call_ids, messages, &mut awaiting_tool_outputs);
            }

            match item_type {
                "message" | "" => {
                    let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                    let role = if role == "developer" { "user" } else { role };

                    let mut message = json!({"role": role, "content": []});

                    if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                        let mut text_parts: Vec<String> = Vec::new();
                        for ci in content {
                            let ct = ci.get("type").and_then(|v| v.as_str()).unwrap_or("input_text");
                            match ct {
                                "input_text" | "text" | "output_text" => {
                                    if let Some(t) = ci.get("text").and_then(|v| v.as_str()) {
                                        text_parts.push(t.to_string());
                                    }
                                }
                                "input_image" => {
                                    if let Some(url) = ci.get("image_url").and_then(|v| v.as_str()) {
                                        text_parts.push(format!("[image: {}]", url));
                                    }
                                }
                                _ => {
                                    if let Some(t) = ci.get("text").and_then(|v| v.as_str()) {
                                        text_parts.push(t.to_string());
                                    }
                                }
                            }
                        }
                        message["content"] = json!(text_parts.join("\n"));
                    } else if let Some(text) = item.get("content").and_then(|v| v.as_str()) {
                        message["content"] = json!(text);
                    }

                    // Check if we're awaiting tool outputs
                    let has_awaiting = awaiting_tool_outputs.iter().any(|id| output_call_ids.contains(id));
                    if has_awaiting {
                        deferred_messages.push(message);
                    } else {
                        messages.push(message);
                    }
                }

                "function_call" => {
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let arguments = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
                    pending_tool_calls.push(json!({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }));
                    pending_tool_call_ids.push(call_id.to_string());
                }

                "custom_tool_call" => {
                    // Codex sends apply_patch calls as custom_tool_call items
                    // with an "input" field (raw patch text) instead of "arguments"
                    // (JSON). The Chat Completions tool schema we advertise for
                    // apply_patch declares a single `input` string parameter, so
                    // the model expects arguments to be JSON: {"input": "..."}.
                    //
                    // If we pass the raw patch text as `arguments` directly, the
                    // model sees a malformed (non-JSON) arguments string in its
                    // conversation history on follow-up turns and returns an
                    // empty response (0 output tokens). Wrap the input as JSON
                    // so the assistant tool_calls entry matches the schema.
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let input = item.get("input").and_then(|v| v.as_str()).unwrap_or("");
                    let arguments = json!({"input": input}).to_string();
                    pending_tool_calls.push(json!({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }));
                    pending_tool_call_ids.push(call_id.to_string());
                }

                "function_call_output" | "custom_tool_call_output" => {
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let output = item.get("output").map(|v| {
                        v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string())
                    }).unwrap_or_default();
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output,
                    }));
                    awaiting_tool_outputs.remove(call_id);
                    if awaiting_tool_outputs.is_empty() && !deferred_messages.is_empty() {
                        flush_deferred(&mut deferred_messages, messages);
                    }
                }

                _ => {}
            }
        }
        flush_pending(&mut pending_tool_calls, &mut pending_tool_call_ids, messages, &mut awaiting_tool_outputs);
        flush_deferred(&mut deferred_messages, messages);
    } else if let Some(text) = input.get("input").and_then(|v| v.as_str()) {
        out["messages"].as_array_mut().unwrap().push(json!({
            "role": "user", "content": text,
        }));
    }

    // tools
    if let Some(tools) = input.get("tools").and_then(|v| v.as_array()) {
        let mut cc_tools: Vec<Value> = Vec::new();
        for tool in tools {
            let tool_type = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");

            // apply_patch custom tool
            if tool_type == "custom" && tool.get("name").and_then(|v| v.as_str()) == Some("apply_patch") {
                cc_tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "description": "Edit files using a patch format.",
                        "strict": false,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {"type": "string", "description": "Patch content. MUST start with *** Begin Patch"}
                            },
                            "required": ["input"]
                        }
                    }
                }));
                continue;
            }

            // Skip built-in tools (web_search, image_generation, etc.)
            if !tool_type.is_empty() && tool_type != "function" && tool.is_object() {
                continue;
            }

            let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let desc = tool.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let params = tool.get("parameters").cloned().unwrap_or(json!({}));
            cc_tools.push(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params,
                }
            }));
        }
        if !cc_tools.is_empty() {
            out["tools"] = json!(cc_tools);
        }
    }

    // reasoning.effort → reasoning_effort
    if let Some(effort) = input.get("reasoning").and_then(|r| r.get("effort")).and_then(|v| v.as_str()) {
        let effort = effort.to_lowercase();
        if !effort.is_empty() {
            out["reasoning_effort"] = json!(effort);
        }
    }

    // tool_choice
    if let Some(tc) = input.get("tool_choice") {
        out["tool_choice"] = tc.clone();
    }

    out
}

/// Inject chat_template_kwargs for models that need explicit thinking activation.
/// Also normalize reasoning_effort (xhigh → high for non-DeepSeek).
pub fn inject_thinking(mut payload: Value, model: &str) -> Value {
    let lower_model = model.to_lowercase();

    let effort = payload.get("reasoning_effort").and_then(|v| v.as_str()).map(|s| s.to_string());

    if let Some(ref effort) = effort {
        let normalized = match effort.as_str() {
            "low" | "medium" | "high" => effort.clone(),
            "xhigh" => "high".to_string(),
            _ => {
                payload.as_object_mut().unwrap().remove("reasoning_effort");
                String::new()
            }
        };
        if !normalized.is_empty() {
            payload["reasoning_effort"] = json!(normalized);
        }
    }

    // Remove "reasoning" object (not valid for chat completions)
    payload.as_object_mut().map(|m| m.remove("reasoning"));

   // Inject chat_template_kwargs
   if let Some(effort) = &effort {
       if lower_model.contains("deepseek") {
           let ds_effort = if effort == "xhigh" { "max" } else { "high" };
           payload["chat_template_kwargs"] = json!({
               "thinking": true,
               "reasoning_effort": ds_effort,
           });
       } else if lower_model.contains("glm") {
           let enable_thinking = effort != "low";
           payload["chat_template_kwargs"] = json!({"enable_thinking": enable_thinking});
       }
   }

    // Qwen3 models: always inject chat_template_kwargs with enable_thinking.
    // Qwen3-Coder-Instruct defaults to non-thinking, but the chat template
    // still checks enable_thinking to select the correct output format
    // (including tool-call formatting). Without it, the model may emit EOS
    // prematurely after a short text preamble instead of making a tool call.
    if lower_model.contains("qwen3") || lower_model.contains("qwen-3") {
        let enable_thinking = match effort.as_deref() {
            Some("low") | Some("medium") | Some("high") | Some("xhigh") => true,
            _ => false, // "none" or not set -> non-thinking mode
        };
        payload["chat_template_kwargs"] = json!({"enable_thinking": enable_thinking});
    }

    // Set a default max_tokens if not already provided.
    // Codex doesn't send max_output_tokens for models whose catalog entry
    // has max_output_tokens = null. Without max_tokens, vLLM may apply a
    // very small default or the model may behave unpredictably.
    if payload.get("max_tokens").is_none() {
        payload["max_tokens"] = json!(16384);
    }

   // Request usage in stream (only if streaming)
   if payload.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
       payload["stream_options"] = json!({"include_usage": true});
   }

    payload
}
