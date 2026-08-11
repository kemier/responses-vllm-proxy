use serde_json::{json, Value};

use crate::logger;

/// Truncate a string to at most `max` characters without splitting a
/// multi-byte UTF-8 character (plain byte slicing would panic).
fn preview(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Attempt to recover a malformed `{"key": <unquoted raw text>` arguments
/// object into valid JSON, preserving the original parameter name.
///
/// Observed in the wild from DeepSeek-V4-Flash for `exec_command`: the model
/// opens the object and names the parameter correctly but then writes the
/// value as raw text instead of a JSON string literal — no surrounding
/// quotes, literal newlines, unescaped inner quotes:
///
/// ```text
/// {"command": cd /some/dir && python3 -c "
/// content = open('src/config.rs').read()
/// ```
///
/// Recovering the key (rather than falling back to a generic `input` wrapper)
/// keeps the replayed history faithful to what the model meant to call, so
/// the conversation stays coherent on subsequent turns.
fn repair_unquoted_json_object(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let rest = trimmed.strip_prefix('{')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let key_end = rest.find('"')?;
    let key = &rest[..key_end];
    if key.is_empty() {
        return None;
    }
    let after = rest[key_end + 1..].trim_start();
    let after = after.strip_prefix(':')?.trim_start();
    // Drop a trailing '}' if the model did close the object.
    let mut value = after.trim_end();
    if let Some(v) = value.strip_suffix('}') {
        value = v.trim_end();
    }
    // If the value happens to be a (possibly unterminated) quoted string,
    // strip the quotes so we do not double-encode them.
    let value = value
        .strip_prefix('"')
        .map(|v| v.trim_end_matches('"'))
        .unwrap_or(value);
    if value.is_empty() {
        return None;
    }
    Some(json!({ key: value }).to_string())
}

/// Ensure a function-call `arguments` string is valid JSON before it is
/// replayed to vLLM as part of an assistant `tool_calls` entry.
///
/// vLLM rejects the ENTIRE request with HTTP 400 ("Assistant tool call
/// function.arguments must be valid JSON.") when any assistant tool call in
/// the conversation history carries a non-JSON arguments string. Because the
/// offending item lives in the *history*, every subsequent turn resends it
/// and fails identically — the session stays permanently wedged (restarting
/// the proxy does not help) until the history is repaired here.
///
/// Observed sources of invalid arguments:
///   * empty string — tool called with no parameters and the stream never
///     produced an `arguments` delta, leaving the accumulator empty
///   * unquoted value — DeepSeek-V4-Flash emits `{"command": <raw text>`
///     without JSON-encoding the string value (confirmed in production logs,
///     `exec_command`, 37 consecutive failing turns on one poisoned item)
///
/// Repair strategy, in order of fidelity:
///   1. pass through untouched when it already parses (the common case)
///   2. recover `{"key": <raw>}` preserving the parameter name
///   3. last resort, wrap the raw text under an `input` key
fn ensure_valid_json_arguments(tool_name: &str, call_id: &str, arguments: &str) -> String {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return "{}".to_string();
    }
    if serde_json::from_str::<Value>(trimmed).is_ok() {
        return trimmed.to_string();
    }

    let (repaired, how) = match repair_unquoted_json_object(trimmed) {
        Some(r) => (r, "recovered original parameter name"),
        None => (json!({ "input": arguments }).to_string(), "wrapped as input"),
    };

    logger::warn(
        "translate_request",
        &format!(
            "[repair] invalid JSON tool arguments — tool='{}' call_id='{}' len={} ({}) \
             preview={:?} — repaired to prevent upstream HTTP 400 \
             (\"Assistant tool call function.arguments must be valid JSON.\")",
            tool_name,
            call_id,
            arguments.len(),
            how,
            preview(trimmed, 300),
        ),
    );
    repaired
}

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
        let mut output_call_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
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
        let mut awaiting_tool_outputs: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut deferred_messages: Vec<Value> = Vec::new();

        let messages = out["messages"].as_array_mut().unwrap();

        let flush_pending =
            |pending: &mut Vec<Value>,
             pending_ids: &mut Vec<String>,
             messages: &mut Vec<Value>,
             awaiting: &mut std::collections::HashSet<String>| {
                if pending.is_empty() {
                    return;
                }
                // Merge tool_calls into the previous assistant message if it has
                // content but no tool_calls. The Chat Completions API expects a
                // single assistant message with both content and tool_calls —
                // two consecutive assistant messages confuse some models (e.g.
                // qwen3 emits EOS after the text preamble instead of continuing
                // to a tool call on subsequent turns).
                let should_merge = messages
                    .last()
                    .and_then(|m| m.get("role").and_then(|r| r.as_str()))
                    .map(|r| r == "assistant")
                    .unwrap_or(false)
                    && messages
                        .last()
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
                flush_pending(
                    &mut pending_tool_calls,
                    &mut pending_tool_call_ids,
                    messages,
                    &mut awaiting_tool_outputs,
                );
            }

            match item_type {
                "message" | "" => {
                    let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                    let role = if role == "developer" { "user" } else { role };

                    let mut message = json!({"role": role, "content": []});

                    if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                        let mut text_parts: Vec<String> = Vec::new();
                        for ci in content {
                            let ct = ci
                                .get("type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("input_text");
                            match ct {
                                "input_text" | "text" | "output_text" => {
                                    if let Some(t) = ci.get("text").and_then(|v| v.as_str()) {
                                        text_parts.push(t.to_string());
                                    }
                                }
                                "input_image" => {
                                    if let Some(url) = ci.get("image_url").and_then(|v| v.as_str())
                                    {
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
                    let has_awaiting = awaiting_tool_outputs
                        .iter()
                        .any(|id| output_call_ids.contains(id));
                    if has_awaiting {
                        deferred_messages.push(message);
                    } else {
                        messages.push(message);
                    }
                }

                "function_call" => {
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    // Guard the replayed history: a non-JSON arguments string
                    // (empty, or truncated by max_tokens mid-tool-call) makes
                    // vLLM reject the whole request with HTTP 400, wedging
                    // every subsequent turn of the session.
                    let arguments = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
                    let arguments = ensure_valid_json_arguments(name, call_id, arguments);
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
                    //
                    // Same applies to Code Mode's `exec` and `wait` tools,
                    // which are freeform (custom) tools that accept raw text
                    // input (JavaScript source / wait parameters).
                    //
                    // NOTE: `wait` is actually a regular function tool, not
                    // custom. It appears here only if codex sends it as
                    // custom_tool_call (shouldn't happen in normal flow).
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
                    let output = item
                        .get("output")
                        .map(|v| {
                            v.as_str()
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| v.to_string())
                        })
                        .unwrap_or_default();
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
        flush_pending(
            &mut pending_tool_calls,
            &mut pending_tool_call_ids,
            messages,
            &mut awaiting_tool_outputs,
        );
        flush_deferred(&mut deferred_messages, messages);
    } else if let Some(text) = input.get("input").and_then(|v| v.as_str()) {
        out["messages"].as_array_mut().unwrap().push(json!({
            "role": "user", "content": text,
        }));
    }

    // tools
    if let Some(tools) = input.get("tools").and_then(|v| v.as_array()) {
        crate::logger::debug(
            "translate_request",
            &format!("[tools] incoming {} tools from codex", tools.len()),
        );
        for tool in tools {
            let ty = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let nm = tool
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| {
                    tool.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                });
            crate::logger::debug(
                "translate_request",
                &format!("[tools]   incoming: type={} name={}", ty, nm),
            );
        }
       let mut cc_tools: Vec<Value> = Vec::new();
       for tool in tools {
           let tool_type = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");

           // Namespace tools (e.g. multi_agent_v1 with spawn_agent, send_input,
           // close_agent, etc.) — flatten into individual function tools.
           // The Responses API sends these as:
           //   { "type": "namespace", "name": "multi_agent_v1",
           //     "description": "...",
           //     "tools": [ { "type": "function", "name": "spawn_agent", ... }, ... ] }
           // The Chat Completions API has no namespace concept, so we flatten
           // each sub-tool into a standalone function with a namespaced name
           // like "multi_agent_v1__spawn_agent". The proxy translates the
           // function call name back when sending the response to Codex.
           if tool_type == "namespace" {
               let ns_name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
               let ns_desc = tool.get("description").and_then(|v| v.as_str()).unwrap_or("");
               let sub_tools = tool.get("tools").and_then(|v| v.as_array());
               if let Some(sub_tools) = sub_tools {
                   for sub_tool in sub_tools {
                       let sub_name = sub_tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
                       let sub_desc = sub_tool.get("description").and_then(|v| v.as_str()).unwrap_or("");
                       let sub_params = sub_tool.get("parameters").cloned().unwrap_or(json!({}));
                       let full_name = if ns_name.is_empty() {
                           sub_name.to_string()
                       } else {
                           format!("{}__{}", ns_name, sub_name)
                       };
                       // Prepend namespace description to sub-tool description
                       // for context, since the namespace grouping is lost.
                       let full_desc = if ns_desc.is_empty() {
                           sub_desc.to_string()
                       } else {
                           format!("[{}] {}", ns_desc, sub_desc)
                       };
                       crate::logger::debug(
                           "translate_request",
                           &format!("[tools]   namespace flatten: {} -> {}", ns_name, full_name),
                       );
                       cc_tools.push(json!({
                           "type": "function",
                           "function": {
                               "name": full_name,
                               "description": full_desc,
                               "strict": false,
                               "parameters": sub_params,
                           }
                       }));
                   }
               }
               continue;
           }

           // apply_patch custom tool
           if tool_type == "custom"
               && tool.get("name").and_then(|v| v.as_str()) == Some("apply_patch")
            {
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

            // Code Mode freeform tool: exec only
            // This is sent as type "custom" with a freeform format (grammar-based).
            // vLLM/Chat Completions doesn't support freeform tools, so we convert
            // it to a standard function tool with a single `input` string parameter.
            // The model generates JSON {"input": "..."} as arguments, which the
            // proxy translates back to custom_tool_call items with raw text input.
            //
            // NOTE: `wait` is a regular ToolSpec::Function (type "function"),
            // NOT a freeform/custom tool. It falls through to the generic
            // function tool handling below with its original parameters
            // (cell_id, yield_time_ms, max_tokens, terminate).
            if tool_type == "custom" {
                let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
                if name == "exec" {
                    let desc = tool
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    cc_tools.push(json!({
                       "type": "function",
                       "function": {
                           "name": name,
                           "description": desc,
                           "strict": false,
                           "parameters": {
                               "type": "object",
                               "properties": {
                                   "input": {
                                       "type": "string",
                                       "description": "JavaScript source code to execute. Optionally start with // @exec: {\"yield_time_ms\": 10000, \"max_output_tokens\": 1000}"
                                   }
                               },
                               "required": ["input"]
                           }
                       }
                   }));
                    continue;
                }
            }

            // Skip built-in tools (web_search, image_generation, etc.)
            if !tool_type.is_empty() && tool_type != "function" && tool.is_object() {
                continue;
            }

            let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let desc = tool
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
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
            // Debug: log tool names and description lengths sent to vLLM
            for t in &cc_tools {
                let name = t
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let desc_len = t
                    .get("function")
                    .and_then(|f| f.get("description"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.len())
                    .unwrap_or(0);
                crate::logger::debug(
                    "translate_request",
                    &format!("[tools] vLLM tool: {} (desc {} chars)", name, desc_len),
                );
            }
            out["tools"] = json!(cc_tools);
        }
    }

    // reasoning.effort → reasoning_effort
    if let Some(effort) = input
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(|v| v.as_str())
    {
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

    let effort = payload
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

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
    if payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        payload["stream_options"] = json!({"include_usage": true});
    }

    payload
}
