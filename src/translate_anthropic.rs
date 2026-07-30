use serde_json::{json, Value};

// ─── Anthropic Messages API → OpenAI Chat Completions ──────────────────────

/// Convert an Anthropic Messages API request body to an OpenAI Chat
/// Completions request body.
pub fn anthropic_to_chat_completions(model: &str, body: &Value, stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "messages": [],
        "stream": stream,
    });

    // vLLM only emits a usage chunk in SSE when stream_options.include_usage
    // is set. Without it, token counts stay at 0 and the ACP usage UI never
    // updates.
    if stream {
        out["stream_options"] = json!({"include_usage": true});
    }

    if let Some(max) = body.get("max_tokens").and_then(|v| v.as_i64()) {
        out["max_tokens"] = json!(max);
    }
    if let Some(temp) = body.get("temperature") {
        out["temperature"] = temp.clone();
    }
    if let Some(top_p) = body.get("top_p") {
        out["top_p"] = top_p.clone();
    }

    // stop_sequences → stop
    if let Some(stops) = body.get("stop_sequences").and_then(|v| v.as_array()) {
        let stop_arr: Vec<&str> = stops.iter().filter_map(|s| s.as_str()).collect();
        if !stop_arr.is_empty() {
            out["stop"] = json!(stop_arr);
        }
    }

    let messages = out["messages"].as_array_mut().unwrap();

    // system → first system message (string or array of content blocks)
    if let Some(system) = body.get("system") {
        let system_text = extract_text_from_content(system);
        if !system_text.is_empty() {
            messages.push(json!({"role": "system", "content": system_text}));
        }
    }

    // Convert messages
    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            convert_anthropic_message(msg, messages);
        }
    }

    // Convert tools
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let openai_tools: Vec<Value> = tools.iter().filter_map(convert_anthropic_tool).collect();
        if !openai_tools.is_empty() {
            out["tools"] = json!(openai_tools);
        }
    }

    // tool_choice
    if let Some(tc) = body.get("tool_choice") {
        if let Some(tc_str) = tc.as_str() {
            match tc_str {
                "auto" => { out["tool_choice"] = json!("auto"); }
                "any" => { out["tool_choice"] = json!("required"); }
                _ => {}
            }
        } else if let Some(tc_obj) = tc.as_object() {
            if tc_obj.get("type").and_then(|v| v.as_str()) == Some("tool") {
                if let Some(name) = tc_obj.get("name").and_then(|v| v.as_str()) {
                    out["tool_choice"] = json!({"type": "function", "function": {"name": name}});
                }
            }
        }
    }

    out
}

/// Extract text from an Anthropic content field (string or array of blocks).
fn extract_text_from_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let mut parts = Vec::new();
            for block in arr {
                if block.get("type").and_then(|v| v.as_str()) == Some("text") {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                        parts.push(text.to_string());
                    }
                }
            }
            parts.join("\n")
        }
        _ => String::new(),
    }
}

/// Convert a single Anthropic message to one or more OpenAI messages.
fn convert_anthropic_message(msg: &Value, messages: &mut Vec<Value>) {
    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");

    match msg.get("content") {
        Some(Value::String(text)) => {
            messages.push(json!({"role": role, "content": text}));
        }
        Some(Value::Array(blocks)) => {
            if role == "assistant" {
                let mut text_parts = Vec::new();
                let mut tool_calls: Vec<Value> = Vec::new();

                for block in blocks {
                    match block.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                                text_parts.push(text.to_string());
                            }
                        }
                        Some("tool_use") => {
                            let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let input = block.get("input").cloned().unwrap_or(json!({}));
                            tool_calls.push(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": serde_json::to_string(&input).unwrap_or_default(),
                                }
                            }));
                        }
                        _ => {}
                    }
                }

                let content = if text_parts.is_empty() { Value::Null } else { json!(text_parts.join("\n")) };
                let mut assistant_msg = json!({"role": "assistant", "content": content});
                if !tool_calls.is_empty() {
                    assistant_msg["tool_calls"] = json!(tool_calls);
                }
                messages.push(assistant_msg);
            } else {
                // User message — may contain text and tool_result blocks
                let mut text_parts = Vec::new();
                let mut tool_results: Vec<(String, String)> = Vec::new();

                for block in blocks {
                    match block.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                                text_parts.push(text.to_string());
                            }
                        }
                        Some("tool_result") => {
                            let tool_use_id = block.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("");
                            let content = block.get("content").map(extract_text_from_content).unwrap_or_default();
                            tool_results.push((tool_use_id.to_string(), content));
                        }
                        _ => {}
                    }
                }

                for (tool_use_id, content) in &tool_results {
                    messages.push(json!({"role": "tool", "tool_call_id": tool_use_id, "content": content}));
                }
                if !text_parts.is_empty() {
                    messages.push(json!({"role": role, "content": text_parts.join("\n")}));
                }
                if text_parts.is_empty() && tool_results.is_empty() {
                    messages.push(json!({"role": role, "content": ""}));
                }
            }
        }
        _ => {
            messages.push(json!({"role": role, "content": ""}));
        }
    }
}

/// Convert an Anthropic tool definition to OpenAI function tool format.
fn convert_anthropic_tool(tool: &Value) -> Option<Value> {
    let name = tool.get("name").and_then(|v| v.as_str())?;
    let description = tool.get("description").and_then(|v| v.as_str()).unwrap_or("");
    let input_schema = tool.get("input_schema").cloned().unwrap_or(json!({}));
    Some(json!({
        "type": "function",
        "function": {"name": name, "description": description, "parameters": input_schema}
    }))
}

// ─── OpenAI Chat Completions SSE → Anthropic Messages SSE ──────────────────

/// State machine for translating OpenAI streaming SSE chunks to Anthropic
/// Messages SSE events.
pub struct AnthropicStreamTranslator {
    message_id: String,
    model: String,
    started: bool,
    current_block_index: i32,
    input_tokens: u64,
    output_tokens: u64,
    text_block_open: bool,
    // True once any text content block has been opened. Used by finish() to
    // decide whether to synthesize a text block from reasoning-only output.
    text_ever_emitted: bool,
    thinking_block_open: bool,
    // Accumulated reasoning text. GLM with thinking enabled may emit the entire
    // response via delta.reasoning with delta.content empty (reasoning-only
    // output). We accumulate it so finish() can synthesize a text content block
    // — without it, the Anthropic stream has only a thinking block and no text
    // block, and clients like Claude Code CLI's /compact fail with "no
    // assistant message in summarization response".
    reasoning_text: String,
    pending_tool_calls: Vec<PendingToolCall>,
    finished: bool,
    stop_reason: String,
}

#[derive(Default)]
struct PendingToolCall {
    index: usize,
    id: String,
    name: String,
    arguments_buffer: String,
    emitted: bool,
}

impl AnthropicStreamTranslator {
    pub fn new(model: &str) -> Self {
        Self {
            message_id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            model: model.to_string(),
            started: false,
            current_block_index: 0,
            input_tokens: 0,
            output_tokens: 0,
            text_block_open: false,
            text_ever_emitted: false,
            thinking_block_open: false,
            reasoning_text: String::new(),
            pending_tool_calls: Vec::new(),
            finished: false,
            stop_reason: String::new(),
        }
    }

    /// Process one OpenAI SSE data line and return Anthropic SSE event strings.
    pub fn translate_line(&mut self, line: &str) -> Vec<String> {
        let trimmed = line.trim();
        if !trimmed.starts_with("data: ") {
            return Vec::new();
        }
        let json_str = &trimmed[6..];
        if json_str.trim() == "[DONE]" {
            return self.finish();
        }

        let chunk: Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        let mut events = Vec::new();

        if !self.started {
            self.started = true;
            events.push(self.message_start_event());
        }

       // Extract usage
       if let Some(usage) = chunk.get("usage") {
           crate::logger::debug(
               "translate_anthropic",
               &format!("[usage] raw vLLM usage chunk: {}", serde_json::to_string(usage).unwrap_or_default()),
           );
            // vLLM streams usage in two chunks:
            //   1. Real usage (prompt_tokens > 0) — after finish_reason
            //   2. Zero-usage cleanup chunk — on connection close
            // Never overwrite real usage with zeros — the cleanup chunk
            // would reset token counts to 0, breaking the ACP usage UI.
            let pt = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            let ct = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            if pt > 0 || ct > 0 || (self.input_tokens == 0 && self.output_tokens == 0) {
                self.input_tokens = pt;
                self.output_tokens = ct;
            }
       }

        if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
            for choice in choices {
                let finish_reason = choice.get("finish_reason").and_then(|v| v.as_str());

                if let Some(delta) = choice.get("delta") {
                    // Reasoning → thinking content block
                    if let Some(reasoning) = delta.get("reasoning").and_then(|v| v.as_str()) {
                        if !reasoning.is_empty() {
                            // Accumulate for the reasoning-only fallback in finish().
                            self.reasoning_text.push_str(reasoning);
                            if !self.thinking_block_open && !self.text_block_open {
                                events.push(self.thinking_block_start_event());
                                self.thinking_block_open = true;
                            }
                            if self.thinking_block_open {
                                events.push(self.thinking_delta_event(reasoning));
                            }
                        }
                    }

                    // Text content
                    if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                        if !content.is_empty() {
                            // Close thinking block if open
                            if self.thinking_block_open {
                                events.push(self.content_block_stop_event(self.current_block_index));
                                self.current_block_index += 1;
                                self.thinking_block_open = false;
                            }
                            if !self.text_block_open {
                                events.push(self.text_block_start_event(self.current_block_index));
                                self.text_block_open = true;
                            }
                            self.text_ever_emitted = true;
                            events.push(self.text_delta_event(self.current_block_index, content));
                        }
                    }

                    // Tool calls (buffer arguments)
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tool_calls {
                            let tc_index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            while self.pending_tool_calls.len() <= tc_index {
                                self.pending_tool_calls.push(PendingToolCall::default());
                            }
                            let pending = &mut self.pending_tool_calls[tc_index];
                            pending.index = tc_index;
                            if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                if !id.is_empty() { pending.id = id.to_string(); }
                            }
                            if let Some(func) = tc.get("function") {
                                if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                    if !name.is_empty() { pending.name = name.to_string(); }
                                }
                                if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                                    pending.arguments_buffer.push_str(args);
                                }
                            }
                        }
                    }
                }

                // Handle finish_reason
                if let Some(reason) = finish_reason {
                    // Close any open block
                    if self.text_block_open {
                        events.push(self.content_block_stop_event(self.current_block_index));
                        self.current_block_index += 1;
                        self.text_block_open = false;
                    }
                    if self.thinking_block_open {
                        events.push(self.content_block_stop_event(self.current_block_index));
                        self.current_block_index += 1;
                        self.thinking_block_open = false;
                    }

                    // Emit pending tool_use blocks
                    for pending in &self.pending_tool_calls {
                        if !pending.emitted && !pending.name.is_empty() {
                            let block_index = self.current_block_index;
                            events.push(self.tool_use_block_start_event(block_index, &pending.id, &pending.name));
                            if !pending.arguments_buffer.is_empty() {
                                events.push(self.tool_use_delta_event(block_index, &pending.arguments_buffer));
                            }
                            events.push(self.content_block_stop_event(block_index));
                            self.current_block_index += 1;
                        }
                    }
                    for pending in &mut self.pending_tool_calls {
                        pending.emitted = true;
                    }

                    let stop_reason = match reason {
                        "stop" => "end_turn",
                        "tool_calls" | "function_call" => "tool_use",
                        "length" => "max_tokens",
                        _ => "end_turn",
                    };
                    // Defer message_delta emission to finish() — vLLM sends
                    // the usage chunk AFTER finish_reason, so emitting here
                    // would always show 0 tokens.
                    self.stop_reason = stop_reason.to_string();
                }
            }
        }

        events
    }

    fn finish(&mut self) -> Vec<String> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut events = Vec::new();
        if self.text_block_open {
            events.push(self.content_block_stop_event(self.current_block_index));
            self.current_block_index += 1;
            self.text_block_open = false;
        }
        if self.thinking_block_open {
            events.push(self.content_block_stop_event(self.current_block_index));
            self.current_block_index += 1;
            self.thinking_block_open = false;
        }
        // Reasoning-only fallback: GLM with thinking enabled may stream the
        // entire response via delta.reasoning and never emit delta.content. In
        // that case the Anthropic stream so far has only a thinking block and
        // no text block. Clients like Claude Code CLI create an assistant
        // message only from a text content_block_stop, so a thinking-only
        // response yields "no assistant message in summarization response" and
        // breaks /compact. Synthesize a text content block from the accumulated
        // reasoning so the response always carries an assistant text message.
        // (Mirrors rewrite_reasoning_to_content() used by /v1/chat/completions.)
        if !self.text_ever_emitted && !self.reasoning_text.is_empty() {
            events.push(self.text_block_start_event(self.current_block_index));
            events.push(self.text_delta_event(self.current_block_index, &self.reasoning_text));
            events.push(self.content_block_stop_event(self.current_block_index));
            self.current_block_index += 1;
        }
        if !self.started {
            self.started = true;
            events.push(self.message_start_event());
        }
        // Emit message_delta now — all usage chunks have been processed.
        let sr = if self.stop_reason.is_empty() { "end_turn" } else { self.stop_reason.as_str() };
        events.push(self.message_delta_event(sr));
        events.push(self.message_stop_event());
        events
    }

    fn message_start_event(&self) -> String {
        let event = json!({
            "type": "message_start",
            "message": {
                "id": self.message_id, "type": "message", "role": "assistant",
                "content": [], "model": self.model,
                "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 1}
            }
        });
        format!("event: message_start\ndata: {}\n\n", event)
    }

    fn text_block_start_event(&self, index: i32) -> String {
        let event = json!({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}});
        format!("event: content_block_start\ndata: {}\n\n", event)
    }

    fn thinking_block_start_event(&self) -> String {
        let event = json!({"type": "content_block_start", "index": self.current_block_index, "content_block": {"type": "thinking", "thinking": ""}});
        format!("event: content_block_start\ndata: {}\n\n", event)
    }

    fn text_delta_event(&self, index: i32, text: &str) -> String {
        let event = json!({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}});
        format!("event: content_block_delta\ndata: {}\n\n", event)
    }

    fn thinking_delta_event(&self, text: &str) -> String {
        let event = json!({"type": "content_block_delta", "index": self.current_block_index, "delta": {"type": "thinking_delta", "thinking": text}});
        format!("event: content_block_delta\ndata: {}\n\n", event)
    }

    fn content_block_stop_event(&self, index: i32) -> String {
        let event = json!({"type": "content_block_stop", "index": index});
        format!("event: content_block_stop\ndata: {}\n\n", event)
    }

    fn tool_use_block_start_event(&self, index: i32, id: &str, name: &str) -> String {
        let event = json!({"type": "content_block_start", "index": index, "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}});
        format!("event: content_block_start\ndata: {}\n\n", event)
    }

    fn tool_use_delta_event(&self, index: i32, partial_json: &str) -> String {
        let event = json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": partial_json}});
        format!("event: content_block_delta\ndata: {}\n\n", event)
    }

    fn message_delta_event(&self, stop_reason: &str) -> String {
        let event = json!({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": null}, "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}});
        format!("event: message_delta\ndata: {}\n\n", event)
    }

    fn message_stop_event(&self) -> String {
        let event = json!({"type": "message_stop"});
        format!("event: message_stop\ndata: {}\n\n", event)
    }
}

/// Convert a non-streaming OpenAI Chat Completions response to Anthropic format.
pub fn chat_completions_to_anthropic(model: &str, response: &Value) -> Value {
    let mut content: Vec<Value> = Vec::new();

    if let Some(choices) = response.get("choices").and_then(|c| c.as_array()) {
        if let Some(choice) = choices.first() {
            if let Some(msg) = choice.get("message") {
                let reasoning = msg.get("reasoning").and_then(|v| v.as_str()).unwrap_or("");
                if !reasoning.is_empty() {
                    content.push(json!({"type": "thinking", "thinking": reasoning}));
                }
                let text = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if !text.is_empty() {
                    content.push(json!({"type": "text", "text": text}));
                } else if !reasoning.is_empty() {
                    // Reasoning-only fallback (non-streaming): GLM with thinking
                    // enabled may return the full response in `reasoning` with
                    // empty `content`. Without a text block the client sees no
                    // assistant message (breaks Claude Code CLI /compact).
                    // Promote the reasoning to a text block. Mirrors the
                    // streaming translator's finish() fallback and
                    // rewrite_reasoning_to_content() in /v1/chat/completions.
                    content.push(json!({"type": "text", "text": reasoning}));
                }
                if let Some(tool_calls) = msg.get("tool_calls").and_then(|c| c.as_array()) {
                    for tc in tool_calls {
                        let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let name = tc.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()).unwrap_or("");
                        let args_str = tc.get("function").and_then(|f| f.get("arguments")).and_then(|v| v.as_str()).unwrap_or("{}");
                        let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                        content.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
                    }
                }
            }

            let stop_reason = match choice.get("finish_reason").and_then(|v| v.as_str()) {
                Some("stop") => "end_turn",
                Some("tool_calls") | Some("function_call") => "tool_use",
                Some("length") => "max_tokens",
                _ => "end_turn",
            };

            let mut out = json!({
                "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
                "type": "message", "role": "assistant", "content": content,
                "model": model, "stop_reason": stop_reason, "stop_sequence": null,
            });

            if let Some(usage) = response.get("usage") {
                let input_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                let output_tokens = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                out["usage"] = json!({"input_tokens": input_tokens, "output_tokens": output_tokens});
            }
            return out;
        }
    }

    json!({
        "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
        "type": "message", "role": "assistant", "content": [],
        "model": model, "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    })
}
