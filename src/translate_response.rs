use serde_json::{json, Value};
use std::collections::HashMap;

/// State machine for translating Chat Completions SSE -> Responses API SSE.
/// Ported from Go's oaiToResponsesState.
pub struct ResponseTranslator {
    pub response_id: String,
    pub created: i64,
    pub seq: i64,
    started: bool,
    completion_pending: bool,
    completed_emitted: bool,

    // Reasoning state
    reasoning_id: Option<String>,
    reasoning_index: Option<usize>,
    reasoning_buf: String,

    // Message text buffers per choice index
    msg_text_buf: HashMap<usize, String>,
    msg_output_ix: HashMap<usize, usize>,
    msg_item_added: HashMap<usize, bool>,
    msg_content_added: HashMap<usize, bool>,
    msg_item_done: HashMap<usize, bool>,

    // Function call state: key = "choice_idx:tool_idx"
    func_args_buf: HashMap<String, String>,
    func_names: HashMap<String, String>,
    func_call_ids: HashMap<String, String>,
    func_output_ix: HashMap<String, usize>,
    func_item_done: HashMap<String, bool>,
    func_item_added: HashMap<String, bool>,

    next_output_ix: usize,

    // Usage aggregation
    prompt_tokens: i64,
    cached_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
    reasoning_tokens: i64,
    usage_seen: bool,

    // Think tag state
    in_think_tag: bool,

    // Original request for completed event
    request_json: Option<Value>,
}

impl ResponseTranslator {
    pub fn new(response_id: &str, request_json: Option<Value>) -> Self {
        Self {
            response_id: response_id.to_string(),
            created: chrono::Utc::now().timestamp(),
            seq: 0,
            started: false,
            completion_pending: false,
            completed_emitted: false,
            reasoning_id: None,
            reasoning_index: None,
            reasoning_buf: String::new(),
            msg_text_buf: HashMap::new(),
            msg_output_ix: HashMap::new(),
            msg_item_added: HashMap::new(),
            msg_content_added: HashMap::new(),
            msg_item_done: HashMap::new(),
            func_args_buf: HashMap::new(),
            func_names: HashMap::new(),
            func_call_ids: HashMap::new(),
            func_output_ix: HashMap::new(),
            func_item_done: HashMap::new(),
            func_item_added: HashMap::new(),
            next_output_ix: 0,
            prompt_tokens: 0,
            cached_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            reasoning_tokens: 0,
            usage_seen: false,
            in_think_tag: false,
            request_json,
        }
    }

    fn next_seq(&mut self) -> i64 {
        self.seq += 1;
        self.seq
    }

    fn alloc_output_index(&mut self) -> usize {
        let ix = self.next_output_ix;
        self.next_output_ix += 1;
        ix
    }

    fn tool_state_key(choice_idx: usize, tool_idx: i64) -> String {
        format!("{}:{}", choice_idx, tool_idx)
    }

    /// Process one SSE `data:` line from chat completions.
    /// Returns zero or more Responses API SSE event strings.
    pub fn process_line(&mut self, line: &str) -> Vec<String> {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            return vec![];
        }
        if let Some(data) = line.strip_prefix("data: ") {
            return self.process_data(data);
        }
        if let Some(data) = line.strip_prefix("data:") {
            return self.process_data(data);
        }
        vec![]
    }

    fn process_data(&mut self, data: &str) -> Vec<String> {
        let data = data.trim();
        if data == "[DONE]" {
            return self.flush_and_complete();
        }

        let parsed: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return vec![],
        };

        let mut out = Vec::new();

        // Emit response.created on first chunk
        if !self.started {
            self.started = true;
            let created = json!({
                "type": "response.created",
                "sequence_number": self.next_seq(),
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created,
                    "status": "in_progress",
                    "background": false,
                    "error": null,
                }
            });
            out.push(self.emit("response.created", &created));

            let in_progress = json!({
                "type": "response.in_progress",
                "sequence_number": self.next_seq(),
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "status": "in_progress",
                }
            });
            out.push(self.emit("response.in_progress", &in_progress));
        }

        // Check for usage in the chunk
        if let Some(usage) = parsed.get("usage") {
            if usage.is_object() {
                self.usage_seen = true;
                self.prompt_tokens = usage.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                self.completion_tokens = usage.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                self.total_tokens = usage.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                if let Some(pd) = usage.get("prompt_tokens_details") {
                    self.cached_tokens = pd.get("cached_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                }
                if let Some(od) = usage.get("output_tokens_details") {
                    self.reasoning_tokens = od.get("reasoning_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                }
            }
        }

        // Process choices
        if let Some(choices) = parsed.get("choices").and_then(|v| v.as_array()) {
            for choice in choices {
                let idx = choice.get("index").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
                let delta = match choice.get("delta") {
                    Some(d) => d,
                    None => continue,
                };

                let finish_reason = choice.get("finish_reason").and_then(|v| v.as_str()).unwrap_or("");
                if !finish_reason.is_empty() {
                    self.completion_pending = true;
                }

                // Content (text)
                if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                    if !content.is_empty() {
                        let events = self.handle_content(idx, content);
                        out.extend(events);
                    }
                }

                // reasoning_content
                if let Some(rc) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                    if !rc.is_empty() {
                        let events = self.handle_reasoning_content(idx, rc);
                        out.extend(events);
                    }
                }

                // Tool calls
                if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    if !tcs.is_empty() {
                        let events = self.handle_tool_calls(idx, tcs);
                        out.extend(events);
                    }
                }

                // Handle finish: close any open items
                if !finish_reason.is_empty() {
                    let events = self.handle_finish(idx);
                    out.extend(events);
                }
            }
        }

        out
    }

    fn handle_content(&mut self, idx: usize, mut content: &str) -> Vec<String> {
        let mut out = Vec::new();

        // Strip think tags
        if !self.in_think_tag {
            if let Some(start) = content.find("<think>") {
                let before = &content[..start];
                if !before.is_empty() {
                    let events = self.emit_text_delta(idx, before);
                    out.extend(events);
                }
                self.in_think_tag = true;
                content = &content[start + 7..]; // skip "<think>"
            }
        }

        if self.in_think_tag {
            if let Some(end) = content.find("</think>") {
                let think_content = &content[..end];
                if !think_content.is_empty() {
                    let events = self.handle_reasoning_content(idx, think_content);
                    out.extend(events);
                }
                self.in_think_tag = false;
                content = &content[end + 8..]; // skip "</think>"
                content = content.trim_start_matches('\n');
            } else {
                // Still inside think tag
                if !content.is_empty() {
                    let events = self.handle_reasoning_content(idx, content);
                    out.extend(events);
                }
                return out;
            }
        }

        if content.is_empty() {
            return out;
        }

        // Stop reasoning if active
        if self.reasoning_id.is_some() {
            let events = self.stop_reasoning();
            out.extend(events);
        }

        out.extend(self.emit_text_delta(idx, content));
        out
    }

    fn emit_text_delta(&mut self, idx: usize, content: &str) -> Vec<String> {
        let mut out = Vec::new();

        if !self.msg_output_ix.contains_key(&idx) {
            let ix = self.alloc_output_index();
            self.msg_output_ix.insert(idx, ix);
        }
        let msg_output_index = self.msg_output_ix[&idx];
        let item_id = format!("msg_{}_{}", self.response_id, idx);

        if !self.msg_item_added.get(&idx).copied().unwrap_or(false) {
            let item = json!({
                "type": "response.output_item.added",
                "sequence_number": self.next_seq(),
                "output_index": msg_output_index,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "status": "in_progress",
                    "content": [],
                    "role": "assistant",
                }
            });
            out.push(self.emit("response.output_item.added", &item));
            self.msg_item_added.insert(idx, true);
        }

        if !self.msg_content_added.get(&idx).copied().unwrap_or(false) {
            let part = json!({
                "type": "response.content_part.added",
                "sequence_number": self.next_seq(),
                "item_id": item_id,
                "output_index": msg_output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": "",
                }
            });
            out.push(self.emit("response.content_part.added", &part));
            self.msg_content_added.insert(idx, true);
        }

        let delta = json!({
            "type": "response.output_text.delta",
            "sequence_number": self.next_seq(),
            "item_id": item_id,
            "output_index": msg_output_index,
            "content_index": 0,
            "delta": content,
            "logprobs": [],
        });
        out.push(self.emit("response.output_text.delta", &delta));

        self.msg_text_buf.entry(idx).or_default().push_str(content);

        out
    }

    fn handle_reasoning_content(&mut self, idx: usize, rc: &str) -> Vec<String> {
        let mut out = Vec::new();

        if self.reasoning_id.is_none() {
            let rid = format!("rs_{}_{}", self.response_id, idx);
            let rix = self.alloc_output_index();
            self.reasoning_id = Some(rid.clone());
            self.reasoning_index = Some(rix);

            let item = json!({
                "type": "response.output_item.added",
                "sequence_number": self.next_seq(),
                "output_index": rix,
                "item": {
                    "id": rid,
                    "type": "reasoning",
                    "status": "in_progress",
                    "summary": [],
                }
            });
            out.push(self.emit("response.output_item.added", &item));

            let part = json!({
                "type": "response.reasoning_summary_part.added",
                "sequence_number": self.next_seq(),
                "item_id": rid,
                "output_index": rix,
                "summary_index": 0,
                "part": {
                    "type": "summary_text",
                    "text": "",
                }
            });
            out.push(self.emit("response.reasoning_summary_part.added", &part));
        }

        let rid = self.reasoning_id.as_ref().unwrap().clone();
        let rix = self.reasoning_index.unwrap();

        let delta = json!({
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": self.next_seq(),
            "item_id": rid,
            "output_index": rix,
            "summary_index": 0,
            "delta": rc,
        });
        out.push(self.emit("response.reasoning_summary_text.delta", &delta));

        self.reasoning_buf.push_str(rc);

        out
    }

    fn stop_reasoning(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(rid) = self.reasoning_id.take() {
            let rix = self.reasoning_index.take().unwrap();

            let done = json!({
                "type": "response.reasoning_summary_text.done",
                "sequence_number": self.next_seq(),
                "item_id": rid,
                "output_index": rix,
                "summary_index": 0,
                "text": self.reasoning_buf.clone(),
            });
            out.push(self.emit("response.reasoning_summary_text.done", &done));

            let part_done = json!({
                "type": "response.reasoning_summary_part.done",
                "sequence_number": self.next_seq(),
                "item_id": rid,
                "output_index": rix,
                "summary_index": 0,
                "part": {
                    "type": "summary_text",
                    "text": self.reasoning_buf.clone(),
                }
            });
            out.push(self.emit("response.reasoning_summary_part.done", &part_done));

            let item_done = json!({
                "type": "response.output_item.done",
                "sequence_number": self.next_seq(),
                "output_index": rix,
                "item": {
                    "id": rid,
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [{
                        "type": "summary_text",
                        "text": self.reasoning_buf.clone(),
                    }],
                }
            });
            out.push(self.emit("response.output_item.done", &item_done));

            self.reasoning_buf.clear();
        }
        out
    }

    fn handle_tool_calls(&mut self, idx: usize, tcs: &[Value]) -> Vec<String> {
        let mut out = Vec::new();

        if self.reasoning_id.is_some() {
            out.extend(self.stop_reasoning());
        }

        if self.msg_item_added.get(&idx).copied().unwrap_or(false)
            && !self.msg_item_done.get(&idx).copied().unwrap_or(false)
        {
            out.extend(self.close_message(idx));
        }

        for tc in tcs {
            let tool_idx = tc.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
            let key = Self::tool_state_key(idx, tool_idx);
            let new_call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let name_chunk = tc.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()).unwrap_or("").to_string();

            if !name_chunk.is_empty() {
                self.func_names.insert(key.clone(), name_chunk);
            }

            let should_emit_item = !self.func_item_added.get(&key).copied().unwrap_or(false)
                && !new_call_id.is_empty();

            if should_emit_item {
                self.func_call_ids.insert(key.clone(), new_call_id.clone());
                let oix = self.alloc_output_index();
                self.func_output_ix.insert(key.clone(), oix);
                self.func_item_added.insert(key.clone(), true);

                let name = self.func_names.get(&key).cloned().unwrap_or_default();
                let item = json!({
                    "type": "response.output_item.added",
                    "sequence_number": self.next_seq(),
                    "output_index": oix,
                    "item": {
                        "id": format!("fc_{}", new_call_id),
                        "type": "function_call",
                        "status": "in_progress",
                        "arguments": "",
                        "call_id": new_call_id,
                        "name": name,
                    }
                });
                out.push(self.emit("response.output_item.added", &item));
            }

            if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")).and_then(|v| v.as_str()) {
                if !args.is_empty() {
                    let call_id = self.func_call_ids.get(&key).cloned().unwrap_or(new_call_id.clone());
                    if !call_id.is_empty() {
                        let oix = self.func_output_ix.get(&key).copied().unwrap_or(0);

                        let delta = json!({
                            "type": "response.function_call_arguments.delta",
                            "sequence_number": self.next_seq(),
                            "item_id": format!("fc_{}", call_id),
                            "output_index": oix,
                            "delta": args,
                        });
                        out.push(self.emit("response.function_call_arguments.delta", &delta));

                        self.func_args_buf.entry(key.clone()).or_default().push_str(args);
                    }
                }
            }
        }

        out
    }

    fn close_message(&mut self, idx: usize) -> Vec<String> {
        let mut out = Vec::new();
        let item_id = format!("msg_{}_{}", self.response_id, idx);
        let msg_output_index = self.msg_output_ix.get(&idx).copied().unwrap_or(0);
        let full_text = self.msg_text_buf.get(&idx).cloned().unwrap_or_default();

        let done = json!({
            "type": "response.output_text.done",
            "sequence_number": self.next_seq(),
            "item_id": item_id,
            "output_index": msg_output_index,
            "content_index": 0,
            "text": full_text,
            "logprobs": [],
        });
        out.push(self.emit("response.output_text.done", &done));

        let part_done = json!({
            "type": "response.content_part.done",
            "sequence_number": self.next_seq(),
            "item_id": item_id,
            "output_index": msg_output_index,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": full_text,
            }
        });
        out.push(self.emit("response.content_part.done", &part_done));

        let item_done = json!({
            "type": "response.output_item.done",
            "sequence_number": self.next_seq(),
            "output_index": msg_output_index,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": full_text,
                }],
                "role": "assistant",
            }
        });
        out.push(self.emit("response.output_item.done", &item_done));

        self.msg_item_done.insert(idx, true);
        out
    }

    fn handle_finish(&mut self, idx: usize) -> Vec<String> {
        let mut out = Vec::new();

        if self.reasoning_id.is_some() {
            out.extend(self.stop_reasoning());
        }

        if self.msg_item_added.get(&idx).copied().unwrap_or(false)
            && !self.msg_item_done.get(&idx).copied().unwrap_or(false)
        {
            out.extend(self.close_message(idx));
        }

        let prefix = format!("{}:", idx);
        let keys_to_close: Vec<String> = self.func_item_added.keys()
            .filter(|k| k.starts_with(&prefix)
                && !self.func_item_done.get(*k).copied().unwrap_or(false))
            .cloned()
            .collect();

        for key in keys_to_close {
            out.extend(self.close_function_call(&key));
        }

        out
    }

    fn close_function_call(&mut self, key: &str) -> Vec<String> {
        let mut out = Vec::new();
        let call_id = self.func_call_ids.get(key).cloned().unwrap_or_default();
        if call_id.is_empty() {
            return out;
        }
        let oix = self.func_output_ix.get(key).copied().unwrap_or(0);
        let args = self.func_args_buf.get(key).cloned().unwrap_or_default();
        let name = self.func_names.get(key).cloned().unwrap_or_default();

        let done = json!({
            "type": "response.function_call_arguments.done",
            "sequence_number": self.next_seq(),
            "item_id": format!("fc_{}", call_id),
            "output_index": oix,
            "arguments": args,
        });
        out.push(self.emit("response.function_call_arguments.done", &done));

        let item_done = json!({
            "type": "response.output_item.done",
            "sequence_number": self.next_seq(),
            "output_index": oix,
            "item": {
                "id": format!("fc_{}", call_id),
                "type": "function_call",
                "status": "completed",
                "arguments": args,
                "call_id": call_id,
                "name": name,
            }
        });
        out.push(self.emit("response.output_item.done", &item_done));

        self.func_item_done.insert(key.to_string(), true);
        out
    }

    fn flush_and_complete(&mut self) -> Vec<String> {
        let mut out = Vec::new();

        if self.reasoning_id.is_some() {
            out.extend(self.stop_reasoning());
        }

        let open_msgs: Vec<usize> = self.msg_item_added.keys()
            .filter(|k| !self.msg_item_done.get(k).copied().unwrap_or(false))
            .copied()
            .collect();
        for idx in open_msgs {
            out.extend(self.close_message(idx));
        }

        let open_funcs: Vec<String> = self.func_item_added.keys()
            .filter(|k| !self.func_item_done.get(*k).copied().unwrap_or(false))
            .cloned()
            .collect();
        for key in open_funcs {
            out.extend(self.close_function_call(&key));
        }

        if !self.completed_emitted {
            self.completed_emitted = true;
            out.push(self.build_completed_event());
        }

        out
    }

    fn build_completed_event(&mut self) -> String {
        let mut response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created,
            "status": "completed",
            "background": false,
            "error": null,
        });

        if let Some(req) = &self.request_json {
            for field in ["instructions", "max_output_tokens", "max_tool_calls", "model",
                "parallel_tool_calls", "previous_response_id", "prompt_cache_key",
                "reasoning", "safety_identifier", "service_tier", "store",
                "temperature", "text", "tool_choice", "tools", "top_logprobs",
                "top_p", "truncation", "user", "metadata"] {
                if let Some(v) = req.get(field) {
                    response[field] = v.clone();
                }
            }
        }

        let mut output = Vec::new();

        for (idx, text) in &self.msg_text_buf {
            let item = json!({
                "id": format!("msg_{}_{}", self.response_id, idx),
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": text,
                }],
                "role": "assistant",
            });
            output.push(item);
        }

        for (key, call_id) in &self.func_call_ids {
            let args = self.func_args_buf.get(key).cloned().unwrap_or_default();
            let name = self.func_names.get(key).cloned().unwrap_or_default();
            let item = json!({
                "id": format!("fc_{}", call_id),
                "type": "function_call",
                "status": "completed",
                "arguments": args,
                "call_id": call_id,
                "name": name,
            });
            output.push(item);
        }

        response["output"] = json!(output);

        if self.usage_seen {
            response["usage"] = json!({
                "input_tokens": self.prompt_tokens,
                "input_tokens_details": {
                    "cached_tokens": self.cached_tokens,
                },
                "output_tokens": self.completion_tokens,
                "output_tokens_details": {
                    "reasoning_tokens": self.reasoning_tokens,
                },
                "total_tokens": self.total_tokens,
            });
        } else {
            response["usage"] = json!({
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            });
        }

        let completed = json!({
            "type": "response.completed",
            "sequence_number": self.next_seq(),
            "response": response,
        });

        self.emit("response.completed", &completed)
    }

    fn emit(&mut self, event: &str, payload: &Value) -> String {
        let mut p = payload.clone();
        if let Some(obj) = p.as_object_mut() {
            obj.insert("type".to_string(), json!(event));
        }
        format!("event: {}\ndata: {}\n\n", event, p)
    }
}

/// Convert a non-streaming chat completions response to a Responses API response object.
/// Used for compact and non-streaming /v1/responses.
pub fn chat_completion_to_response(raw: &Value, request_json: Option<&Value>) -> Value {
    let id = raw.get("id").and_then(|v| v.as_str()).unwrap_or("resp_unknown").to_string();
    let response_id = if id.starts_with("chatcmpl_") {
        format!("resp_{}", &id[9..])
    } else if id.starts_with("resp_") {
        id.clone()
    } else {
        format!("resp_{}", id)
    };

    let created = raw.get("created").and_then(|v| v.as_i64()).unwrap_or_else(|| chrono::Utc::now().timestamp());

    let mut resp = json!({
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "background": false,
        "error": null,
    });

    if let Some(req) = request_json {
        for field in ["instructions", "max_output_tokens", "max_tool_calls", "model",
            "parallel_tool_calls", "previous_response_id", "prompt_cache_key",
            "reasoning", "safety_identifier", "service_tier", "store",
            "temperature", "text", "tool_choice", "tools", "top_logprobs",
            "top_p", "truncation", "user", "metadata"] {
            if let Some(v) = req.get(field) {
                resp[field] = v.clone();
            }
        }
    } else if let Some(model) = raw.get("model") {
        resp["model"] = model.clone();
    }

    let mut output = Vec::new();

    let rc_text = raw.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("reasoning_content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let include_reasoning = !rc_text.is_empty()
        || request_json.and_then(|r| r.get("reasoning")).is_some();

    if include_reasoning {
        let rid = if response_id.starts_with("resp_") {
            response_id[5..].to_string()
        } else {
            response_id.clone()
        };
        let mut reasoning_item = json!({
            "id": format!("rs_{}", rid),
            "type": "reasoning",
            "encrypted_content": "",
            "summary": [],
        });
        if !rc_text.is_empty() {
            reasoning_item["summary"] = json!([{
                "type": "summary_text",
                "text": rc_text,
            }]);
        }
        output.push(reasoning_item);
    }

    if let Some(choices) = raw.get("choices").and_then(|v| v.as_array()) {
        for choice in choices {
            if let Some(msg) = choice.get("message") {
                if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                    if !content.is_empty() {
                        let choice_idx = choice.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
                        let item = json!({
                            "id": format!("msg_{}_{}", response_id, choice_idx),
                            "type": "message",
                            "status": "completed",
                            "content": [{
                                "type": "output_text",
                                "annotations": [],
                                "logprobs": [],
                                "text": content,
                            }],
                            "role": "assistant",
                        });
                        output.push(item);
                    }
                }

                if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in tcs {
                        let call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let name = tc.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let args = tc.get("function").and_then(|f| f.get("arguments")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let item = json!({
                            "id": format!("fc_{}", call_id),
                            "type": "function_call",
                            "status": "completed",
                            "arguments": args,
                            "call_id": call_id,
                            "name": name,
                        });
                        output.push(item);
                    }
                }
            }
        }
    }

    resp["output"] = json!(output);

    if let Some(usage) = raw.get("usage") {
        if usage.get("prompt_tokens").is_some() || usage.get("completion_tokens").is_some() || usage.get("total_tokens").is_some() {
            resp["usage"] = json!({
                "input_tokens": usage.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
                "input_tokens_details": {
                    "cached_tokens": usage.get("prompt_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                },
                "output_tokens": usage.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
                "output_tokens_details": {
                    "reasoning_tokens": usage.get("output_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                },
                "total_tokens": usage.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
            });
        } else {
            resp["usage"] = usage.clone();
        }
    }

    resp
}
