use serde_json::{json, Value};
use std::collections::HashMap;

/// Return the byte length of the longest suffix of `text` that is also a
/// prefix of `tag`. Used to detect partial tag matches at chunk boundaries.
///
/// For example, if `text` ends with "<thi" and `tag` is "<think>", this
/// returns 4 (the length of "<thi"), so the caller knows to hold back those
/// 4 bytes for the next chunk.
///
/// Returns at most `tag.len() - 1` — a full match means `find()` would have
/// located the tag already.
fn longest_suffix_prefix(text: &str, tag: &str) -> usize {
    if text.is_empty() || tag.is_empty() {
        return 0;
    }
    let max_len = text.len().min(tag.len() - 1);
    let mut len = max_len;
    while len > 0 {
        // Only check at valid UTF-8 character boundaries.
        if text.is_char_boundary(text.len() - len) {
            let suffix = &text[text.len() - len..];
            if tag.starts_with(suffix) {
                return len;
            }
        }
        len -= 1;
    }
    0
}

/// Check if a tool name corresponds to a freeform/custom tool (apply_patch).
/// These tools must be emitted as `custom_tool_call` items in the Responses API,
/// not as `function_call` items, because Codex dispatches them via the
/// `ApplyPatchHandler` which only accepts `ToolPayload::Custom`.
fn is_custom_tool(name: &str) -> bool {
    name == "apply_patch"
}

/// Extract the `input` field from a JSON arguments string.
///
/// The model sends apply_patch arguments as JSON: {"input": "*** Begin Patch\n..."}
/// The Responses API `custom_tool_call` item expects a plain `input` string,
/// not a JSON `arguments` string.
fn extract_custom_tool_input(arguments: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(arguments) {
        if let Some(input) = v.get("input").and_then(|v| v.as_str()) {
            return input.to_string();
        }
    }
    // Fallback: if arguments is not valid JSON or does not have "input",
    // use the raw arguments string as the tool input.
    arguments.to_string()
}

/// Per-choice-index message state, consolidating the former 5 separate HashMaps.
struct MessageState {
    text_buf: String,
    output_ix: usize,
    item_added: bool,
    content_added: bool,
    item_done: bool,
    /// Pre-computed item_id ("msg_{response_id}_{idx}") to avoid format!()
    /// on every text delta — the hottest path in the proxy.
    item_id: String,
}

/// Per-tool-call state (keyed by "choice_idx:tool_idx"), consolidating the
/// former 6 separate HashMaps.
struct FunctionCallState {
    args_buf: String,
    name: String,
    call_id: String,
    /// Pre-computed item_id ("fc_{call_id}") to avoid format!() on every
    /// tool call delta chunk — the hot path for tool calls.
    item_id: String,
    output_ix: usize,
    item_done: bool,
    item_added: bool,
}

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

    // Message state per choice index (single HashMap instead of 5)
    msg_states: HashMap<usize, MessageState>,

    // Function call state per "choice_idx:tool_idx" (single HashMap instead of 6)
    func_states: HashMap<String, FunctionCallState>,

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
    // Buffer for content that might contain a partial think tag at a chunk
    // boundary. When streaming, a tag like "<think>" can be split across
    // multiple chunks (e.g. "<thi" + "nk>"). We hold back any suffix of the
    // current chunk that is a prefix of the tag we're searching for, so the
    // complete tag can be detected when the next chunk arrives.
    think_pending: String,

    // When true, emit raw JSON strings (for WebSocket) instead of SSE
    // `event: ...\ndata: ...` envelopes (for HTTP SSE).
    raw_json: bool,

    // Original request for completed event
    request_json: Option<Value>,
}

impl ResponseTranslator {
    pub fn new(response_id: &str, request_json: Option<Value>) -> Self {
        Self::new_impl(response_id, request_json, false)
    }

    /// Create a translator that emits raw JSON strings (no SSE envelope),
    /// suitable for WebSocket transport where each frame is a single JSON object.
    pub fn new_raw_json(response_id: &str, request_json: Option<Value>) -> Self {
        Self::new_impl(response_id, request_json, true)
    }

    fn new_impl(response_id: &str, request_json: Option<Value>, raw_json: bool) -> Self {
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
            msg_states: HashMap::new(),
            func_states: HashMap::new(),
            next_output_ix: 0,
            prompt_tokens: 0,
            cached_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            reasoning_tokens: 0,
            usage_seen: false,
            in_think_tag: false,
            think_pending: String::new(),
            raw_json,
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
    /// Returns zero or more Responses API event strings.
    /// In SSE mode (default) each string is an SSE frame:
    ///   event: <event>\ndata: <json>\n\n
    /// In raw JSON mode each string is just the JSON object.
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
            out.push(self.emit("response.created", created));

            let in_progress = json!({
                "type": "response.in_progress",
                "sequence_number": self.next_seq(),
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "status": "in_progress",
                }
            });
            out.push(self.emit("response.in_progress", in_progress));
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

    fn handle_content(&mut self, idx: usize, content: &str) -> Vec<String> {
        let mut out = Vec::new();

        // Append the new chunk to the pending buffer. This lets us detect
        // think tags that are split across chunk boundaries.
        self.think_pending.push_str(content);

        // Process the buffer in a loop: each iteration either finds a complete
        // tag (and continues with the remainder) or determines that no more
        // complete tags can be found (and holds back a possible partial tag
        // prefix before returning).
        loop {
            if !self.in_think_tag {
                // Looking for opening tag "<think>"
                if let Some(start) = self.think_pending.find("<think>") {
                    // Emit any text before the tag as a text delta.
                    // Use mem::take to swap out the buffer, then split it.
                    // This avoids allocating a new String for `before` —
                    // we drain from the taken buffer instead.
                    if start > 0 {
                        let mut taken = std::mem::take(&mut self.think_pending);
                        let before: String = taken.drain(..start).collect();
                        // Put the remainder (after the tag) back.
                        taken.drain(..7); // skip the tag itself
                        self.think_pending = taken;
                        if !before.is_empty() {
                            if self.reasoning_id.is_some() {
                                out.extend(self.stop_reasoning());
                            }
                            out.extend(self.emit_text_delta(idx, &before));
                        }
                    } else {
                        // start == 0: just remove the tag
                        self.think_pending.drain(..7);
                    }
                    self.in_think_tag = true;
                    // Continue the loop to process the remainder (which may
                    // contain "</think>" or more text).
                    continue;
                } else {
                    // No complete opening tag found. Check whether the buffer
                    // ends with a prefix of "<think>" (e.g. "<thi"). If so,
                    // hold that suffix back for the next chunk and emit the
                    // rest as text.
                    let suffix_len = longest_suffix_prefix(&self.think_pending, "<think>");
                    let emit_len = self.think_pending.len() - suffix_len;
                    if emit_len > 0 {
                        // Use mem::take + drain to avoid allocating a new String.
                        // When suffix_len == 0 (99%+ of deltas), we take the
                        // entire buffer, drain all of it into `to_emit`, and
                        // put back an empty String — no heap allocation.
                        let mut taken = std::mem::take(&mut self.think_pending);
                        let to_emit: String = taken.drain(..emit_len).collect();
                        self.think_pending = taken; // holds the suffix (if any)
                        if self.reasoning_id.is_some() {
                            out.extend(self.stop_reasoning());
                        }
                        out.extend(self.emit_text_delta(idx, &to_emit));
                    }
                    return out;
                }
            }

            // in_think_tag is true: looking for closing tag "</think>"
            if let Some(end) = self.think_pending.find("</think>") {
                // Use mem::take + drain to avoid allocating a new String.
                let mut taken = std::mem::take(&mut self.think_pending);
                let think_content: String = taken.drain(..end).collect();
                taken.drain(..8); // skip the closing tag
                self.think_pending = taken;
                if !think_content.is_empty() {
                    out.extend(self.handle_reasoning_content(idx, &think_content));
                }
                self.in_think_tag = false;
                // Trim leading newlines that typically follow </think>.
                let trim_len = self.think_pending.len() - self.think_pending.trim_start_matches('\n').len();
                if trim_len > 0 {
                    self.think_pending.drain(..trim_len);
                }
                // Continue the loop to process the remainder as text.
                continue;
            } else {
                // No complete closing tag found. Check whether the buffer ends
                // with a prefix of "</think>" (e.g. "</thi"). Hold that suffix
                // back and emit the rest as reasoning content.
                let suffix_len = longest_suffix_prefix(&self.think_pending, "</think>");
                let emit_len = self.think_pending.len() - suffix_len;
                if emit_len > 0 {
                    // Use mem::take + drain to avoid allocating a new String.
                    let mut taken = std::mem::take(&mut self.think_pending);
                    let to_emit: String = taken.drain(..emit_len).collect();
                    self.think_pending = taken;
                    out.extend(self.handle_reasoning_content(idx, &to_emit));
                }
                return out;
            }
        }
    }

   fn emit_text_delta(&mut self, idx: usize, content: &str) -> Vec<String> {
       let mut out = Vec::new();

       // Single entry() call to get-or-insert the MessageState, then extract
       // the values we need so no further HashMap lookups are required in the
       // hot path. Previously this did 5+ separate lookups per text delta.
       let (msg_output_index, item_id, need_item_added, need_content_added) = {
           if !self.msg_states.contains_key(&idx) {
               let ix = self.alloc_output_index();
               let item_id = format!("msg_{}_{}", self.response_id, idx);
               self.msg_states.insert(idx, MessageState {
                   text_buf: String::new(),
                   output_ix: ix,
                   item_added: true,
                   content_added: true,
                   item_done: false,
                   item_id: item_id.clone(),
               });
               (ix, item_id, true, true)
           } else {
               // Single lookup via get_mut to read values and set flags.
               // We need to read output_ix, item_id, item_added, content_added.
               // Then set item_added=true and content_added=true after emitting.
               // But we can't hold &mut across emit calls, so extract first.
               let ms = self.msg_states.get_mut(&idx).unwrap();
               let ix = ms.output_ix;
               let iid = ms.item_id.clone();
               let need_item = !ms.item_added;
               let need_content = !ms.content_added;
               ms.item_added = true;
               ms.content_added = true;
               (ix, iid, need_item, need_content)
           }
       };

        if need_item_added {
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
           out.push(self.emit("response.output_item.added", item));
       }

        if need_content_added {
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
           out.push(self.emit("response.content_part.added", part));
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
       out.push(self.emit("response.output_text.delta", delta));

        self.msg_states.get_mut(&idx).unwrap().text_buf.push_str(content);

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
            out.push(self.emit("response.output_item.added", item));

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
            out.push(self.emit("response.reasoning_summary_part.added", part));
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
        out.push(self.emit("response.reasoning_summary_text.delta", delta));

        self.reasoning_buf.push_str(rc);

        out
    }

   fn stop_reasoning(&mut self) -> Vec<String> {
       let mut out = Vec::new();
       if let Some(rid) = self.reasoning_id.take() {
           let rix = self.reasoning_index.take().unwrap();

           // Take ownership of the buffer instead of cloning it 3×.
           // After this point reasoning_buf is empty, so the explicit
           // .clear() below is no longer needed.
           let full_reasoning = std::mem::take(&mut self.reasoning_buf);

           let done = json!({
               "type": "response.reasoning_summary_text.done",
               "sequence_number": self.next_seq(),
               "item_id": rid,
               "output_index": rix,
               "summary_index": 0,
               "text": full_reasoning.clone(),
           });
           out.push(self.emit("response.reasoning_summary_text.done", done));

           let part_done = json!({
               "type": "response.reasoning_summary_part.done",
               "sequence_number": self.next_seq(),
               "item_id": rid,
               "output_index": rix,
               "summary_index": 0,
               "part": {
                   "type": "summary_text",
                   "text": full_reasoning.clone(),
               }
           });
           out.push(self.emit("response.reasoning_summary_part.done", part_done));

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
                       "text": full_reasoning,
                   }],
               }
           });
           out.push(self.emit("response.output_item.done", item_done));

       }
       out
   }

    fn handle_tool_calls(&mut self, idx: usize, tcs: &[Value]) -> Vec<String> {
        let mut out = Vec::new();

        if self.reasoning_id.is_some() {
            out.extend(self.stop_reasoning());
        }

        if self.msg_states.get(&idx).map(|s| s.item_added && !s.item_done).unwrap_or(false) {
            out.extend(self.close_message(idx));
        }

        for tc in tcs {
            let tool_idx = tc.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
            let key = Self::tool_state_key(idx, tool_idx);
            let new_call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let name_chunk = tc.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()).unwrap_or("").to_string();

            if !name_chunk.is_empty() {
                let fs = self.func_states.entry(key.clone()).or_insert_with(|| FunctionCallState {
                    args_buf: String::new(),
                    name: String::new(),
                    call_id: String::new(),
                    item_id: String::new(),
                    output_ix: 0,
                    item_done: false,
                    item_added: false,
                });
                fs.name = name_chunk;
            }

            let should_emit_item = self.func_states.get(&key).map(|s| !s.item_added).unwrap_or(true)
                && !new_call_id.is_empty();

           if should_emit_item {
                // Set up the state entry first, then extract the values we
                // need so no &mut borrow is held across emit/next_seq calls.
                let oix = self.alloc_output_index();
                let name = self.func_states.get(&key).map(|s| s.name.clone()).unwrap_or_default();
                let precomputed_item_id = format!("fc_{}", new_call_id);
                {
                    let fs = self.func_states.entry(key.clone()).or_insert_with(|| FunctionCallState {
                        args_buf: String::new(),
                        name: String::new(),
                        call_id: String::new(),
                        item_id: String::new(),
                        output_ix: 0,
                        item_done: false,
                        item_added: false,
                    });
                    fs.call_id = new_call_id.clone();
                    fs.item_id = precomputed_item_id.clone();
                    fs.output_ix = oix;
                    fs.item_added = true;
                }

                let is_custom = is_custom_tool(&name);
                let item = if is_custom {
                    json!({
                        "type": "response.output_item.added",
                        "sequence_number": self.next_seq(),
                        "output_index": oix,
                        "item": {
                            "id": precomputed_item_id,
                            "type": "custom_tool_call",
                            "status": "in_progress",
                            "input": "",
                            "call_id": new_call_id,
                            "name": name,
                        }
                    })
               } else {
                   json!({
                       "type": "response.output_item.added",
                       "sequence_number": self.next_seq(),
                        "output_index": oix,
                       "item": {
                           "id": precomputed_item_id,
                           "type": "function_call",
                           "status": "in_progress",
                           "arguments": "",
                           "call_id": new_call_id,
                           "name": name,
                       }
                   })
                };
                out.push(self.emit("response.output_item.added", item));
            }

            if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")).and_then(|v| v.as_str()) {
               if !args.is_empty() {
                    let (item_id, oix, is_custom) = {
                       let fs = self.func_states.entry(key.clone()).or_insert_with(|| FunctionCallState {
                           args_buf: String::new(),
                           name: String::new(),
                           call_id: new_call_id.clone(),
                           item_id: format!("fc_{}", new_call_id),
                           output_ix: 0,
                           item_done: false,
                           item_added: false,
                       });
                       fs.args_buf.push_str(args);
                       let iid = if fs.item_id.is_empty() { format!("fc_{}", new_call_id) } else { fs.item_id.clone() };
                        (iid, fs.output_ix, is_custom_tool(&fs.name))
                   };

                    if !item_id.is_empty() && !is_custom {
                        // Regular function calls: stream the raw argument
                        // fragments as deltas. Codex's function_call handler
                        // accumulates these into a JSON arguments string.
                        let event_name = "response.function_call_arguments.delta";
                        let delta = json!({
                            "type": event_name,
                            "sequence_number": self.next_seq(),
                            "item_id": item_id,
                            "output_index": oix,
                            "delta": args,
                        });
                        out.push(self.emit(event_name, delta));
                    }
                    // For custom tools (apply_patch): do NOT emit streaming
                    // deltas here. The model sends arguments as JSON fragments
                    // like `{"input": "*** Begin Patch\n..."}` split across
                    // many chunks. Emitting those raw JSON fragments as
                    // `custom_tool_call_input.delta` events feeds garbage
                    // into Codex's StreamingPatchParser, which expects patch
                    // text (starting with `*** Begin Patch`), not JSON. This
                    // corrupts the parser's state and causes the patch to be
                    // rejected as malformed.
                    //
                    // Instead, we buffer the fragments and emit the extracted
                    // patch text as a single clean delta in
                    // `close_function_call`, followed by the `.done` event.
                }
            }
        }

        out
    }

    fn close_message(&mut self, idx: usize) -> Vec<String> {
        let mut out = Vec::new();
        let item_id = format!("msg_{}_{}", self.response_id, idx);
        let (msg_output_index, full_text) = {
            let ms = self.msg_states.get_mut(&idx).unwrap();
            // Clone the text — build_completed_event also needs it later to
            // construct the response.output array. Using mem::take here would
            // empty the buffer and break response.completed.
            (ms.output_ix, ms.text_buf.clone())
        };

        let done = json!({
            "type": "response.output_text.done",
            "sequence_number": self.next_seq(),
            "item_id": item_id,
            "output_index": msg_output_index,
            "content_index": 0,
            "text": full_text,
            "logprobs": [],
        });
        out.push(self.emit("response.output_text.done", done));

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
        out.push(self.emit("response.content_part.done", part_done));

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
        out.push(self.emit("response.output_item.done", item_done));

        self.msg_states.get_mut(&idx).unwrap().item_done = true;
        out
    }

    fn handle_finish(&mut self, idx: usize) -> Vec<String> {
        let mut out = Vec::new();

        if self.reasoning_id.is_some() {
            out.extend(self.stop_reasoning());
        }

        if self.msg_states.get(&idx).map(|s| s.item_added && !s.item_done).unwrap_or(false) {
            out.extend(self.close_message(idx));
        }

        let prefix = format!("{}:", idx);
        let keys_to_close: Vec<String> = self.func_states.iter()
            .filter(|(k, s)| k.starts_with(&prefix) && !s.item_done)
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_close {
            out.extend(self.close_function_call(&key));
        }

        out
    }

    fn close_function_call(&mut self, key: &str) -> Vec<String> {
        let mut out = Vec::new();
        let (item_id, call_id, oix, args, name) = {
            let fs = match self.func_states.get_mut(key) {
                Some(fs) => fs,
                None => return out,
            };
            if fs.call_id.is_empty() {
                return out;
            }
            // Use mem::take to avoid cloning the potentially large args buffer.
            // This is the final operation on this state — the buffer is no
            // longer needed after close.
            (fs.item_id.clone(), fs.call_id.clone(), fs.output_ix, fs.args_buf.clone(), fs.name.clone())
        };

        let is_custom = is_custom_tool(&name);

        if is_custom {
            // apply_patch uses custom_tool_call format: input field, not arguments
            let input = extract_custom_tool_input(&args);

            // Emit the complete extracted input as a single delta so Codex's
            // streaming patch parser (StreamingPatchParser) receives the full
            // patch text in one clean chunk. We suppressed streaming deltas in
            // `handle_tool_calls` (they would have been raw JSON fragments like
            // `{"in` / `put": "*** ` / `Begin Patch...` which the parser cannot
            // understand — it expects patch text starting with `*** Begin Patch`).
            // This single delta provides the complete, clean patch text.
            let final_delta = json!({
                "type": "response.custom_tool_call_input.delta",
                "sequence_number": self.next_seq(),
                "item_id": item_id,
                "call_id": call_id,
                "output_index": oix,
                "delta": input,
            });
            out.push(self.emit("response.custom_tool_call_input.delta", final_delta));

            let done = json!({
                "type": "response.custom_tool_call_input.done",
                "sequence_number": self.next_seq(),
                "item_id": item_id,
                "output_index": oix,
                "input": input,
            });
            out.push(self.emit("response.custom_tool_call_input.done", done));

            let item_done = json!({
                "type": "response.output_item.done",
                "sequence_number": self.next_seq(),
                "output_index": oix,
                "item": {
                    "id": item_id,
                    "type": "custom_tool_call",
                    "status": "completed",
                    "input": input,
                    "call_id": call_id,
                    "name": name,
                }
            });
            out.push(self.emit("response.output_item.done", item_done));
        } else {
            let done = json!({
                "type": "response.function_call_arguments.done",
                "sequence_number": self.next_seq(),
                "item_id": item_id,
                "output_index": oix,
                "arguments": args,
            });
            out.push(self.emit("response.function_call_arguments.done", done));

            let item_done = json!({
                "type": "response.output_item.done",
                "sequence_number": self.next_seq(),
                "output_index": oix,
                "item": {
                    "id": item_id,
                    "type": "function_call",
                    "status": "completed",
                    "arguments": args,
                    "call_id": call_id,
                    "name": name,
                }
            });
            out.push(self.emit("response.output_item.done", item_done));
        }

        self.func_states.get_mut(key).unwrap().item_done = true;
        out
    }

    fn flush_and_complete(&mut self) -> Vec<String> {
        let mut out = Vec::new();

        // Flush any pending think-tag buffer content. If we're not inside a
        // think tag, the pending content is just text that was held back
        // because it might have been a partial tag prefix. If we ARE inside a
        // think tag that was never closed, treat the remaining content as
        // reasoning.
        if !self.think_pending.is_empty() {
            // Use mem::take to avoid cloning the buffer — we clear it
            // immediately after, so taking ownership is safe and free.
            let pending = std::mem::take(&mut self.think_pending);
            if self.in_think_tag {
                // Unclosed think tag: treat remaining content as reasoning.
                out.extend(self.handle_reasoning_content(0, &pending));
            } else {
                // Not in a think tag: this is text that was held back
                // because it might have been a partial tag prefix.
                if self.reasoning_id.is_some() {
                    out.extend(self.stop_reasoning());
                }
                out.extend(self.emit_text_delta(0, &pending));
            }
        }

        if self.reasoning_id.is_some() {
            out.extend(self.stop_reasoning());
        }

        let open_msgs: Vec<usize> = self.msg_states.iter()
            .filter(|(_, s)| s.item_added && !s.item_done)
            .map(|(k, _)| *k)
            .collect();
        for idx in open_msgs {
            out.extend(self.close_message(idx));
        }

        let open_funcs: Vec<String> = self.func_states.iter()
            .filter(|(_, s)| s.item_added && !s.item_done)
            .map(|(k, _)| k.clone())
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
        // end_turn: false if the model emitted tool calls (codex should execute
        // them and continue the turn); true if only text was produced (turn is
        // done). Without this field, codex defaults to None and treats the
        // turn as complete even when tool calls were emitted.
        let end_turn = self.func_states.is_empty();

        let mut response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created,
            "status": "completed",
            "background": false,
            "error": null,
            "end_turn": end_turn,
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

        for (idx, ms) in &self.msg_states {
            let item = json!({
                "id": format!("msg_{}_{}", self.response_id, idx),
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": ms.text_buf,
                }],
                "role": "assistant",
            });
            output.push(item);
        }

        for (_, fs) in &self.func_states {
            let is_custom = is_custom_tool(&fs.name);
            let item = if is_custom {
                let input = extract_custom_tool_input(&fs.args_buf);
                json!({
                    "id": format!("fc_{}", fs.call_id),
                    "type": "custom_tool_call",
                    "status": "completed",
                    "input": input,
                    "call_id": fs.call_id,
                    "name": fs.name,
                })
            } else {
                json!({
                    "id": format!("fc_{}", fs.call_id),
                    "type": "function_call",
                    "status": "completed",
                    "arguments": fs.args_buf,
                    "call_id": fs.call_id,
                    "name": fs.name,
                })
            };
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

        self.emit("response.completed", completed)
    }

    /// Return the output items from this response — the same items that
    /// `build_completed_event` puts in the `output` array. Used by the
    /// WebSocket handler to append the assistant's response (messages,
    /// function calls, reasoning) to the cached request's input array,
    /// so that on the next incremental request (previous_response_id),
    /// the merged conversation includes the assistant's previous output.
    ///
    /// Without this, the LLM sees only the original user messages + new
    /// tool results, but NOT the assistant's text/tool calls from the
    /// previous turn — causing it to re-do the same work (infinite loop).
    pub fn get_output_items(&self) -> Vec<Value> {
        let mut output = Vec::new();

        // Reasoning item (if any)
        if let Some(rid) = &self.reasoning_id {
            let full_reasoning = &self.reasoning_buf;
            let item = json!({
                "id": rid,
                "type": "reasoning",
                "status": "completed",
                "summary": [{
                    "type": "summary_text",
                    "text": full_reasoning,
                }],
            });
            output.push(item);
        }

        // Message items
        for (idx, ms) in &self.msg_states {
            let item = json!({
                "id": format!("msg_{}_{}", self.response_id, idx),
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": ms.text_buf,
                }],
                "role": "assistant",
            });
            output.push(item);
        }

        // Function call items
        for (_, fs) in &self.func_states {
            let is_custom = is_custom_tool(&fs.name);
            let item = if is_custom {
                let input = extract_custom_tool_input(&fs.args_buf);
                json!({
                    "id": format!("fc_{}", fs.call_id),
                    "type": "custom_tool_call",
                    "status": "completed",
                    "input": input,
                    "call_id": fs.call_id,
                    "name": fs.name,
                })
            } else {
                json!({
                    "id": format!("fc_{}", fs.call_id),
                    "type": "function_call",
                    "status": "completed",
                    "arguments": fs.args_buf,
                    "call_id": fs.call_id,
                    "name": fs.name,
                })
            };
            output.push(item);
        }

        output
    }

    /// Emit a translated event. Takes ownership of the payload Value (no clone).
    /// In SSE mode, wraps it in `event: <event>\ndata: <json>\n\n`.
    /// In raw JSON mode, returns just the JSON string.
    fn emit(&mut self, event: &str, payload: Value) -> String {
        // The `type` field is already included in every json!() macro call
        // above (e.g. "type": "response.output_text.delta"). Previously this
        // function also did obj.insert("type".to_string(), json!(event))
        // which allocated a "type".to_string() key on every single event and
        // overwrote the value that was already set. Since the json! macro
        // already sets the correct type, we no longer need to insert it here.
        // We DO still need to handle the case where a payload doesn't have
        // a type field (defensive), but that never happens in practice.
        if self.raw_json {
            payload.to_string()
        } else {
            format!("event: {}\ndata: {}\n\n", event, payload)
        }
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

    // end_turn: false if the model emitted tool calls, true otherwise.
    let has_tool_calls = raw.get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("tool_calls"))
        .is_some();

    let mut resp = json!({
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "background": false,
        "error": null,
        "end_turn": !has_tool_calls,
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
                        let is_custom = is_custom_tool(&name);
                        let item = if is_custom {
                            let input = extract_custom_tool_input(&args);
                            json!({
                                "id": format!("fc_{}", call_id),
                                "type": "custom_tool_call",
                                "status": "completed",
                                "input": input,
                                "call_id": call_id,
                                "name": name,
                            })
                        } else {
                            json!({
                                "id": format!("fc_{}", call_id),
                                "type": "function_call",
                                "status": "completed",
                                "arguments": args,
                                "call_id": call_id,
                                "name": name,
                            })
                        };
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
