use serde_json::{json, Value};
use std::collections::HashMap;

use crate::logger;

/// Split a namespaced tool name (e.g. "multi_agent_v1__spawn_agent") into
/// (namespace, name) = ("multi_agent_v1", "spawn_agent").
/// Returns (None, full_name) if there is no "__" separator.
fn split_namespaced_tool_name(full_name: &str) -> (Option<String>, String) {
    if let Some(idx) = full_name.find("__") {
        let ns = &full_name[..idx];
        let name = &full_name[idx + 2..];
        if !ns.is_empty() && !name.is_empty() {
            return (Some(ns.to_string()), name.to_string());
        }
    }
    (None, full_name.to_string())
}


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
/// `ApplyPatchHandler` / `CodeModeExecuteHandler` which only accept
/// `ToolPayload::Custom`.
///
/// NOTE: `wait` is a regular function tool (`ToolSpec::Function`), NOT a
/// freeform/custom tool. Its handler `CodeModeWaitHandler` expects
/// `ToolPayload::Function { arguments }`, so it must be emitted as a
/// standard `function_call` item, not a `custom_tool_call`.
fn is_custom_tool(name: &str) -> bool {
    // apply_patch: freeform tool, dispatched via ApplyPatchHandler
    // exec: Code Mode freeform tool, dispatched via CodeModeExecuteHandler
    //   (expects ToolPayload::Custom { input })
    // wait: regular function tool (ToolSpec::Function), dispatched via
    //   CodeModeWaitHandler (expects ToolPayload::Function { arguments })
    //   — NOT custom, so excluded here.
    name == "apply_patch" || name == "exec"
}

/// Extract the `input` field from a JSON arguments string.
///
/// The model sends apply_patch/exec/wait arguments as JSON: {"input": "..."}
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
    // Once content/tool_calls have closed the reasoning block for this turn,
    // GLM-style interleaving (reasoning -> content -> reasoning -> content)
    // must not reopen a new reasoning item — that produces multiple
    // fragmented reasoning blocks (often trailing stragglers like ".").
    // After the first close, any further reasoning_content is routed to the
    // message stream as visible text instead.
    reasoning_closed_for_turn: bool,

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
    request_json: Option<std::sync::Arc<Value>>,

    // Upstream error detection: when the vLLM backend returns an in-stream
    // error (e.g. `{"message":"upstream error","code":503}`) it arrives as a
    // `data:` line with none of the expected keys (choices/usage/id/model).
    // Without this flag, process_data silently drops it and flush_and_complete
    // emits a successful response.completed — causing codex to treat the turn
    // as a completed-but-empty response (end_turn, 0 tokens, usage 0%).
    upstream_error: Option<(u16, String)>,
}

impl ResponseTranslator {
    pub fn new(response_id: &str, request_json: Option<Value>) -> Self {
        Self::new_impl(response_id, request_json.map(std::sync::Arc::new), false)
    }

    /// Create a translator that emits raw JSON strings (no SSE envelope),
    /// suitable for WebSocket transport where each frame is a single JSON object.
    pub fn new_raw_json(response_id: &str, request_json: Option<Value>) -> Self {
        Self::new_impl(response_id, request_json.map(std::sync::Arc::new), true)
    }

    /// Create a translator that emits raw JSON strings, accepting an Arc<Value>
    /// to avoid cloning the full request JSON (which can be large with many
    /// input items on long sessions).
    pub fn new_raw_json_arc(
        response_id: &str,
        request_json: Option<std::sync::Arc<Value>>,
    ) -> Self {
        Self::new_impl(response_id, request_json, true)
    }

    fn new_impl(
        response_id: &str,
        request_json: Option<std::sync::Arc<Value>>,
        raw_json: bool,
    ) -> Self {
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
            reasoning_closed_for_turn: false,
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
            upstream_error: None,
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

        // Basic schema validation: upstream SSE chunks must be JSON objects
        // with at least one expected top-level key. This prevents forwarding
        // malformed or unexpected data to the client (CWE-502 mitigation).
        if !parsed.is_object() {
            return vec![];
        }

        // Upstream error detection: vLLM / TGI backends sometimes return
        // in-stream error objects like `{"message":"upstream error","code":503}`
        // or `{"error":{"message":"...","code":429}}`. These arrive as `data:`
        // lines with no `choices`/`usage`/`id`/`model` keys. Without this
        // check, they'd be silently skipped and the subsequent [DONE] would
        // emit response.completed — making codex believe the turn succeeded
        // with empty output (end_turn, 0 tokens, usage 0%).
        //
        // When detected, we stash the error and return empty events. The
        // caller (main.rs) checks has_upstream_error() after the stream ends
        // and sends a proper error event instead of response.completed.
        if let Some((code, msg)) = extract_upstream_error(&parsed) {
            logger::error(
                "translate_response",
                &format!(
                    "Upstream in-stream error detected (code={}): {}",
                    code,
                    &msg[..msg.len().min(300)]
                ),
            );
            self.upstream_error = Some((code, msg));
            return vec![];
        }

        let has_expected_key = parsed.get("choices").is_some()
            || parsed.get("usage").is_some()
            || parsed.get("id").is_some()
            || parsed.get("model").is_some();
        if !has_expected_key {
            logger::warn(
                "translate_response",
                &format!(
                    "Unexpected upstream SSE data (no recognized keys), skipping: {}",
                    &data[..data.len().min(200)]
                ),
            );
            return vec![];
        }

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
                let p = usage
                    .get("prompt_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let c = usage
                    .get("completion_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let t = usage
                    .get("total_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                // vLLM streams usage in two chunks:
                //   1. Real usage (choices: [], prompt_tokens > 0) — after finish_reason
                //   2. Zero-usage cleanup chunk (choices: [{delta:{}}], all zeros) — on connection close
                // Never overwrite real usage with zeros — the cleanup chunk
                // would corrupt the response.completed event sent to codex,
                // causing "Context 98% left" when it should be ~53%.
                logger::debug(
                    "translate_response",
                    &format!(
                        "[usage] raw vLLM usage object: {}",
                        serde_json::to_string(usage).unwrap_or_default()
                    ),
                );
                if p > 0 || c > 0 || t > 0 || !self.usage_seen {
                    self.usage_seen = true;
                    self.prompt_tokens = p;
                    self.completion_tokens = c;
                    self.total_tokens = t;
                    if let Some(pd) = usage.get("prompt_tokens_details") {
                        self.cached_tokens = pd
                            .get("cached_tokens")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        logger::debug("translate_response", &format!(
                            "[usage] KV-cache hit: cached_tokens={} / prompt_tokens={} ({:.1}% hit rate)",
                            self.cached_tokens, p,
                            if p > 0 { self.cached_tokens as f64 / p as f64 * 100.0 } else { 0.0 }
                       ));
                    }
                    if self.cached_tokens == 0 && p > 0 {
                        logger::debug("translate_response", &format!(
                            "[usage] KV-cache MISS: no prompt_tokens_details.cached_tokens in response (prompt_tokens={})", p
                       ));
                    }
                    if let Some(od) = usage.get("output_tokens_details") {
                        self.reasoning_tokens = od
                            .get("reasoning_tokens")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                    }
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

                let finish_reason = choice
                    .get("finish_reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !finish_reason.is_empty() {
                    self.completion_pending = true;
                }


                // reasoning_content — some vLLM backends (DeepSeek, Qwen3) use
                // delta.reasoning_content for reasoning tokens.
                // reasoning — GLM-5.2 vLLM uses delta.reasoning instead.
                // Check both keys so reasoning tokens are captured regardless
                // of which field name the upstream uses.
                // MUST be processed BEFORE content — GLM-5.2 sometimes sends
                // both content and reasoning in the same delta. If content is
                // processed first, it calls stop_reasoning() which closes the
                // current reasoning block; then the reasoning token creates a
                // NEW reasoning block, causing interleaved
                // message/reasoning/message that orphans the first content
                // chunk (e.g. a stray "There").
                let rc = delta
                    .get("reasoning_content")
                    .and_then(|v| v.as_str())
                    .or_else(|| delta.get("reasoning").and_then(|v| v.as_str()));
                if let Some(rc) = rc {
                    if !rc.is_empty() {
                        logger::debug("translate_response", &format!(
                            "[delta] idx={} reasoning_content (len={}): {:?}",
                            idx, rc.len(), &rc[..rc.len().min(200)]
                        ));
                        let events = self.handle_reasoning_content(idx, rc);
                        out.extend(events);
                    }
                }

                // Content (text) — processed AFTER reasoning so that reasoning
                // tokens in the same delta are appended to the current block
                // before content closes it via stop_reasoning().
                if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                    if !content.is_empty() {
                        logger::debug("translate_response", &format!(
                            "[delta] idx={} content (len={}): {:?}",
                            idx, content.len(), &content[..content.len().min(200)]
                        ));
                        let events = self.handle_content(idx, content);
                        out.extend(events);
                    }
                }

                // Log all delta keys for debugging
                if let Some(obj) = delta.as_object() {
                    let keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
                    logger::debug("translate_response", &format!(
                        "[delta] idx={} delta_keys={:?}", idx, keys
                    ));
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

        // Append the new chunk to the pending buffer for tag detection
        // across chunk boundaries.
        self.think_pending.push_str(content);

        loop {
            if !self.in_think_tag {
                // Looking for opening think tag. Models emit two variants:
                //   Kimi/GLM:   thinking (7 chars)
                //   DeepSeek:  thinking (10 chars)
                let (open_pos, open_len) = match (
                    self.think_pending.find("\x3c\x74\x68\x69\x6e\x6b\x3e"),
                    self.think_pending
                        .find("\x3c\x74\x68\x69\x6e\x6b\x69\x6e\x67\x3e"),
                ) {
                    (Some(s), Some(l)) if s <= l => (Some(s), 7),
                    (Some(_), Some(l)) => (Some(l), 10),
                    (Some(s), None) => (Some(s), 7),
                    (None, Some(l)) => (Some(l), 10),
                    (None, None) => (None, 0),
                };
                if let Some(start) = open_pos {
                    // Emit any text before the opening tag as visible text.
                    if start > 0 {
                        let mut taken = std::mem::take(&mut self.think_pending);
                        let before: String = taken.drain(..start).collect();
                        taken.drain(..open_len); // skip the tag itself
                        self.think_pending = taken;
                        if !before.is_empty() {
                            out.extend(self.emit_text_delta(idx, &before));
                        }
                    } else {
                        self.think_pending.drain(..open_len);
                    }
                    self.in_think_tag = true;
                    continue;
                } else {
                    // No complete opening tag. Hold back a possible partial
                    // tag prefix and emit the rest as text.
                    let suffix_len =
                        longest_suffix_prefix(&self.think_pending, "\x3c\x74\x68\x69\x6e\x6b\x3e")
                            .max(longest_suffix_prefix(
                                &self.think_pending,
                                "\x3c\x74\x68\x69\x6e\x6b\x69\x6e\x67\x3e",
                            ));
                    let emit_len = self.think_pending.len() - suffix_len;
                    if emit_len > 0 {
                        let mut taken = std::mem::take(&mut self.think_pending);
                        let to_emit: String = taken.drain(..emit_len).collect();
                        self.think_pending = taken;
                        out.extend(self.emit_text_delta(idx, &to_emit));
                    }
                    return out;
                }
            }

            // in_think_tag is true: looking for closing think tag.
            //   Kimi/GLM:   response (8 chars)
            //   DeepSeek:  response (11 chars)
            let (close_pos, close_len) = match (
                self.think_pending.find("\x3c\x2f\x74\x68\x69\x6e\x6b\x3e"),
                self.think_pending
                    .find("\x3c\x2f\x74\x68\x69\x6e\x6b\x69\x6e\x67\x3e"),
            ) {
                (Some(s), Some(l)) if s <= l => (Some(s), 8),
                (Some(_), Some(l)) => (Some(l), 11),
                (Some(s), None) => (Some(s), 8),
                (None, Some(l)) => (Some(l), 11),
                (None, None) => (None, 0),
            };
            if let Some(end) = close_pos {
                // Content between tags: emit as VISIBLE TEXT, not reasoning.
                // The think tags are just markers — strip them and show the
                // content as normal output text. Only API-native reasoning
                // (via the reasoning_content delta field) should produce
                // thinking blocks.
                let mut taken = std::mem::take(&mut self.think_pending);
                let think_content: String = taken.drain(..end).collect();
                taken.drain(..close_len); // skip the closing tag
                self.think_pending = taken;
                if !think_content.is_empty() {
                    out.extend(self.emit_text_delta(idx, &think_content));
                }
                self.in_think_tag = false;
                // Trim leading newlines that typically follow the closing tag.
                let trim_len =
                    self.think_pending.len() - self.think_pending.trim_start_matches('\n').len();
                if trim_len > 0 {
                    self.think_pending.drain(..trim_len);
                }
                continue;
            } else {
                // No complete closing tag. Hold back a possible partial tag
                // prefix and emit the rest as visible text.
                let suffix_len =
                    longest_suffix_prefix(&self.think_pending, "\x3c\x2f\x74\x68\x69\x6e\x6b\x3e")
                        .max(longest_suffix_prefix(
                            &self.think_pending,
                            "\x3c\x2f\x74\x68\x69\x6e\x6b\x69\x6e\x67\x3e",
                        ));
                let emit_len = self.think_pending.len() - suffix_len;
                if emit_len > 0 {
                    let mut taken = std::mem::take(&mut self.think_pending);
                    let to_emit: String = taken.drain(..emit_len).collect();
                    self.think_pending = taken;
                    out.extend(self.emit_text_delta(idx, &to_emit));
                }
                return out;
            }
        }
    }

    /// Strip orphaned think tags from text content. These can appear when
    /// old cached conversation history (generated before the dual-tag fix)
    /// contains dangling think closing tags that the parser didn't catch.
    /// This is a safety net  the tag parser should handle all cases, but
    /// cached history from before the fix may have leaked tags.
    fn strip_orphan_think_tags(content: &str) -> String {
        // Only strip if the ENTIRE content (after trimming whitespace) is
        // just a standalone think tag. This avoids corrupting content where
        // the model legitimately mentions tags in code or text.
        let trimmed = content.trim();
        if trimmed == "<think>"
            || trimmed == "</think>"
            || trimmed == "<thinking>"
            || trimmed == "</thinking>"
        {
            return String::new();
        }
        content.to_string()
    }

    fn emit_text_delta(&mut self, idx: usize, content: &str) -> Vec<String> {
        let mut out = Vec::new();
        // Safety net: strip any orphaned think tags that leaked from old
        // cached conversation history (generated before the dual-tag fix).
        let content = Self::strip_orphan_think_tags(content);
        if content.is_empty() {
            return out;
        }

        // If an API-native reasoning block (from `delta.reasoning_content`)
        // is currently open, close it now before emitting the first text
        // delta of the response. Without this the reasoning summary item
        // is left in `in_progress` state, and codex never sees the
        // `response.reasoning_summary_text.done` event.
        if self.reasoning_id.is_some() {
            out.extend(self.stop_reasoning());
        }

        // Single entry() call to get-or-insert the MessageState, then extract
        // the values we need so no further HashMap lookups are required in the
        // hot path. Previously this did 5+ separate lookups per text delta.
        let (msg_output_index, item_id, need_item_added, need_content_added) = {
            if !self.msg_states.contains_key(&idx) {
                let ix = self.alloc_output_index();
                let item_id = format!("msg_{}_{}", self.response_id, idx);
                self.msg_states.insert(
                    idx,
                    MessageState {
                        text_buf: String::new(),
                        output_ix: ix,
                        item_added: true,
                        content_added: true,
                        item_done: false,
                        item_id: item_id.clone(),
                    },
                );
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

        self.msg_states
            .get_mut(&idx)
            .unwrap()
            .text_buf
            .push_str(&content);

        out
    }

    fn handle_reasoning_content(&mut self, idx: usize, rc: &str) -> Vec<String> {
        // If the reasoning block for this turn has already been closed by
        // an earlier content/tool_call delta, GLM-style interleaving would
        // otherwise reopen a brand-new reasoning item here — producing
        // multiple fragmented reasoning blocks (often a trailing "."
        // straggler). Once closed, route further reasoning tokens into the
        // message stream as visible text instead.
        if self.reasoning_closed_for_turn {
            return self.emit_text_delta(idx, rc);
        }

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

        let rix = self.reasoning_index.unwrap();
        let seq = self.next_seq();
        let rid = self.reasoning_id.as_ref().unwrap();

        let delta = json!({
            "type": "response.reasoning_summary_text.delta",
             "sequence_number": seq,
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
            // Latch: once the reasoning block is closed for this turn, any
            // further reasoning_content deltas (GLM interleaving) are routed
            // to the content stream instead of reopening a new block.
            self.reasoning_closed_for_turn = true;
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
                 "text": &full_reasoning,
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
                     "text": &full_reasoning,
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

        if self
            .msg_states
            .get(&idx)
            .map(|s| s.item_added && !s.item_done)
            .unwrap_or(false)
        {
            out.extend(self.close_message(idx));
        }

        for tc in tcs {
            let tool_idx = tc.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
            let key = Self::tool_state_key(idx, tool_idx);
            let new_call_id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name_chunk = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            if !name_chunk.is_empty() {
                let fs = self
                    .func_states
                    .entry(key.clone())
                    .or_insert_with(|| FunctionCallState {
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

            let should_emit_item = self
                .func_states
                .get(&key)
                .map(|s| !s.item_added)
                .unwrap_or(true)
                && !new_call_id.is_empty();

            if should_emit_item {
                // Set up the state entry first, then extract the values we
                // need so no &mut borrow is held across emit/next_seq calls.
                let oix = self.alloc_output_index();
                let name = self
                    .func_states
                    .get(&key)
                    .map(|s| s.name.clone())
                    .unwrap_or_default();
                let precomputed_item_id = format!("fc_{}", new_call_id);
                {
                    let fs =
                        self.func_states
                            .entry(key.clone())
                            .or_insert_with(|| FunctionCallState {
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
                let (ns, short_name) = split_namespaced_tool_name(&name);
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
                            "name": short_name,
                            "namespace": ns,
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
                            "name": short_name,
                            "namespace": ns,
                        }
                    })
                };
                out.push(self.emit("response.output_item.added", item));
            }

            if let Some(args) = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
            {
                if !args.is_empty() {
                    let (item_id, oix, is_custom) = {
                        let fs = self.func_states.entry(key.clone()).or_insert_with(|| {
                            FunctionCallState {
                                args_buf: String::new(),
                                name: String::new(),
                                call_id: new_call_id.clone(),
                                item_id: format!("fc_{}", new_call_id),
                                output_ix: 0,
                                item_done: false,
                                item_added: false,
                            }
                        });
                        fs.args_buf.push_str(args);
                        let iid = if fs.item_id.is_empty() {
                            format!("fc_{}", new_call_id)
                        } else {
                            fs.item_id.clone()
                        };
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

        if self
            .msg_states
            .get(&idx)
            .map(|s| s.item_added && !s.item_done)
            .unwrap_or(false)
        {
            out.extend(self.close_message(idx));
        }

        let prefix = format!("{}:", idx);
        let keys_to_close: Vec<String> = self
            .func_states
            .iter()
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
            (
                fs.item_id.clone(),
                fs.call_id.clone(),
                fs.output_ix,
                fs.args_buf.clone(),
                fs.name.clone(),
            )
        };

        let is_custom = is_custom_tool(&name);
        let (ns, short_name) = split_namespaced_tool_name(&name);

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
                    "name": short_name,
                    "namespace": ns,
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
                    "name": short_name,
                    "namespace": ns,
                }
            });
            out.push(self.emit("response.output_item.done", item_done));
        }

        self.func_states.get_mut(key).unwrap().item_done = true;
        out
    }

    fn flush_and_complete(&mut self) -> Vec<String> {
        let mut out = Vec::new();

        // Flush any pending think-tag buffer content.
        if !self.think_pending.is_empty() {
            let pending = std::mem::take(&mut self.think_pending);
            if self.in_think_tag {
                // Unclosed think tag: the model opened  thinking but never
                // emitted a closing tag. The remaining content is likely the
                // model's actual answer, not reasoning. Close the reasoning
                // block and emit the content as visible text.
                if self.reasoning_id.is_some() {
                    out.extend(self.stop_reasoning());
                }
                self.in_think_tag = false;
                out.extend(self.emit_text_delta(0, &pending));
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

        let open_msgs: Vec<usize> = self
            .msg_states
            .iter()
            .filter(|(_, s)| s.item_added && !s.item_done)
            .map(|(k, _)| *k)
            .collect();
        for idx in open_msgs {
            out.extend(self.close_message(idx));
        }

        let open_funcs: Vec<String> = self
            .func_states
            .iter()
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
            for field in [
                "instructions",
                "max_output_tokens",
                "max_tool_calls",
                "model",
                "parallel_tool_calls",
                "previous_response_id",
                "prompt_cache_key",
                "reasoning",
                "safety_identifier",
                "service_tier",
                "store",
                "temperature",
                "text",
                "tool_choice",
                "tools",
                "top_logprobs",
                "top_p",
                "truncation",
                "user",
                "metadata",
            ] {
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

        for fs in self.func_states.values() {
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
        for fs in self.func_states.values() {
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

    /// Returns the upstream in-stream error (code, message) if one was
    /// detected during `process_line`. Callers should check this after the
    /// stream ends and send an error event instead of response.completed.
    pub fn upstream_error(&self) -> Option<(u16, String)> {
        self.upstream_error.clone()
    }

    /// Returns true if the translator detected an upstream in-stream error.
    pub fn has_upstream_error(&self) -> bool {
        self.upstream_error.is_some()
    }
}

/// Detect upstream error objects embedded in SSE streams.
///
/// vLLM / TGI backends return errors in several shapes:
///   1. `{"message":"upstream error","code":503}` (flat)
///   2. `{"error":{"message":"...","code":429}}` (nested)
///   3. `{"error":"rate limit exceeded"}` (string)
///   4. `{"detail":"Service Unavailable"}` (FastAPI style)
///
/// Returns `Some((code, message))` if the parsed JSON looks like an upstream
/// error object, `None` otherwise.
/// Parse a context-length-exceeded error from the backend, returning
/// `(total_tokens, max_tokens)`.
///
/// vLLM phrases it as:
///   "This model's maximum context length is 196608 tokens. However, you
///    requested 32000 output tokens and your prompt contains at least 164609
///    input tokens, for a total of at least 196609 tokens. Please reduce ..."
///
/// This matters because Claude Code CLI recovers from an over-long prompt by
/// compacting the conversation, but only when it recognizes Anthropic's own
/// wording — it matches
/// `/prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)/i` and uses the two
/// numbers to decide how much history to drop. Reporting the backend's wording
/// verbatim leaves the client with an unrecognized error and a dead session, so
/// callers re-emit this as "prompt is too long: N tokens > M maximum".
pub fn parse_context_length_error(body: &str) -> Option<(u64, u64)> {
    let lower = body.to_lowercase();
    // Only treat this as a context-length problem when the backend says so.
    if !lower.contains("maximum context length") && !lower.contains("context length is") {
        return None;
    }

    let max_tokens = number_after(&lower, "maximum context length is ")
        .or_else(|| number_after(&lower, "context length is "))?;

    // Prefer the total the backend computed; fall back to output + input.
    let total = number_after(&lower, "for a total of at least ")
        .or_else(|| number_after(&lower, "for a total of "))
        .or_else(|| {
            let requested = number_after(&lower, "you requested ")?;
            let input = number_after(&lower, "prompt contains at least ")
                .or_else(|| number_after(&lower, "prompt contains "))?;
            Some(requested + input)
        })?;

    Some((total, max_tokens))
}

/// Read the first run of digits that follows `marker`.
fn number_after(haystack: &str, marker: &str) -> Option<u64> {
    let start = haystack.find(marker)? + marker.len();
    let digits: String = haystack[start..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

pub fn extract_upstream_error(parsed: &Value) -> Option<(u16, String)> {
    // Shape 2: {"error":{"message":"...","code":NNN}}
    if let Some(err_obj) = parsed.get("error").and_then(|e| e.as_object()) {
        let msg = err_obj.get("message").and_then(|m| m.as_str()).unwrap_or("");
        let code = err_obj.get("code").and_then(|c| c.as_u64()).unwrap_or(0) as u16;
        if !msg.is_empty() || code > 0 {
            return Some((if code == 0 { 502 } else { code }, msg.to_string()));
        }
    }

    // Shape 3: {"error":"rate limit exceeded"}
    if let Some(err_str) = parsed.get("error").and_then(|e| e.as_str()) {
        if !err_str.is_empty() {
            return Some((502, err_str.to_string()));
        }
    }

    // Shape 1: {"message":"upstream error","code":503}
    // Shape 4: {"detail":"Service Unavailable"}
    let has_code = parsed.get("code").and_then(|c| c.as_u64()).is_some();
    let msg = parsed
        .get("message")
        .and_then(|m| m.as_str())
        .or_else(|| parsed.get("detail").and_then(|d| d.as_str()));
    if has_code {
        let code = parsed.get("code").and_then(|c| c.as_u64()).unwrap_or(0) as u16;
        let message = msg.unwrap_or("upstream error");
        return Some((if code == 0 { 502 } else { code }, message.to_string()));
    }

    None
}

/// Convert a non-streaming chat completions response to a Responses API response object.
/// Used for compact and non-streaming /v1/responses.
pub fn chat_completion_to_response(raw: &Value, request_json: Option<&Value>) -> Value {
    let id = raw
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("resp_unknown")
        .to_string();
    let response_id = if let Some(rest) = id.strip_prefix("chatcmpl_") {
        format!("resp_{}", rest)
    } else if id.starts_with("resp_") {
        // Already correctly prefixed — keep verbatim. The previous logic
        // stripped the prefix and then re-added it nowhere, so a
        // `resp_already` id became `already`.
        id
    } else {
        format!("resp_{}", id)
    };

    let created = raw
        .get("created")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| chrono::Utc::now().timestamp());

    // end_turn: false if the model emitted tool calls, true otherwise.
    let has_tool_calls = raw
        .get("choices")
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
        for field in [
            "instructions",
            "max_output_tokens",
            "max_tool_calls",
            "model",
            "parallel_tool_calls",
            "previous_response_id",
            "prompt_cache_key",
            "reasoning",
            "safety_identifier",
            "service_tier",
            "store",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
            "metadata",
        ] {
            if let Some(v) = req.get(field) {
                resp[field] = v.clone();
            }
        }
    } else if let Some(model) = raw.get("model") {
        resp["model"] = model.clone();
    }

    let mut output = Vec::new();

    let rc_text = raw
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| {
            // Some vLLM backends use "reasoning_content", GLM-5.2 uses "reasoning"
            m.get("reasoning_content").or_else(|| m.get("reasoning"))
        })
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let include_reasoning =
        !rc_text.is_empty() || request_json.and_then(|r| r.get("reasoning")).is_some();

    if include_reasoning {
        let rid = if let Some(rest) = response_id.strip_prefix("resp_") {
            rest.to_string()
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
                        let call_id = tc
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let args = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let is_custom = is_custom_tool(&name);
                        let (ns, short_name) = split_namespaced_tool_name(&name);
                        let item = if is_custom {
                            let input = extract_custom_tool_input(&args);
                            json!({
                                "id": format!("fc_{}", call_id),
                                "type": "custom_tool_call",
                                "status": "completed",
                                "input": input,
                                "call_id": call_id,
                                "name": short_name,
                                "namespace": ns,
                            })
                        } else {
                            json!({
                                "id": format!("fc_{}", call_id),
                                "type": "function_call",
                                "status": "completed",
                                "arguments": args,
                                "call_id": call_id,
                                "name": short_name,
                                "namespace": ns,
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
        if usage.get("prompt_tokens").is_some()
            || usage.get("completion_tokens").is_some()
            || usage.get("total_tokens").is_some()
        {
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
