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

    // ── Inline-text tool-call scanner ──────────────────────────────────────
    // GLM-5.x sometimes emits tool calls as inline markup in `delta.content`
    // (e.g. `[Bash command="git status" description="..."]`) instead of via
    // the structured `delta.tool_calls` SSE field. When that happens the
    // markup would otherwise pass straight through as a `text` content block
    // and Claude Code prints it verbatim. The scanner buffers text and, when
    // it recognizes the markup, converts it into a proper `tool_use` block.
    //
    // `text_buffer` accumulates raw `delta.content` chunks so a tool-call tag
    // split across many SSE chunks can be detected and parsed as a whole.
    text_buffer: String,
    // True once we've seen an opening `[` that begins a tool-call tag and are
    // accumulating the body up to the matching `]`.
    in_tool_tag: bool,
    // Accumulated body inside a tool-call tag (the part between `[Name` and
    // the closing `]`), used to parse key="value" pairs once complete.
    tool_tag_body: String,
    // Bracket depth inside a tool-call tag. The arguments can legitimately
    // contain `[`/`]` (e.g. shell glob qualifiers), so we only close the tag
    // at depth 0. String literals (between unescaped `"`) are skipped over so
    // brackets/quotes inside a value don't affect depth or close the tag.
    tool_tag_depth: i32,
    // True while the next char to scan inside a tool tag is inside a
    // double-quoted string value.
    in_tool_tag_string: bool,
    // True once at least one inline-text tool call has been converted into a
    // tool_use block this turn. Forces stop_reason=tool_use even when the
    // upstream finish_reason is "stop" (the model used the text channel).
    inline_tool_emitted: bool,
}

#[derive(Default)]
struct PendingToolCall {
    index: usize,
    id: String,
    name: String,
    arguments_buffer: String,
    emitted: bool,
}

// ── Inline tool-call tag parsing helpers ───────────────────────────────────
//
// These operate on the streaming text buffer. The grammar they recognize:
//
//     tag      := "[" name ws+ attr+ "]"
//     attr     := key ws* "=" ws* value
//     value    := '"' (char | escape)* '"'
//     escape   := '\"' | '\\' | anything-after-backslash
//     name/key := ASCII alpha / digit / '_'
//
// `find_tool_tag_open` locates the `[` that begins such a tag (conservatively);
// `take_tool_tag_body` drains the buffer up to the matching `]` at depth 0,
// honoring string literals; `parse_tool_tag_body` extracts name + args.

/// A parsed inline tool-call tag: `[Bash command="..." description="..."]`.
struct ParsedToolCall {
    name: String,
    /// Ordered key→value map. Preserved as an ordered map so the JSON input
    /// we emit matches the model's argument order (matters for some clients).
    arguments: serde_json::Map<String, Value>,
}

enum ToolTagMatch {
    /// A complete tag opener was found; the inner value is the byte index of
    /// the opening `[` in the buffer.
    Full(usize),
    /// The buffer ends with bytes that could be the start of a tag opener
    /// (e.g. `[Ba`); the inner value is how many trailing bytes to hold back.
    Partial(usize),
    /// No tag opener is possible in the buffer.
    None,
}

enum TagBodyResult {
    /// The closing `]` was found and consumed; `body` holds the tag interior.
    Closed,
    /// Need more input to reach the closing `]`.
    Pending,
}

/// True if `c` may appear in a tool name or attribute key.
fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Scan `buf` for the opening `[` of a tool-call tag.
///
/// Returns `Full(pos)` when `buf[pos] == '['` and the bytes immediately after
/// form a tool-name identifier followed by whitespace and a `key=` attribute —
/// the conservative signature that distinguishes `[Bash command="..."]` from an
/// array literal like `[1, 2, 3]` or prose like `[see note]`.
///
/// Returns `Partial(hold)` when the buffer ends with a prefix of such an opener
/// (so a tag split across chunks can still be recognized once more arrives).
/// Returns `None` when no opener is possible.
fn find_tool_tag_open(buf: &str) -> ToolTagMatch {
    let bytes = buf.as_bytes();
    let mut last_bracket = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            // Tentatively record this bracket; validate below.
            if validate_opener(buf, i) {
                return ToolTagMatch::Full(i);
            }
            last_bracket = Some(i);
        }
        i += 1;
    }
    // If the buffer ends with a possible partial opener (a trailing `[` plus a
    // partial identifier), hold those bytes back for the next chunk.
    if let Some(start) = last_bracket {
        let hold = bytes.len() - start;
        if hold <= AnthropicStreamTranslator::TOOL_TAG_MAX_HOLD {
            return ToolTagMatch::Partial(hold);
        }
    } else if bytes.last() == Some(&b'[') {
        return ToolTagMatch::Partial(1);
    }
    ToolTagMatch::None
}

/// Validate that `buf[pos] == '['` begins a tool-call tag opener. Returns true
/// if it looks like a tag opener. We require: `[`, ≥1 ident char (name),
/// whitespace, ≥1 ident char (first attr key), then `=`. This is strict enough
/// to reject prose arrays and loose enough to match `[Bash command="..."]`.
fn validate_opener(buf: &str, pos: usize) -> bool {
    let rest = match buf.get(pos + 1..) {
        Some(r) => r,
        None => return false,
    };
    let mut chars = rest.char_indices().peekable();
    // Tool name: at least one identifier character.
    let mut name_len = 0;
    while let Some(&(_, c)) = chars.peek() {
        if is_ident_char(c) {
            name_len += 1;
            chars.next();
        } else {
            break;
        }
    }
    if name_len == 0 {
        return false;
    }
    // Whitespace separating the tool name from the first attribute.
    let mut saw_ws = false;
    while let Some(&(_, c)) = chars.peek() {
        if c.is_whitespace() {
            saw_ws = true;
            chars.next();
        } else {
            break;
        }
    }
    if !saw_ws {
        return false;
    }
    // First attribute key: at least one identifier character, then '='.
    let mut key_len = 0;
    while let Some(&(_, c)) = chars.peek() {
        if is_ident_char(c) {
            key_len += 1;
            chars.next();
        } else {
            break;
        }
    }
    if key_len == 0 {
        // Could be a partial key still streaming. Signal "not yet" — the
        // caller treats a trailing bracket as Partial, so we'll retry.
        return false;
    }
    matches!(chars.peek(), Some(&(_, '=')))
}

/// Drain `buf` into `body`, tracking bracket depth and string literals, until
/// the matching `]` at depth 0 is found (and consumed). Returns `Closed` when
/// the tag is complete, or `Pending` when more input is needed.
///
/// On entry, the opening `[` has already been consumed; `body` accumulates the
/// interior. Brackets inside string literals are ignored; unescaped `[`
/// increases depth and unescaped `]` at depth 0 closes the tag.
///
/// Scans by `char` (not byte) so multi-byte UTF-8 in values is preserved; the
/// control characters we track (`[`, `]`, `"`, `\`) are all ASCII.
fn take_tool_tag_body(
    buf: &mut String,
    body: &mut String,
    depth: &mut i32,
    in_string: &mut bool,
) -> TagBodyResult {
    // Walk the buffer by char, copying each into `body`, until we hit the
    // closing `]` at depth 0 (outside a string). We track byte offsets so we
    // can drain exactly the consumed prefix from `buf`.
    let mut consumed_bytes = 0usize;
    let mut chars = buf.char_indices().peekable();
    while let Some((byte_idx, c)) = chars.next() {
        consumed_bytes = byte_idx + c.len_utf8();
        if *in_string {
            body.push(c);
            if c == '\\' {
                // Escaped next char: consume it together. If it's at the end of
                // the buffer, wait for more input.
                if let Some(&(_, next)) = chars.peek() {
                    body.push(next);
                    chars.next();
                    consumed_bytes = byte_idx + c.len_utf8() + next.len_utf8();
                    continue;
                } else {
                    // Don't consume the dangling backslash yet.
                    consumed_bytes = byte_idx;
                    break;
                }
            }
            if c == '"' {
                *in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                *in_string = true;
                body.push('"');
            }
            '[' => {
                *depth += 1;
                body.push('[');
            }
            ']' => {
                if *depth == 0 {
                    // Closing bracket found. Drain everything up to and
                    // including it from `buf`.
                    let close_byte = byte_idx + 1; // include the ']'
                    let _ = buf.drain(..close_byte);
                    return TagBodyResult::Closed;
                }
                *depth -= 1;
                body.push(']');
            }
            _ => {
                body.push(c);
            }
        }
    }
    // Reached end of buffer without a closing bracket.
    let _ = buf.drain(..consumed_bytes);
    TagBodyResult::Pending
}

/// Parse the interior of a tool-call tag (the text between `[` and `]`) into a
/// `ParsedToolCall`. The interior looks like:
///     Bash command="git status" description="Show status"
/// Returns `None` if the body does not contain at least a name + one
/// `key="value"` pair.
fn parse_tool_tag_body(body: &str) -> Option<ParsedToolCall> {
    let body = body.trim();
    // Tool name: leading identifier.
    let mut name_end = 0;
    for (i, c) in body.char_indices() {
        if is_ident_char(c) {
            name_end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if name_end == 0 {
        return None;
    }
    let name = body[..name_end].to_string();
    let rest = body[name_end..].trim_start();

    let mut arguments = serde_json::Map::new();
    let mut s = rest;
    while !s.is_empty() {
        // Skip leading whitespace/commas between attributes.
        s = s.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
        if s.is_empty() {
            break;
        }
        // Attribute key.
        let key_end = s.find(|c: char| !is_ident_char(c)).unwrap_or(s.len());
        if key_end == 0 {
            // Non-identifier where a key was expected; bail.
            return None;
        }
        let key = s[..key_end].to_string();
        s = s[key_end..].trim_start();
        // Expect '='.
        if !s.starts_with('=') {
            return None;
        }
        s = s[1..].trim_start();
        // Expect opening '"'.
        if !s.starts_with('"') {
            return None;
        }
        s = &s[1..];
        // Value: read until unescaped closing '"'. Use char_indices so we can
        // compute the byte length consumed (including the closing quote).
        let mut value = String::new();
        let mut closed_byte: Option<usize> = None;
        let mut chars = s.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            if c == '\\' {
                if let Some(&(_, next)) = chars.peek() {
                    match next {
                        '"' => value.push('"'),
                        '\\' => value.push('\\'),
                        'n' => value.push('\n'),
                        't' => value.push('\t'),
                        other => {
                            value.push('\\');
                            value.push(other);
                        }
                    }
                    chars.next();
                } else {
                    value.push('\\');
                }
            } else if c == '"' {
                closed_byte = Some(i);
                break;
            } else {
                value.push(c);
            }
        }
        let close_pos = closed_byte?;
        // Advance `s` past the closing quote.
        s = &s[close_pos + 1..];
        arguments.insert(key, Value::String(value));
    }

    if arguments.is_empty() {
        return None;
    }
    Some(ParsedToolCall { name, arguments })
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
            text_buffer: String::new(),
            in_tool_tag: false,
            tool_tag_body: String::new(),
            tool_tag_depth: 0,
            in_tool_tag_string: false,
            inline_tool_emitted: false,
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

                    // Text content → route through the inline-tool-call scanner.
                    // GLM-5.x can emit tool calls as `[Name key="val"]` markup in
                    // `delta.content`; the scanner buffers text, converts any such
                    // markup into `tool_use` blocks, and passes the rest through as
                    // ordinary `text_delta` events.
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
                            events.extend(self.handle_text_content(content));
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
                        "stop" => {
                            // If we converted an inline-text tool call into a
                            // tool_use block, the turn must end with `tool_use`
                            // (not `end_turn`) so the client executes the tool —
                            // even though the upstream finish_reason is `stop`,
                            // because the model used the text channel.
                            if self.inline_tool_emitted { "tool_use" } else { "end_turn" }
                        }
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

    /// Close out the stream after an in-stream upstream error, without
    /// synthesizing the empty-text fallback that `finish()` applies.
    ///
    /// The backend can reject a request with HTTP 200 plus an SSE error event
    /// (e.g. `{"message":"upstream error","code":503}` when
    /// prompt_tokens + max_tokens exceeds the model's context length). Those
    /// lines carry no `choices`, so `translate_line()` skips them and
    /// `finish()` would emit an empty assistant text block — which Claude Code
    /// CLI accepts as a legitimate turn that simply said nothing, silently
    /// ending the conversation instead of reporting the failure.
    ///
    /// The caller emits an Anthropic `error` event instead. We only close any
    /// content block left open by partial output, so the emitted stream stays
    /// well-formed; no message_delta/message_stop is produced, because the
    /// turn did not complete.
    pub fn abort_for_error(&mut self) -> Vec<String> {
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
        self.finished = true;
        events
    }

    /// Returns true if the upstream sent a finish_reason for any choice.
    /// When false, the stream ended without the model signaling completion
    /// (e.g. silent stall, connection drop). Callers should emit an error
    /// event instead of finish() to avoid the client treating a truncated
    /// stream as a complete turn (end_turn).
    pub fn has_finish_reason(&self) -> bool {
        !self.stop_reason.is_empty()
    }

    fn finish(&mut self) -> Vec<String> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut events = Vec::new();
        // Emit message_start FIRST if not already started. The Anthropic SSE
        // protocol requires message_start before any content_block events.
        // Previously this was placed after the synthesized content blocks
        // below, causing Claude Code CLI's stream parser to discard those
        // blocks (they arrived before message_start) and fail with
        // "no assistant message in summarization response" during /compact
        // when the upstream returned an empty response.
        if !self.started {
            self.started = true;
            events.push(self.message_start_event());
        }
        // Flush any text the inline-tool-call scanner is still holding back
        // (a partial tag opener or an unclosed tag at stream end). Without
        // this, a truncated `[Bash command="..."` (no closing `]`) would be
        // silently dropped. Emit it verbatim as text so nothing is lost.
        events.extend(self.flush_text_buffer());
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
        // Empty-response fallback: if the upstream produced no assistant
        // content at all — no text, no reasoning, and no tool_use (e.g. an
        // in-stream error object the translator skipped, an empty body, or a
        // dropped connection) — the Anthropic stream so far has zero content
        // blocks. Clients like Claude Code CLI require an assistant text
        // message; a contentless message yields "no assistant message in
        // summarization response" and breaks /compact. Synthesize an empty
        // text content block so every response carries an assistant message.
        let has_tool_use = self
            .pending_tool_calls
            .iter()
            .any(|t| !t.name.is_empty());
        if !self.text_ever_emitted && self.reasoning_text.is_empty() && !has_tool_use {
            events.push(self.text_block_start_event(self.current_block_index));
            events.push(self.content_block_stop_event(self.current_block_index));
            self.current_block_index += 1;
        }
        // Emit message_delta now — all usage chunks have been processed.
        let sr = if self.stop_reason.is_empty() { "end_turn" } else { self.stop_reason.as_str() };
        events.push(self.message_delta_event(sr));
        events.push(self.message_stop_event());
        events
    }

    // ── Inline-text tool-call scanner ──────────────────────────────────────
    //
    // GLM-5.x sometimes emits tool calls as inline markup in `delta.content`
    // instead of via the structured `delta.tool_calls` SSE field, e.g.:
    //
    //     I'll check that now.
    //     [Bash command="git status --short" description="Show git status"]
    //
    // Without this scanner, the `[Bash ...]` markup would be streamed out as
    // an Anthropic `text` content block and Claude Code would render it as
    // literal assistant text (prefixed with the `●` text bullet) instead of
    // executing the tool. The scanner buffers `delta.content`, recognizes the
    // `[Name key="val" ...]` markup even when split across many SSE chunks,
    // converts each complete tag into a `tool_use` block, and passes through
    // everything else as ordinary `text_delta` events.
    //
    // Recognition is deliberately conservative: a tag is only treated as a
    // tool call when the byte after `[` starts an ASCII-identifier tool name
    // followed by whitespace and a `key=` attribute. This avoids swallowing
    // array literals or bracketed prose that happen to appear in model output.

    /// Maximum length of plain text we hold back while scanning for the opening
    /// of a tool-call tag. A tag opening is at most `[` + a tool name + ` key=`,
    /// which is well under 64 chars; holding back this many bytes lets a tag
    /// split across chunks be rejoined without ever buffering unbounded text.
    const TOOL_TAG_MAX_HOLD: usize = 64;

    fn handle_text_content(&mut self, content: &str) -> Vec<String> {
        let mut out = Vec::new();
        self.text_buffer.push_str(content);

        loop {
            if !self.in_tool_tag {
                // Looking for the start of a tool-call tag: `[` followed by an
                // ASCII-identifier tool name.
                match find_tool_tag_open(&self.text_buffer) {
                    ToolTagMatch::Full(start) => {
                        // Emit any plain text before the tag as text_delta.
                        if start > 0 {
                            let before: String = self.text_buffer.drain(..start).collect();
                            if !before.is_empty() {
                                out.push(self.text_delta_event(self.current_block_index, &before));
                            }
                        }
                        // Consume the `[` and enter tag mode.
                        self.text_buffer.drain(..1);
                        self.in_tool_tag = true;
                        self.tool_tag_body.clear();
                        self.tool_tag_depth = 0;
                        self.in_tool_tag_string = false;
                        continue;
                    }
                    ToolTagMatch::Partial(hold) => {
                        // The buffer ends with a possible partial tag opening
                        // (e.g. `[Ba`). Emit everything before it, hold the rest.
                        let emit_len = self.text_buffer.len().saturating_sub(hold);
                        if emit_len > 0 {
                            let to_emit: String = self.text_buffer.drain(..emit_len).collect();
                            if !to_emit.is_empty() {
                                out.push(self.text_delta_event(self.current_block_index, &to_emit));
                            }
                        }
                        return out;
                    }
                    ToolTagMatch::None => {
                        // No tag opening possible. Emit all but a suffix that
                        // could be the start of `[` (a lone trailing `[`).
                        let hold = if self.text_buffer.ends_with('[') { 1 } else { 0 };
                        let emit_len = self.text_buffer.len().saturating_sub(hold);
                        if emit_len > 0 {
                            let to_emit: String = self.text_buffer.drain(..emit_len).collect();
                            if !to_emit.is_empty() {
                                out.push(self.text_delta_event(self.current_block_index, &to_emit));
                            }
                        }
                        return out;
                    }
                }
            }

            // Inside a tool-call tag: accumulate the body until the matching
            // `]` at bracket depth 0, tracking string literals so brackets and
            // quotes inside a value don't affect depth or close the tag.
            match take_tool_tag_body(&mut self.text_buffer, &mut self.tool_tag_body,
                                     &mut self.tool_tag_depth, &mut self.in_tool_tag_string) {
                TagBodyResult::Closed => {
                    // The closing `]` was consumed. Parse the body and emit a
                    // tool_use block, then return to plain-text scanning.
                    if let Some(tc) = parse_tool_tag_body(&self.tool_tag_body) {
                        out.extend(self.emit_inline_tool_use(&tc));
                    } else {
                        // Not actually a parseable tool call (e.g. matched the
                        // conservative opener but body was malformed). Emit the
                        // original `[…]` text verbatim so nothing is lost.
                        let raw = format!("[{}]", self.tool_tag_body);
                        out.push(self.text_delta_event(self.current_block_index, &raw));
                    }
                    self.in_tool_tag = false;
                    self.tool_tag_body.clear();
                    continue;
                }
                TagBodyResult::Pending => {
                    // Need more chunks to reach the closing `]`.
                    return out;
                }
            }
        }
    }

    /// Emit a parsed inline tool call as an Anthropic `tool_use` content block.
    /// Closes any open text block first (a tool call always starts a new block),
    /// then emits `content_block_start` (tool_use) + `input_json_delta` +
    /// `content_block_stop`. Also records it in `pending_tool_calls` so the
    /// finish-time `stop_reason` logic sees that a tool was used.
    fn emit_inline_tool_use(&mut self, tc: &ParsedToolCall) -> Vec<String> {
        let mut out = Vec::new();
        // Close the open text block — the tool_use block is a new block.
        if self.text_block_open {
            out.push(self.content_block_stop_event(self.current_block_index));
            self.current_block_index += 1;
            self.text_block_open = false;
        }
        let block_index = self.current_block_index;
        let id = format!("chatcmpl-tool-{}", &uuid::Uuid::new_v4().simple().to_string()[..16]);
        out.push(self.tool_use_block_start_event(block_index, &id, &tc.name));
        // Build the JSON input from the parsed key/value pairs and stream it
        // as a single input_json_delta. Values are JSON strings, so embedded
        // quotes/newlines are escaped by serde_json.
        let input_json = serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".to_string());
        if !input_json.is_empty() && input_json != "{}" {
            out.push(self.tool_use_delta_event(block_index, &input_json));
        }
        out.push(self.content_block_stop_event(block_index));
        self.current_block_index = block_index + 1;

        // Record so finish()/stop_reason sees a tool was used.
        let pending = PendingToolCall {
            index: self.pending_tool_calls.len(),
            id: id.clone(),
            name: tc.name.clone(),
            arguments_buffer: input_json.clone(),
            emitted: true,
        };
        self.pending_tool_calls.push(pending);
        self.inline_tool_emitted = true;
        // Reopen the text block for any prose the model emits after the tag.
        out.push(self.text_block_start_event(self.current_block_index));
        self.text_block_open = true;
        out
    }

    /// Flush any text the scanner is still holding back at stream end.
    ///
    /// Two cases:
    ///  - Not inside a tag, but `text_buffer` ends with a partial opener
    ///    (e.g. `[Ba`) that was held for the next chunk that never came.
    ///  - Inside an unclosed tag (`in_tool_tag == true`): the model emitted
    ///    `[Bash command="..."` with no closing `]`. Emit the literal `[` +
    ///    accumulated body + remaining buffer as text so nothing is silently
    ///    dropped.
    ///
    /// Emits a `text_delta` for the recovered text. The caller is responsible
    /// for opening/closing the surrounding text block.
    fn flush_text_buffer(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if self.in_tool_tag {
            // Reconstruct the verbatim `[body…` text. The opening `[` was
            // consumed on entry to tag mode, so prepend it.
            let mut raw = String::from("[");
            raw.push_str(&self.tool_tag_body);
            raw.push_str(&self.text_buffer);
            self.text_buffer.clear();
            self.tool_tag_body.clear();
            self.in_tool_tag = false;
            self.tool_tag_depth = 0;
            self.in_tool_tag_string = false;
            if !raw.is_empty() {
                if !self.text_block_open {
                    out.push(self.text_block_start_event(self.current_block_index));
                    self.text_block_open = true;
                    self.text_ever_emitted = true;
                }
                out.push(self.text_delta_event(self.current_block_index, &raw));
            }
        } else if !self.text_buffer.is_empty() {
            let raw = std::mem::take(&mut self.text_buffer);
            if !self.text_block_open {
                out.push(self.text_block_start_event(self.current_block_index));
                self.text_block_open = true;
                self.text_ever_emitted = true;
            }
            out.push(self.text_delta_event(self.current_block_index, &raw));
        }
        out
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

            // Always emit a usage object. Upstream may omit `usage` entirely
            // (empty or error-ish responses); the Anthropic message shape
            // requires it, so default both counters to 0 rather than leaving
            // the field absent.
            let usage = response.get("usage");
            let input_tokens = usage
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let output_tokens = usage
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            out["usage"] = json!({"input_tokens": input_tokens, "output_tokens": output_tokens});
            // Empty-response fallback (non-streaming): if no content block was
            // added (no text, no reasoning, no tool_use), synthesize an empty
            // text block so the message always carries an assistant message.
            // A contentless response breaks Claude Code CLI /compact with
            // "no assistant message in summarization response". Mirrors the
            // streaming translator's finish() empty-response fallback.
            if content.is_empty() {
                content.push(json!({"type": "text", "text": ""}));
                out["content"] = json!(content);
            }
            return out;
        }
    }

    // No choices at all (e.g. upstream error object or empty body). Synthesize
    // an empty text block so the response always carries an assistant message,
    // avoiding "no assistant message in summarization response" in /compact.
    json!({
        "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
        "type": "message", "role": "assistant", "content": [{"type": "text", "text": ""}],
        "model": model, "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: feed SSE lines to the translator and return all events.
    fn run_stream(lines: &[&str]) -> Vec<String> {
        let mut t = AnthropicStreamTranslator::new("test-model");
        let mut all = Vec::new();
        for line in lines {
            all.extend(t.translate_line(line));
        }
        all
    }

    /// Extract the JSON payload from an SSE event string.
    fn parse_event(event: &str) -> Option<Value> {
        if let Some(pos) = event.find("data: ") {
            serde_json::from_str(event[pos + 6..].trim()).ok()
        } else {
            None
        }
    }

    /// Find the message_delta event from the event list and return its JSON.
    fn find_message_delta(events: &[String]) -> Option<Value> {
        for ev in events.iter().rev() {
            if ev.contains("message_delta") {
                return parse_event(ev);
            }
        }
        None
    }

    /// When finish_reason is "stop", stop_reason should be "end_turn".
    #[test]
    fn test_normal_stop_emits_end_turn() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"Hello world"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let delta = find_message_delta(&events).expect("should have message_delta");
        let stop_reason = delta
            .get("delta")
            .and_then(|d| d.get("stop_reason"))
            .and_then(|s| s.as_str());
        assert_eq!(stop_reason, Some("end_turn"));
    }

    /// When finish_reason is "length" (max_tokens), stop_reason should be
    /// "max_tokens" so Claude Code knows the response was truncated.
    #[test]
    fn test_max_tokens_emits_max_tokens_stop_reason() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"truncated mid-sentence"},"finish_reason":"length"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":16384,"total_tokens":16484}}"#,
            "data: [DONE]",
        ]);

        let delta = find_message_delta(&events).expect("should have message_delta");
        let stop_reason = delta
            .get("delta")
            .and_then(|d| d.get("stop_reason"))
            .and_then(|s| s.as_str());
        assert_eq!(
            stop_reason,
            Some("max_tokens"),
            "finish_reason=length should produce stop_reason=max_tokens"
        );
    }

    /// When finish_reason is "tool_calls", stop_reason should be "tool_use".
    #[test]
    fn test_tool_calls_emits_tool_use_stop_reason() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            "data: [DONE]",
        ]);

        let delta = find_message_delta(&events).expect("should have message_delta");
        let stop_reason = delta
            .get("delta")
            .and_then(|d| d.get("stop_reason"))
            .and_then(|s| s.as_str());
        assert_eq!(stop_reason, Some("tool_use"));
    }

    /// When the stream ends with [DONE] but NO finish_reason was ever sent
    /// (silent stall / connection drop), has_finish_reason() should return
    /// false so the caller can emit an error instead of finish() defaulting
    /// to "end_turn".
    #[test]
    fn test_missing_finish_reason_detected() {
        let mut t = AnthropicStreamTranslator::new("test-model");
        // Stream content but never send finish_reason
        let _ = t.translate_line(
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#,
        );
        // Simulate stream ending with [DONE] — no finish_reason chunk
        let _ = t.translate_line("data: [DONE]");

        assert!(
            !t.has_finish_reason(),
            "stream without finish_reason should have has_finish_reason() = false"
        );
    }

    /// When finish_reason is present, has_finish_reason() should return true.
    #[test]
    fn test_finish_reason_present_detected() {
        let mut t = AnthropicStreamTranslator::new("test-model");
        let _ = t.translate_line(
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"complete"},"finish_reason":"stop"}]}"#,
        );
        let _ = t.translate_line("data: [DONE]");

        assert!(
            t.has_finish_reason(),
            "stream with finish_reason='stop' should have has_finish_reason() = true"
        );
    }

    // ── Inline-text tool-call scanner tests ────────────────────────────────

    /// Collect all `tool_use` content_block_start events from the stream.
    fn collect_tool_use(events: &[String]) -> Vec<Value> {
        events
            .iter()
            .filter_map(|ev| {
                let v = parse_event(ev)?;
                if v.get("type").and_then(|t| t.as_str()) == Some("content_block_start") {
                    let cb = v.get("content_block")?;
                    if cb.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        return Some(cb.clone());
                    }
                }
                None
            })
            .collect()
    }

    /// Extract the concatenated `partial_json` (input_json_delta) for a given
    /// block index.
    fn input_json_for(events: &[String], index: i64) -> Option<String> {
        let mut s = String::new();
        for ev in events {
            let v = parse_event(ev)?;
            if v.get("type").and_then(|t| t.as_str()) != Some("content_block_delta") {
                continue;
            }
            if v.get("index").and_then(|i| i.as_i64()) != Some(index) {
                continue;
            }
            if let Some(pj) = v
                .get("delta")
                .and_then(|d| d.get("partial_json"))
                .and_then(|p| p.as_str())
            {
                s.push_str(pj);
            }
        }
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    /// A complete `[Bash command="..." description="..."]` tag emitted as a
    /// single content delta should be converted into a tool_use block, not
    /// streamed as text.
    #[test]
    fn test_inline_tool_call_single_chunk() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"[Bash command=\"git status\" description=\"Show git status\"]"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 1, "expected exactly one tool_use block");
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Bash")
        );
        // The tool_use block is emitted at index 1 (after the text block that
        // holds preceding/following prose). Find the input_json_delta.
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["command"], "git status");
        assert_eq!(parsed["description"], "Show git status");

        // stop_reason must be tool_use even though finish_reason was "stop".
        let delta = find_message_delta(&events).expect("should have message_delta");
        let stop_reason = delta
            .get("delta")
            .and_then(|d| d.get("stop_reason"))
            .and_then(|s| s.as_str());
        assert_eq!(stop_reason, Some("tool_use"));
    }

    /// A tool-call tag split across many SSE chunks (the realistic case) must
    /// still be reassembled and converted to a tool_use block.
    #[test]
    fn test_inline_tool_call_split_across_chunks() {
        let tag = "[Bash command=\"git status --short\" description=\"Show status\"]";
        // Feed the tag in several irregular slices to stress chunk-boundary
        // handling (splits mid-identifier, mid-value, mid-escape).
        let slices: &[&str] = &[
            "[Bash comm",
            "and=\"git status --short\" desc",
            "ription=\"Show status\"",
            "]",
        ];
        // Reconstruct the expected tag from the slices to sanity-check.
        assert_eq!(slices.concat(), tag);
        let mut lines: Vec<String> = Vec::new();
        for slice in slices {
            let content_json = serde_json::to_string(slice).unwrap();
            lines.push(format!(
                r#"data: {{"id":"1","model":"test","choices":[{{"index":0,"delta":{{"content":{}}}}}]}}"#,
                content_json
            ));
        }
        lines.push(r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#.to_string());
        lines.push(r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#.to_string());
        lines.push("data: [DONE]".to_string());
        let lines_ref: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();

        let events = run_stream(&lines_ref);
        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 1, "split tag should yield one tool_use");
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Bash")
        );
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["command"], "git status --short");
        assert_eq!(parsed["description"], "Show status");
    }

    /// Prose before a tool call should be emitted as text, and the tool call
    /// after it as a tool_use block.
    #[test]
    fn test_inline_tool_call_with_preceding_text() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"I'll check that now.\n[Bash command=\"ls -la\" description=\"List files\"]"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 1);
        // The text before the tag should appear as a text_delta at index 0.
        let text_deltas: Vec<String> = events
            .iter()
            .filter_map(|ev| {
                let v = parse_event(ev)?;
                if v.get("type").and_then(|t| t.as_str()) == Some("content_block_delta")
                    && v.get("index").and_then(|i| i.as_i64()) == Some(0)
                {
                    v.get("delta")?.get("text")?.as_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect();
        let combined: String = text_deltas.concat();
        assert!(combined.contains("I'll check that now."), "preceding text should pass through, got: {combined}");
        assert!(!combined.contains("[Bash"), "tool tag should NOT leak into text, got: {combined}");
    }

    /// A multi-line command value (newlines inside the quoted string) should
    /// parse correctly — this mirrors the real failure case where the model
    /// emits a multi-line shell script as the command.
    #[test]
    fn test_inline_tool_call_multiline_value() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"[Bash command=\"git status\ngit log -1\" description=\"Status and last commit\"]"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 1);
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["command"], "git status\ngit log -1");
        assert_eq!(parsed["description"], "Status and last commit");
    }

    /// A value containing an embedded double-quote (escaped as \") must parse
    /// without prematurely closing the string — e.g. grep -n '"katex"' file.
    #[test]
    fn test_inline_tool_call_escaped_quote_in_value() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"[Bash command=\"grep -n \\\"katex\\\" package.json\" description=\"Find katex\"]"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 1);
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["command"], "grep -n \"katex\" package.json");
    }

    /// A bracket inside a string value (e.g. a shell glob `[abc]`) must not
    /// close the tag prematurely.
    #[test]
    fn test_inline_tool_call_bracket_in_value() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"[Bash command=\"ls file[0-9].txt\" description=\"List numbered files\"]"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 1);
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["command"], "ls file[0-9].txt");
    }

    /// Two consecutive tool calls in one content delta.
    #[test]
    fn test_inline_tool_call_multiple() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"[Bash command=\"git status\" description=\"Status\"][Bash command=\"git log\" description=\"Log\"]"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 2, "expected two tool_use blocks");
        assert_eq!(tool_uses[0].get("name").and_then(|n| n.as_str()), Some("Bash"));
        assert_eq!(tool_uses[1].get("name").and_then(|n| n.as_str()), Some("Bash"));
    }

    /// A bracket that does NOT look like a tool-call tag (array literal or
    /// prose) must pass through as ordinary text, not be swallowed.
    #[test]
    fn test_non_tool_bracket_passes_through_as_text() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"Options: [1, 2, 3] and [see notes]"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 0, "non-tool brackets must not become tool_use");
        let text: String = events
            .iter()
            .filter_map(|ev| {
                let v = parse_event(ev)?;
                if v.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                    v.get("delta")?.get("text")?.as_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect();
        assert!(text.contains("[1, 2, 3]"), "array literal should pass through, got: {text}");
        assert!(text.contains("[see notes]"), "prose bracket should pass through, got: {text}");
    }

    /// A truncated tag at stream end (no closing `]`) should be emitted as
    /// verbatim text rather than held forever or dropped.
    #[test]
    fn test_inline_tool_call_truncated_emitted_as_text() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"[Bash command=\"git status\""},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        // No tool_use block — the tag never closed.
        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 0, "unclosed tag should not yield a tool_use");
        // The raw text should be present so nothing is silently lost.
        let text: String = events
            .iter()
            .filter_map(|ev| {
                let v = parse_event(ev)?;
                if v.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                    v.get("delta")?.get("text")?.as_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect();
        assert!(text.contains("[Bash"), "unclosed tag text should pass through, got: {text}");
    }
}
