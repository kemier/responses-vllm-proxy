use crate::logger;
use serde_json::{json, Value};

// ─── Anthropic Messages API → OpenAI Chat Completions ──────────────────────

/// Convert an Anthropic Messages API request body to an OpenAI Chat
/// Completions request body.
pub fn anthropic_to_chat_completions(model: &str, body: &Value, stream: bool) -> Value {
    // Default: assume the target can consume image content (image_url parts).
    // The /v1/messages handler passes vision_capable=false for models without
    // the `multimodal` routing tag so images are stripped before the backend
    // rejects the request with a 500.
    anthropic_to_chat_completions_with_vision(model, body, stream, true)
}

/// Same as [`anthropic_to_chat_completions`] but lets the caller control
/// whether image blocks are forwarded as OpenAI `image_url` content parts.
/// When `vision_capable` is false, image blocks are dropped (the backend has
/// no vision encoder and returns HTTP 500 "not a multimodal model").
pub fn anthropic_to_chat_completions_with_vision(
    model: &str,
    body: &Value,
    stream: bool,
    vision_capable: bool,
) -> Value {
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

    // effort → reasoning_effort. llama.cpp: "none" disables thinking;
    // otherwise the value is passed to the jinja chat template. The Qwen3.8
    // template only defines low/medium/xhigh (no "high"), so the Anthropic
    // effort enum is folded onto those levels.
    if let Some(re) = map_effort_to_reasoning_effort(body) {
        out["reasoning_effort"] = json!(re);
    }

    let messages = out["messages"].as_array_mut().unwrap();

    // system → first system message (string or array of content blocks)
    if let Some(system) = body.get("system") {
        let system_text = extract_text_from_content(system);
        if !system_text.is_empty() {
            messages.push(json!({"role": "system", "content": system_text}));
        }
    }

    // Convert messages. Claude Code CLI sometimes sends `system`-role messages
    // INSIDE `messages` (e.g. the "Available agent types for the Agent tool"
    // block injected mid-conversation when spawning a subagent). The Qwen chat
    // template rejects system messages after index 0 ("System message must be
    // at the beginning."), so stray system messages are converted to user
    // messages IN PLACE. Keeping the position preserves the rendered token
    // prefix turn-over-turn — merging them into the leading system message
    // rewrote the prefix and forced a full re-prefill of the entire
    // conversation on the next request.
    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            if role == "system" {
                let text = msg
                    .get("content")
                    .map(extract_text_from_content)
                    .unwrap_or_default();
                if !text.is_empty() {
                    messages.push(json!({"role": "user", "content": text}));
                }
                continue;
            }
            convert_anthropic_message(msg, messages, vision_capable);
        }
    }

    // Convert tools
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let add_placeholder = needs_noargs_placeholder(model);
        let openai_tools: Vec<Value> = tools
            .iter()
            .filter_map(|t| convert_anthropic_tool(t, add_placeholder))
            .collect();
        if add_placeholder {
            let patched: Vec<&str> = openai_tools
                .iter()
                .filter(|t| {
                    t["function"]["parameters"]["properties"]
                        .get(NOARGS_PLACEHOLDER)
                        .is_some()
                })
                .filter_map(|t| t["function"]["name"].as_str())
                .collect();
            if !patched.is_empty() {
                logger::debug(
                    "translate_anthropic",
                    &format!(
                        "[noargs] injected {} into: [{}]",
                        NOARGS_PLACEHOLDER,
                        patched.join(",")
                    ),
                );
            }
        }
        if !openai_tools.is_empty() {
            out["tools"] = json!(openai_tools);
        }
    }

    // tool_choice
    if let Some(tc) = body.get("tool_choice") {
        if let Some(tc_str) = tc.as_str() {
            match tc_str {
                "auto" => {
                    out["tool_choice"] = json!("auto");
                }
                "any" => {
                    out["tool_choice"] = json!("required");
                }
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
fn convert_anthropic_message(msg: &Value, messages: &mut Vec<Value>, vision_capable: bool) {
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
                            // Never emit an empty arguments string: vLLM
                            // rejects the whole request with HTTP 400
                            // ("Assistant tool call function.arguments must
                            // be valid JSON.") if it is not valid JSON.
                            let arguments =
                                serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
                            let arguments = if arguments.trim().is_empty() {
                                "{}".to_string()
                            } else {
                                arguments
                            };
                            tool_calls.push(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                }
                            }));
                        }
                        _ => {}
                    }
                }

                let content = if text_parts.is_empty() {
                    Value::Null
                } else {
                    json!(text_parts.join("\n"))
                };
                let mut assistant_msg = json!({"role": "assistant", "content": content});
                if !tool_calls.is_empty() {
                    assistant_msg["tool_calls"] = json!(tool_calls);
                }
                messages.push(assistant_msg);
            } else {
                // User message — may contain text, image and tool_result blocks
                let mut text_parts = Vec::new();
                let mut tool_results: Vec<(String, String)> = Vec::new();
                // Image blocks (Anthropic base64 source) are forwarded to the
                // backend as OpenAI `image_url` content parts (data: URL). The
                // backend vLLM accepts these for vision-capable models; for
                // text-only models we still emit them and let the backend's
                // chat template ignore/downcast them (same behavior as the
                // Responses path's `input_image` handling).
                let mut image_parts: Vec<Value> = Vec::new();

                for block in blocks {
                    match block.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                                text_parts.push(text.to_string());
                            }
                        }
                        Some("image") => {
                            // Anthropic image block:
                            //   { "type": "image", "source": { "type": "base64",
                            //     "media_type": "image/png", "data": "..." } }
                            // Non-vision models have no vision encoder: sending
                            // image_url parts to a text-only backend returns a
                            // hard 500 ("not a multimodal model"), so drop the
                            // image. Same policy as the Responses path and
                            // ericli's isVisionCapable guard.
                            if vision_capable {
                                if let Some(source) = block.get("source") {
                                    if source.get("type").and_then(|v| v.as_str()) == Some("base64")
                                    {
                                        let media_type = source
                                            .get("media_type")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("image/png");
                                        if let Some(data) =
                                            source.get("data").and_then(|v| v.as_str())
                                        {
                                            let data_url =
                                                format!("data:{};base64,{}", media_type, data);
                                            image_parts.push(json!({
                                                "type": "image_url",
                                                "image_url": {"url": data_url}
                                            }));
                                        }
                                    }
                                }
                            }
                        }
                        Some("tool_result") => {
                            let tool_use_id = block
                                .get("tool_use_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let content = block
                                .get("content")
                                .map(extract_text_from_content)
                                .unwrap_or_default();
                            tool_results.push((tool_use_id.to_string(), content));
                        }
                        _ => {}
                    }
                }

                for (tool_use_id, content) in &tool_results {
                    messages.push(
                        json!({"role": "tool", "tool_call_id": tool_use_id, "content": content}),
                    );
                }
                if !text_parts.is_empty() {
                    messages.push(json!({"role": role, "content": text_parts.join("\n")}));
                }
                let has_images = !image_parts.is_empty();
                if has_images {
                    // Merge trailing text into the image content array so the
                    // backend sees one multimodal user message, matching the
                    // OpenAI vision convention. Example:
                    //   { "role":"user", "content":[{"type":"text","text":"..."},
                    //     {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}] }
                    let mut content_parts = image_parts;
                    if let Some(last) = messages.last_mut() {
                        if last.get("role").and_then(|v| v.as_str()) == Some(role)
                            && last.get("content").and_then(|v| v.as_str()).is_some()
                        {
                            let text = last["content"].take().as_str().unwrap_or("").to_string();
                            if !text.is_empty() {
                                content_parts.insert(0, json!({"type": "text", "text": text}));
                            }
                            messages.pop();
                        }
                    }
                    messages.push(json!({"role": role, "content": content_parts}));
                }
                if text_parts.is_empty() && tool_results.is_empty() && !has_images {
                    messages.push(json!({"role": role, "content": ""}));
                }
            }
        }
        _ => {
            messages.push(json!({"role": role, "content": ""}));
        }
    }
}

/// Property name injected into zero-parameter tool schemas as a workaround for
/// the backend's streaming tool-call parser.
///
/// GLM-5.2-FP8 on the vLLM engine silently DROPS any tool call whose arguments
/// object is empty when `stream: true`: the turn comes back as
/// `finish_reason: "stop"` with no `tool_calls` and only the model's preamble
/// text ("Now let me list them."), so the client sees a turn that ended without
/// doing anything. The same request with `stream: false` returns the tool call
/// correctly, and giving the tool a single parameter also fixes it — verified
/// against the backend directly, bypassing this proxy.
///
/// So for affected models we give argument-less tools one optional throwaway
/// property. The model then emits `{"__noargs": ""}` instead of `{}`, the
/// parser keeps the tool call, and `strip_placeholder_arg` removes the property
/// again before the tool_use block reaches the client — which would otherwise
/// reject it as not matching the tool's schema.
const NOARGS_PLACEHOLDER: &str = "__noargs";

/// True when `model` needs the zero-argument tool workaround. Scoped to GLM:
/// DeepSeek-V4 streams empty-argument tool calls correctly (verified), and an
/// unexpected property in the schema is not free — it costs tokens and can
/// tempt a model into putting real content in it.
fn needs_noargs_placeholder(model: &str) -> bool {
    model.to_lowercase().contains("glm")
}

/// Remove the injected placeholder from a tool call's JSON arguments so the
/// client sees the arguments its own schema declares. Returns the input
/// unchanged when the placeholder is absent or the arguments aren't a JSON
/// object, so a real tool call is never mangled.
fn strip_placeholder_arg(arguments: &str) -> String {
    let Ok(mut parsed) = serde_json::from_str::<Value>(arguments) else {
        return arguments.to_string();
    };
    let Some(obj) = parsed.as_object_mut() else {
        return arguments.to_string();
    };
    if obj.remove(NOARGS_PLACEHOLDER).is_none() {
        return arguments.to_string();
    }
    serde_json::to_string(&parsed).unwrap_or_else(|_| arguments.to_string())
}

/// Convert an Anthropic tool definition to OpenAI function tool format.
///
/// `add_noargs_placeholder` gives a tool that declares no properties one
/// optional throwaway property — see [`NOARGS_PLACEHOLDER`].
fn convert_anthropic_tool(tool: &Value, add_noargs_placeholder: bool) -> Option<Value> {
    let name = tool.get("name").and_then(|v| v.as_str())?;
    let description = tool
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let mut input_schema = tool.get("input_schema").cloned().unwrap_or(json!({}));
    if add_noargs_placeholder {
        let has_properties = input_schema
            .get("properties")
            .and_then(|p| p.as_object())
            .map(|p| !p.is_empty())
            .unwrap_or(false);
        if !has_properties {
            // Build a fresh object schema rather than patching in place: an
            // argument-less tool may declare `input_schema` as `{}`, as
            // `{"type":"object"}`, or omit `properties` entirely.
            input_schema = json!({
                "type": "object",
                "properties": {
                    NOARGS_PLACEHOLDER: {
                        "type": "string",
                        "description": "Unused. Pass an empty string.",
                    }
                },
                // Required, not optional. With an optional placeholder the model
                // frequently still emits `{}` and the tool call is dropped again
                // (measured 1/5 success optional vs 5/5 required). Requiring it
                // guarantees non-empty arguments; strip_placeholder_arg removes
                // it before the client ever sees it.
                "required": [NOARGS_PLACEHOLDER],
            });
        }
    }
    Some(json!({
        "type": "function",
        "function": {"name": name, "description": description, "parameters": input_schema}
    }))
}

// ─── OpenAI Chat Completions SSE → Anthropic Messages SSE ──────────────────

/// State machine for translating OpenAI streaming SSE chunks to Anthropic
/// Messages SSE events.
/// Map the Anthropic request's effort signal onto the backend's
/// `reasoning_effort` field. Sources, in order:
///   1. `output_config.effort` — Claude Code v2+ ("minimal"|"low"|"medium"|
///      "high"|"xhigh"|"max")
///   2. `thinking` — classic Anthropic shape ({"type":"disabled"} or
///      {"type":"enabled","budget_tokens":N}); budget is folded onto levels.
/// Returns None when neither is present → the template's own default applies
/// (xhigh for Qwen3.8).
pub fn map_effort_to_reasoning_effort(body: &Value) -> Option<String> {
    if let Some(effort) = body
        .get("output_config")
        .and_then(|o| o.get("effort"))
        .and_then(|v| v.as_str())
    {
        return match effort {
            "minimal" | "none" | "off" => Some("none".to_string()),
            "low" => Some("low".to_string()),
            "medium" => Some("medium".to_string()),
            "high" | "xhigh" | "max" => Some("xhigh".to_string()),
            other => {
                logger::warn(
                    "translate_anthropic",
                    &format!("unknown output_config.effort: {other} — leaving template default"),
                );
                None
            }
        };
    }
    if let Some(thinking) = body.get("thinking") {
        if thinking.get("type").and_then(|v| v.as_str()) == Some("disabled") {
            return Some("none".to_string());
        }
        if let Some(budget) = thinking.get("budget_tokens").and_then(|v| v.as_i64()) {
            return Some(match budget {
                0 => "none".to_string(),
                b if b <= 2048 => "low".to_string(),
                b if b <= 8192 => "medium".to_string(),
                _ => "xhigh".to_string(),
            });
        }
    }
    None
}

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

    // ── Turn diagnostics ───────────────────────────────────────────────────
    // Total bytes of assistant text actually streamed to the client, plus a
    // short preview of the head of that text. Used only by the turn-summary
    // log line in finish(): when a client reports a silent/premature end_turn
    // we need to know whether the upstream produced no text, a few junk
    // tokens, or a tool call the client failed to act on.
    text_chars: usize,
    text_preview: String,
    // Raw `delta.content` exactly as the upstream sent it, before the
    // inline-tool-call scanner rewrites/withholds anything. Capped. When a turn
    // ends with a short preamble and no tool call, this is the only way to tell
    // "the model really stopped" apart from "the scanner ate a tool-call tag".
    raw_content: String,
    raw_chars: usize,
    // True once a non-streaming re-run recovered a tool call the streaming
    // response dropped. Reported in the turn summary so the workaround's hit
    // rate is visible in the log.
    recovered_non_streaming: bool,

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
    // Which inline-tool grammar the open tag matched. The square-bracket form
    // (`[Name key="value"]`) has a closing `]`; the angle-bracket form
    // (`<Name\nkey: "value"`) that GLM-5.2 also emits has NO closing tag and is
    // terminated by a blank line / the next opener / stream end. The body
    // drain and parser differ per kind.
    tool_tag_kind: ToolTagKind,
}

/// Which inline-tool-call grammar a streaming tag matched.
#[derive(Default, PartialEq, Clone, Copy)]
enum ToolTagKind {
    #[default]
    /// `[Name key="value" ...]` — closed by a matching `]` at depth 0.
    Square,
    /// `<Name\nkey: "value"` — NO closing tag; terminated by a blank line,
    /// the next opener, or stream end. (A bare-angle form; less common.)
    Angle,
    /// The HuggingFace/Qwen chat-template tool-call wrapper: the literal tag
    /// formed by `\x3c` + `tool_call` + `\x3e` (angle-bracket, no attributes),
    /// a newline, then the tool NAME on its own line, then `key: "value"`
    /// attribute lines. GLM-5.2 emits this in claude-code-acp sessions. May or
    /// may not be closed by the matching `\x3c/tool_call\x3e`; treat unclosed as
    /// terminating at the next opener / blank line / stream end.
    ToolCall,
}

/// The literal opener tag for the `ToolCall` grammar. Written with hex escapes
/// so the source contains no literal angle-bracket-tool token.
const TOOL_CALL_OPEN_TAG: &str = "\x3ctool_call\x3e";
/// The matching closer, when GLM emits one.
const TOOL_CALL_CLOSE_TAG: &str = "\x3c/tool_call\x3e";

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
    /// A complete tag opener was found; the inner values are the byte index of
    /// the opening bracket in the buffer and which grammar matched.
    Full(usize, ToolTagKind),
    /// The buffer ends with bytes that could be the start of a tag opener
    /// (e.g. `[Ba`); the inner value is how many trailing bytes to hold back.
    Partial(usize),
    /// No tag opener is possible in the buffer.
    None,
}

enum TagBodyResult {
    /// The tag is complete (closing `]` for Square, or a terminator for
    /// Angle); `body` holds the tag interior.
    Closed,
    /// Need more input to reach the terminator.
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
/// Scan `buf` for the opening bracket of a tool-call tag (either grammar).
///
/// Returns `Full(pos, kind)` when `buf[pos]` begins a tool-call tag opener —
/// `[` for the square grammar (`[Name key="value"]`) or `<` for the angle
/// grammar (`<Name\nkey: "value"`). The conservative validation in
/// `validate_opener` / `validate_angle_opener` distinguishes tool-call openers
/// from array literals (`[1, 2, 3]`), prose brackets (`[see note]`), and stray
/// `<` / HTML-like tags.
///
/// Returns `Partial(hold)` when the buffer ends with a prefix of such an opener
/// (so a tag split across chunks can still be recognized once more arrives).
/// Returns `None` when no opener is possible.
fn find_tool_tag_open(buf: &str) -> ToolTagMatch {
    let bytes = buf.as_bytes();
    // Track the last bracket of EACH kind that *might* be a partial opener, so
    // we can hold its trailing bytes back when no Full match is found.
    let mut last_square: Option<usize> = None;
    let mut last_angle: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'[' {
            if validate_opener(buf, i) {
                return ToolTagMatch::Full(i, ToolTagKind::Square);
            }
            last_square = Some(i);
        } else if b == b'<' {
            // The HF tool-call wrapper (TOOL_CALL_OPEN_TAG) takes precedence:
            // it also begins with `<` but is a distinct, well-known token.
            if validate_toolcall_opener(buf, i) {
                return ToolTagMatch::Full(i, ToolTagKind::ToolCall);
            }
            if validate_angle_opener(buf, i) {
                return ToolTagMatch::Full(i, ToolTagKind::Angle);
            }
            // Only hold back `<` that could begin an opener — a `<` not
            // followed by an identifier char (e.g. `a < b`, HTML `<br>`) is
            // plain text. validate_angle_opener returning false means it's not
            // a Full match; record it only if it plausibly starts an ident or
            // the tool-call wrapper prefix (`<t...`).
            if bytes.get(i + 1).is_some_and(|c| is_ident_char(*c as char)) {
                last_angle = Some(i);
            }
        }
        i += 1;
    }
    // If the buffer ends with a possible partial opener, hold those bytes back.
    // Prefer the later of the two candidates so we don't emit text that is
    // actually part of a pending opener of the other kind.
    let candidate = match (last_square, last_angle) {
        (Some(s), Some(a)) => Some(if s >= a { (s, b'[') } else { (a, b'<') }),
        (Some(s), None) => Some((s, b'[')),
        (None, Some(a)) => Some((a, b'<')),
        (None, None) => None,
    };
    if let Some((start, _kind)) = candidate {
        let hold = bytes.len() - start;
        if hold <= AnthropicStreamTranslator::TOOL_TAG_MAX_HOLD {
            return ToolTagMatch::Partial(hold);
        }
    } else if bytes.last() == Some(&b'[') {
        return ToolTagMatch::Partial(1);
    }
    ToolTagMatch::None
}

/// How many trailing bytes of `buf` to hold back when no opener matched, in
/// case the buffer ends with the start of an opener that more chunks would
/// complete. Holds a lone trailing `[` (square) or a trailing `<` immediately
/// followed by ident chars (angle); everything else is flushed as text.
fn opener_hold_suffix(buf: &str) -> usize {
    let bytes = buf.as_bytes();
    if bytes.is_empty() {
        return 0;
    }
    // Lone trailing `[` could start a square opener.
    if bytes[bytes.len() - 1] == b'[' {
        return 1;
    }
    // A trailing run of `<` + ident chars (e.g. `<Ag`, `<B`) could start an
    // angle opener once more ident/newline arrives. Hold back the `<` and any
    // following ident chars.
    if let Some(lt_pos) = buf.rfind('<') {
        let tail = &buf[lt_pos + 1..];
        if !tail.is_empty() && tail.chars().all(is_ident_char) {
            return buf.len() - lt_pos;
        }
    }
    0
}

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

/// Validate that `buf[pos] == '<'` begins an ANGLE-grammar tool-call opener.
/// Returns true if it looks like the angle form GLM-5.2 emits, e.g.:
///
/// ```text
/// <Agent
/// description: "..."
/// prompt: "..."
/// ```
///
/// We require: `<`, ≥1 ident char (tool name), then a NEWLINE (the name and
/// its first attribute are on separate lines — this is what distinguishes the
/// tool-call form from HTML tags like `<br>` or `<div class=...>`, where the
/// name is followed by whitespace then an attribute on the SAME line, and
/// from comparison operators like `a < b`). The first attribute line must be a
/// `key:` (colon) pair to confirm it is the angle grammar, not a stray `<`.
fn validate_angle_opener(buf: &str, pos: usize) -> bool {
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
    // The name MUST be immediately followed by a newline (the angle form puts
    // the first attribute on the next line). A same-line attribute means this
    // is an HTML/XML tag, not a tool call — pass through as text.
    match chars.peek() {
        Some(&(_, '\n')) => {
            chars.next();
        }
        Some(&(byte_idx, '\r')) => {
            // Allow CRLF line endings.
            chars.next();
            match chars.peek() {
                Some(&(_, '\n')) => {
                    chars.next();
                }
                _ => {
                    // Lone CR not followed by LF — treat as not-an-opener.
                    let _ = byte_idx;
                    return false;
                }
            }
        }
        _ => return false,
    }
    // Skip blank lines / leading whitespace on the first attribute line.
    let mut saw_key = false;
    while let Some(&(_, c)) = chars.peek() {
        if c.is_whitespace() && c != '\n' && c != '\r' {
            chars.next();
        } else if c == '\n' || c == '\r' {
            // Blank line right after the name — not the angle form.
            return false;
        } else {
            break;
        }
    }
    // First attribute key: at least one identifier char, then ':'.
    while let Some(&(_, c)) = chars.peek() {
        if is_ident_char(c) {
            saw_key = true;
            chars.next();
        } else {
            break;
        }
    }
    if !saw_key {
        return false;
    }
    matches!(chars.peek(), Some(&(_, ':')))
}

/// Validate that `buf[pos..]` begins the HF tool-call wrapper opener
/// (`TOOL_CALL_OPEN_TAG`). The real form GLM-5.2 emits is:
///
/// ```text
/// <Agent
/// description: "..."
/// prompt: "..."
/// ```
///
/// (opener tag, newline, tool name, newline, then `key: "value"` lines). We
/// require: the literal opener tag, a newline, at least one ident char (tool
/// name), a newline, and the first attribute line to start with an ident
/// followed by `:`. This is strict enough to reject HTML tags and stray `<`
/// while matching the wrapper. Returns false (not yet) on a partial prefix so
/// the caller holds the bytes back for the next chunk.
fn validate_toolcall_opener(buf: &str, pos: usize) -> bool {
    let rest = match buf.get(pos..) {
        Some(r) => r,
        None => return false,
    };
    // Must start with the literal opener tag.
    if !rest.starts_with(TOOL_CALL_OPEN_TAG) {
        return false;
    }
    let after_tag = &rest[TOOL_CALL_OPEN_TAG.len()..];
    let mut chars = after_tag.char_indices().peekable();
    // Tool name: at least one identifier character, IMMEDIATELY after the
    // opener tag (GLM emits `<tag>Agent\n…`, with no newline between the tag
    // and the name).
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
    // Newline after the tool name.
    match chars.peek() {
        Some(&(_, '\n')) => {
            chars.next();
        }
        Some(&(_, '\r')) => {
            chars.next();
            match chars.peek() {
                Some(&(_, '\n')) => {
                    chars.next();
                }
                _ => return false,
            }
        }
        _ => return false,
    }
    // Skip leading whitespace on the first attribute line (but not a blank line).
    while let Some(&(_, c)) = chars.peek() {
        if c.is_whitespace() && c != '\n' && c != '\r' {
            chars.next();
        } else {
            break;
        }
    }
    // First attribute key: at least one ident char, then ':'.
    let mut saw_key = false;
    while let Some(&(_, c)) = chars.peek() {
        if is_ident_char(c) {
            saw_key = true;
            chars.next();
        } else {
            break;
        }
    }
    if !saw_key {
        return false;
    }
    matches!(chars.peek(), Some(&(_, ':')))
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

/// Drain `buf` into `body` for the ANGLE grammar (no closing tag). The body is
/// a tool name on the first line followed by `key: "value"` lines. We consume
/// until a terminator is reached:
///   - a BLANK line (two consecutive newlines) — the angle tool block ends and
///     prose may follow, OR
///   - the start of the NEXT opener: a `<` followed by an ident char + newline
///     pattern, or a `[` square opener (validated). When the next opener begins
///     we stop WITHOUT consuming it, so the outer loop re-enters tag mode.
///
/// Because there is no closing bracket, we cannot track depth the way the
/// square grammar does. Instead we scan line-by-line: a line that is a bare
/// `key: "..."` or `key: value` belongs to the tag; a blank line terminates;
/// the start of a new opener terminates. Quoted values may contain newlines, so
/// we honor `in_string` while scanning.
///
/// On entry, the opening `<` and the tool-name line have NOT yet been consumed
/// from `buf` (unlike the square path, which consumes the `[` first). We drain
/// the opener + body together into `body` here. Returns `Closed` once a
/// terminator is reached (terminator NOT consumed unless it's a blank line),
/// or `Pending` when more input is needed.
fn take_angle_tag_body(buf: &mut String, body: &mut String, in_string: &mut bool) -> TagBodyResult {
    let bytes = buf.as_bytes();
    let mut consumed = 0usize;
    let mut i = 0usize;
    let mut consecutive_newlines = 0u32;
    while i < bytes.len() {
        let c = bytes[i] as char;
        // Inside a quoted value: consume verbatim until the closing quote.
        if *in_string {
            body.push(c);
            consumed = i + 1;
            if c == '\\' && i + 1 < bytes.len() {
                // Escaped char — consume together.
                body.push(bytes[i + 1] as char);
                consumed = i + 2;
                i += 2;
                continue;
            }
            if c == '"' {
                *in_string = false;
            }
            consecutive_newlines = 0;
            i += 1;
            continue;
        }
        match c {
            '"' => {
                *in_string = true;
                body.push('"');
                consumed = i + 1;
                consecutive_newlines = 0;
                i += 1;
            }
            '\n' => {
                // A blank line (two newlines with only whitespace between)
                // terminates the angle tool block. Detect: if the PREVIOUS
                // char pushed to body was also a newline (i.e. an empty line),
                // stop here.
                if consecutive_newlines >= 1 {
                    // Blank line terminator. The newline at `i` is the second
                    // of the pair; include it in body so the parser sees the
                    // trailing blank, then stop. Consume through this newline.
                    body.push('\n');
                    consumed = i + 1;
                    let _ = buf.drain(..consumed);
                    return TagBodyResult::Closed;
                }
                body.push('\n');
                consumed = i + 1;
                consecutive_newlines += 1;
                i += 1;
            }
            '<' => {
                // Possible start of the NEXT angle opener. If it validates (or
                // looks like `<Ident\n`), stop BEFORE it so the outer loop
                // re-enters tag mode. Only do this once we already have a body
                // (i.e. past the first opener's name line) — otherwise `<` is
                // part of this tag's value.
                if !body.is_empty() && validate_angle_opener(buf, i) {
                    let _ = buf.drain(..consumed);
                    return TagBodyResult::Closed;
                }
                body.push('<');
                consumed = i + 1;
                consecutive_newlines = 0;
                i += 1;
            }
            '[' => {
                // Possible start of a SQUARE opener interleaved after an angle
                // tool. Stop before it so the outer loop handles it.
                if !body.is_empty() && validate_opener(buf, i) {
                    let _ = buf.drain(..consumed);
                    return TagBodyResult::Closed;
                }
                body.push('[');
                consumed = i + 1;
                consecutive_newlines = 0;
                i += 1;
            }
            _ => {
                // Any non-newline, non-bracket char resets the blank-line
                // counter and is appended.
                body.push(c);
                consumed = i + 1;
                consecutive_newlines = 0;
                i += 1;
            }
        }
    }
    // Reached end of buffer without a terminator. Drain what we consumed and
    // wait for more input.
    let _ = buf.drain(..consumed);
    TagBodyResult::Pending
}

/// Parse the interior of an ANGLE-grammar tool call into a `ParsedToolCall`.
/// The body looks like (the opening `<` is included so we can strip it):
///
/// ```text
/// <Agent
/// description: "Check which client commits..."
/// prompt: "..."
/// tools: "Bash"
/// ```
///
/// Attributes are `key: "value"` (colon separator, quoted values) or
/// `key: value` (bare). The tool name is the identifier immediately after `<`,
/// terminated by a newline. Returns `None` if the body has no name or no
/// `key:` pair.
fn parse_angle_tag_body(body: &str) -> Option<ParsedToolCall> {
    // Strip the leading `<`.
    let body = body.strip_prefix('<')?;
    // Tool name: identifier up to the first newline (or end).
    let mut name_end = 0;
    for (i, c) in body.char_indices() {
        if is_ident_char(c) {
            name_end = i + c.len_utf8();
        } else if c == '\n' || c == '\r' {
            break;
        } else {
            // Non-ident, non-newline right after the name — not the angle form.
            return None;
        }
    }
    if name_end == 0 {
        return None;
    }
    let name = body[..name_end].to_string();
    let rest = &body[name_end..];

    let mut arguments = serde_json::Map::new();
    // Process line by line. Values may be quoted (and quoted values may span
    // newlines), so we walk the whole `rest` with a small state machine rather
    // than splitting on '\n' naively.
    let mut lines = String::new();
    let mut in_string = false;
    let mut chars = rest.chars().peekable();
    while let Some(c) = chars.next() {
        if in_string {
            lines.push(c);
            if c == '\\' {
                if let Some(&next) = chars.peek() {
                    lines.push(next);
                    chars.next();
                }
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                lines.push('"');
            }
            '\n' => {
                // End of a logical line. Try to parse it as `key: value`.
                if let Some((k, v)) = parse_angle_attr_line(&lines) {
                    arguments.insert(k, Value::String(v));
                }
                lines.clear();
            }
            _ => {
                lines.push(c);
            }
        }
    }
    // Trailing line without a newline.
    if !lines.trim().is_empty() {
        if let Some((k, v)) = parse_angle_attr_line(&lines) {
            arguments.insert(k, Value::String(v));
        }
    }

    if arguments.is_empty() {
        return None;
    }
    Some(ParsedToolCall { name, arguments })
}

/// Parse a single `key: "value"` or `key: value` attribute line (the colon
/// form used by the angle grammar). Returns `None` for lines that aren't an
/// attribute (e.g. a stray blank or prose), so the caller can skip them.
fn parse_angle_attr_line(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Key: leading identifier.
    let key_end = trimmed
        .find(|c: char| !is_ident_char(c))
        .unwrap_or(trimmed.len());
    if key_end == 0 {
        return None;
    }
    let key = trimmed[..key_end].to_string();
    let after_key = trimmed[key_end..].trim_start();
    // Expect ':'.
    let after_colon = after_key.strip_prefix(':')?;
    let value_raw = after_colon.trim();
    // Quoted value: strip surrounding quotes and unescape \" \\ \n \t.
    let value = if value_raw.starts_with('"') && value_raw.ends_with('"') && value_raw.len() >= 2 {
        let inner = &value_raw[1..value_raw.len() - 1];
        unquote_angle_value(inner)
    } else {
        // Bare value (no quotes) — take as-is.
        value_raw.to_string()
    };
    Some((key, value))
}

/// Unescape a quoted angle-grammar value: `\"` → `"`, `\\` → `\`, `\n` →
/// newline, `\t` → tab. Other backslash sequences are preserved literally.
fn unquote_angle_value(inner: &str) -> String {
    let mut out = String::new();
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Drain `buf` into `body` for the ToolCall (HF wrapper) grammar. The opener
/// tag has already been matched but NOT consumed (it stays at the front of
/// `buf`); we drain the opener + tool name + attribute lines into `body`.
///
/// Terminators (return `Closed`):
///   - the close tag `TOOL_CALL_CLOSE_TAG` (consumed through it), OR
///   - a BLANK line (two consecutive newlines) — consumed through the second
///     newline, OR
///   - the start of the NEXT opener (`<`-wrapper or `[`-square, validated) —
///     NOT consumed, so the outer loop re-enters tag mode.
///
/// Quoted values may contain newlines, so we honor `in_string` while scanning.
/// Returns `Pending` when more input is needed.
fn take_toolcall_body(buf: &mut String, body: &mut String, in_string: &mut bool) -> TagBodyResult {
    let bytes = buf.as_bytes();
    let mut consumed = 0usize;
    let mut i = 0usize;
    let mut consecutive_newlines = 0u32;
    while i < bytes.len() {
        let c = bytes[i] as char;
        // Inside a quoted value: consume verbatim until the closing quote.
        if *in_string {
            body.push(c);
            consumed = i + 1;
            if c == '\\' && i + 1 < bytes.len() {
                body.push(bytes[i + 1] as char);
                consumed = i + 2;
                i += 2;
                continue;
            }
            if c == '"' {
                *in_string = false;
            }
            consecutive_newlines = 0;
            i += 1;
            continue;
        }
        // Check for the close tag at this position (only meaningful once we
        // have a body past the opener + name).
        if !body.is_empty() && buf[i..].starts_with(TOOL_CALL_CLOSE_TAG) {
            let end = i + TOOL_CALL_CLOSE_TAG.len();
            body.push_str(TOOL_CALL_CLOSE_TAG);
            consumed = end;
            let _ = buf.drain(..consumed);
            return TagBodyResult::Closed;
        }
        match c {
            '"' => {
                *in_string = true;
                body.push('"');
                consumed = i + 1;
                consecutive_newlines = 0;
                i += 1;
            }
            '\n' => {
                if consecutive_newlines >= 1 {
                    // Blank line terminator.
                    body.push('\n');
                    consumed = i + 1;
                    let _ = buf.drain(..consumed);
                    return TagBodyResult::Closed;
                }
                body.push('\n');
                consumed = i + 1;
                consecutive_newlines += 1;
                i += 1;
            }
            '<' => {
                // Possible start of the NEXT wrapper opener. Stop before it so
                // the outer loop re-enters tag mode. (Only once we have a body
                // past the opener itself.)
                if !body.is_empty()
                    && buf[i..].starts_with(TOOL_CALL_OPEN_TAG)
                    && validate_toolcall_opener(buf, i)
                {
                    let _ = buf.drain(..consumed);
                    return TagBodyResult::Closed;
                }
                body.push('<');
                consumed = i + 1;
                consecutive_newlines = 0;
                i += 1;
            }
            '[' => {
                if !body.is_empty() && validate_opener(buf, i) {
                    let _ = buf.drain(..consumed);
                    return TagBodyResult::Closed;
                }
                body.push('[');
                consumed = i + 1;
                consecutive_newlines = 0;
                i += 1;
            }
            _ => {
                body.push(c);
                consumed = i + 1;
                consecutive_newlines = 0;
                i += 1;
            }
        }
    }
    let _ = buf.drain(..consumed);
    TagBodyResult::Pending
}

/// Parse the body of a ToolCall (HF wrapper) grammar tag into a `ParsedToolCall`.
/// The body looks like (opener tag included):
///
/// ```text
/// <Agent
/// description: "..."
/// prompt: "..."
/// ```
///
/// (possibly with a trailing close tag). Strip the opener, read the tool name
/// from the first line, then parse each remaining line as a `key: "value"` (or
/// `key: value`) attribute via `parse_angle_attr_line`. Returns `None` if the
/// body has no name or no attribute.
fn parse_toolcall_body(body: &str) -> Option<ParsedToolCall> {
    // Strip the opener tag.
    let body = body.strip_prefix(TOOL_CALL_OPEN_TAG)?;
    // Strip a trailing close tag if present.
    let body = body.strip_suffix(TOOL_CALL_CLOSE_TAG).unwrap_or(body);
    // First line is the tool name.
    let nl = body.find('\n')?;
    let name = body[..nl].trim();
    if name.is_empty() || !name.chars().next().is_some_and(is_ident_char) {
        return None;
    }
    let name = name.to_string();
    let rest = &body[nl + 1..];

    let mut arguments = serde_json::Map::new();
    // Walk line by line, honoring quoted strings (values may span newlines).
    let mut lines = String::new();
    let mut in_string = false;
    let mut chars = rest.chars().peekable();
    while let Some(c) = chars.next() {
        if in_string {
            lines.push(c);
            if c == '\\' {
                if let Some(&next) = chars.peek() {
                    lines.push(next);
                    chars.next();
                }
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                lines.push('"');
            }
            '\n' => {
                if let Some((k, v)) = parse_angle_attr_line(&lines) {
                    arguments.insert(k, Value::String(v));
                }
                lines.clear();
            }
            _ => {
                lines.push(c);
            }
        }
    }
    if !lines.trim().is_empty() {
        if let Some((k, v)) = parse_angle_attr_line(&lines) {
            arguments.insert(k, Value::String(v));
        }
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
            text_chars: 0,
            text_preview: String::new(),
            raw_content: String::new(),
            raw_chars: 0,
            recovered_non_streaming: false,
            text_buffer: String::new(),
            in_tool_tag: false,
            tool_tag_body: String::new(),
            tool_tag_depth: 0,
            in_tool_tag_string: false,
            inline_tool_emitted: false,
            tool_tag_kind: ToolTagKind::default(),
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
                &format!(
                    "[usage] raw vLLM usage chunk: {}",
                    serde_json::to_string(usage).unwrap_or_default()
                ),
            );
            // vLLM streams usage in two chunks:
            //   1. Real usage (prompt_tokens > 0) — after finish_reason
            //   2. Zero-usage cleanup chunk — on connection close
            // Never overwrite real usage with zeros — the cleanup chunk
            // would reset token counts to 0, breaking the ACP usage UI.
            let pt = usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let ct = usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
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
                                events
                                    .push(self.content_block_stop_event(self.current_block_index));
                                self.current_block_index += 1;
                                self.thinking_block_open = false;
                            }
                            if !self.text_block_open {
                                events.push(self.text_block_start_event(self.current_block_index));
                                self.text_block_open = true;
                            }
                            self.text_ever_emitted = true;
                            // Record the raw upstream text before the scanner
                            // touches it (turn-summary diagnostic).
                            self.raw_chars += content.len();
                            const RAW_MAX: usize = 1200;
                            if self.raw_content.len() < RAW_MAX {
                                let room = RAW_MAX - self.raw_content.len();
                                let take = content
                                    .char_indices()
                                    .map(|(i, _)| i)
                                    .chain(std::iter::once(content.len()))
                                    .take_while(|i| *i <= room)
                                    .last()
                                    .unwrap_or(0);
                                self.raw_content.push_str(&content[..take]);
                            }
                            events.extend(self.handle_text_content(content));
                        }
                    }

                    // Tool calls (buffer arguments)
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tool_calls {
                            let tc_index =
                                tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            while self.pending_tool_calls.len() <= tc_index {
                                self.pending_tool_calls.push(PendingToolCall::default());
                            }
                            let pending = &mut self.pending_tool_calls[tc_index];
                            pending.index = tc_index;
                            if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                if !id.is_empty() {
                                    pending.id = id.to_string();
                                }
                            }
                            if let Some(func) = tc.get("function") {
                                if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                    if !name.is_empty() {
                                        pending.name = name.to_string();
                                    }
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
                            events.push(self.tool_use_block_start_event(
                                block_index,
                                &pending.id,
                                &pending.name,
                            ));
                            if !pending.arguments_buffer.is_empty() {
                                // Drop the zero-argument workaround property so
                                // the client's schema validation still passes.
                                let args = strip_placeholder_arg(&pending.arguments_buffer);
                                events.push(self.tool_use_delta_event(block_index, &args));
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
                            if self.inline_tool_emitted {
                                "tool_use"
                            } else {
                                "end_turn"
                            }
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

    /// True when this turn looks like the backend silently swallowed a tool
    /// call, so the caller should re-run the request non-streaming.
    ///
    /// GLM-5.2-FP8 on vLLM drops tool calls whose arguments object is empty
    /// when `stream: true`: the turn arrives as `finish_reason: "stop"` with no
    /// `tool_calls` and at most a short preamble ("Now let me list them."), and
    /// sometimes with output tokens but *no* content at all (the parser ate the
    /// tool-call markup without producing a tool call). The identical request
    /// with `stream: false` returns the tool call every time, so a non-streaming
    /// re-run recovers the turn. Verified against the backend with no proxy in
    /// the path; `NOARGS_PLACEHOLDER` alone is not enough because the model
    /// often still emits `{}` despite the schema.
    ///
    /// `max_text` bounds how much assistant text may already have been streamed.
    /// A genuine final answer is long, so it stays well clear of the retry and
    /// costs nothing; the stalled turns measured 0–110 characters.
    pub fn should_retry_non_streaming(&self, max_text: usize) -> bool {
        let has_tool_use = self.pending_tool_calls.iter().any(|t| !t.name.is_empty());
        self.stop_reason == "end_turn" && !has_tool_use && self.text_chars <= max_text
    }

    /// Splice tool calls from a non-streaming response into this stream as
    /// Anthropic `tool_use` blocks and flip the turn to `stop_reason: tool_use`.
    ///
    /// Returns the SSE events to forward. Must be called after the upstream
    /// stream's `finish_reason` has been processed (so text/thinking blocks are
    /// closed and `current_block_index` is correct) and before `finish()`.
    /// Any text already streamed is kept — it is the model's own preamble for
    /// the action, and the non-streaming re-run returns empty content.
    pub fn inject_tool_calls_from_message(&mut self, message: &Value) -> Vec<String> {
        let mut events = Vec::new();
        let Some(tool_calls) = message.get("tool_calls").and_then(|c| c.as_array()) else {
            return events;
        };
        for tc in tool_calls {
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if name.is_empty() {
                continue;
            }
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()));
            let raw_args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let args = strip_placeholder_arg(raw_args);

            let block_index = self.current_block_index;
            events.push(self.tool_use_block_start_event(block_index, &id, name));
            events.push(self.tool_use_delta_event(block_index, &args));
            events.push(self.content_block_stop_event(block_index));
            self.current_block_index += 1;

            // Record it so finish() sees a tool_use (skipping the empty-text
            // fallback) and reports it in the turn summary. `emitted` keeps the
            // finish_reason handler from emitting the block a second time.
            self.pending_tool_calls.push(PendingToolCall {
                index: self.pending_tool_calls.len(),
                id,
                name: name.to_string(),
                arguments_buffer: args,
                emitted: true,
            });
        }
        if !events.is_empty() {
            // Force stop_reason=tool_use: the upstream said "stop", but the turn
            // now carries a tool call the client must execute.
            self.stop_reason = "tool_use".to_string();
            self.recovered_non_streaming = true;
        }
        events
    }

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
            let reasoning = std::mem::take(&mut self.reasoning_text);
            events.push(self.text_delta_event(self.current_block_index, &reasoning));
            self.reasoning_text = reasoning;
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
        let has_tool_use = self.pending_tool_calls.iter().any(|t| !t.name.is_empty());
        if !self.text_ever_emitted && self.reasoning_text.is_empty() && !has_tool_use {
            events.push(self.text_block_start_event(self.current_block_index));
            events.push(self.content_block_stop_event(self.current_block_index));
            self.current_block_index += 1;
        }
        // Emit message_delta now — all usage chunks have been processed.
        // flush_text_buffer() above may have emitted an inline tool call (e.g.
        // an angle-grammar tag with no terminator that only resolved at stream
        // end) and set inline_tool_emitted AFTER stop_reason was locked in when
        // finish_reason arrived. Re-evaluate: if an inline tool was emitted but
        // stop_reason defaulted to end_turn, override to tool_use so the client
        // executes the tool.
        let sr = if self.inline_tool_emitted
            && (self.stop_reason.is_empty() || self.stop_reason == "end_turn")
        {
            "tool_use"
        } else if self.stop_reason.is_empty() {
            "end_turn"
        } else {
            self.stop_reason.as_str()
        };
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
                // Looking for the start of a tool-call tag: either grammar.
                match find_tool_tag_open(&self.text_buffer) {
                    ToolTagMatch::Full(start, kind) => {
                        // Emit any plain text before the tag as text_delta.
                        if start > 0 {
                            let before: String = self.text_buffer.drain(..start).collect();
                            if !before.is_empty() {
                                out.push(self.text_delta_event(self.current_block_index, &before));
                            }
                        }
                        // Enter tag mode. For the SQUARE grammar the opening `[`
                        // is consumed now (take_tool_tag_body expects the body
                        // without it). For the ANGLE grammar the opening `<` is
                        // LEFT in the buffer — take_angle_tag_body drains the
                        // opener + body together, and parse_angle_tag_body
                        // strips the leading `<`.
                        match kind {
                            ToolTagKind::Square => {
                                self.text_buffer.drain(..1);
                            }
                            ToolTagKind::Angle => {
                                // `<` stays in text_buffer; take_angle_tag_body
                                // consumes it into `tool_tag_body`.
                            }
                            ToolTagKind::ToolCall => {
                                // The wrapper opener tag stays in text_buffer;
                                // take_toolcall_body drains it into `tool_tag_body`.
                            }
                        }
                        self.in_tool_tag = true;
                        self.tool_tag_kind = kind;
                        self.tool_tag_body.clear();
                        self.tool_tag_depth = 0;
                        self.in_tool_tag_string = false;
                        continue;
                    }
                    ToolTagMatch::Partial(hold) => {
                        // The buffer ends with a possible partial tag opening
                        // (e.g. `[Ba` or `<Ag`). Emit everything before it, hold
                        // the rest for the next chunk.
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
                        // could be the start of an opener (a lone trailing `[`
                        // or `<` followed by an ident char).
                        let hold = opener_hold_suffix(&self.text_buffer);
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

            // Inside a tool-call tag: drain the body. Dispatch on the grammar
            // detected at opener time.
            match self.tool_tag_kind {
                ToolTagKind::Square => {
                    // SQUARE grammar: body is `Name key="value" ...`; closing
                    // `]` at bracket depth 0 ends it. The opening `[` was
                    // already consumed; body accumulates the interior.
                    match take_tool_tag_body(
                        &mut self.text_buffer,
                        &mut self.tool_tag_body,
                        &mut self.tool_tag_depth,
                        &mut self.in_tool_tag_string,
                    ) {
                        TagBodyResult::Closed => {
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
                ToolTagKind::Angle => {
                    // ANGLE grammar: body is `<Name\nkey: "value"\n…`; there is
                    // NO closing bracket. take_angle_tag_body drains the opener
                    // + body together into `tool_tag_body` and returns Closed at
                    // a terminator (blank line / next opener / stream end
                    // handled by flush).
                    match take_angle_tag_body(
                        &mut self.text_buffer,
                        &mut self.tool_tag_body,
                        &mut self.in_tool_tag_string,
                    ) {
                        TagBodyResult::Closed => {
                            if let Some(tc) = parse_angle_tag_body(&self.tool_tag_body) {
                                out.extend(self.emit_inline_tool_use(&tc));
                            } else {
                                // Matched the conservative opener but body was
                                // malformed. Emit the accumulated text verbatim
                                // (it already includes the `<`).
                                let raw = self.tool_tag_body.clone();
                                out.push(self.text_delta_event(self.current_block_index, &raw));
                            }
                            self.in_tool_tag = false;
                            self.tool_tag_body.clear();
                            continue;
                        }
                        TagBodyResult::Pending => {
                            // Need more chunks to reach a terminator.
                            return out;
                        }
                    }
                }
                ToolTagKind::ToolCall => {
                    // TOOLCALL (HF wrapper) grammar: body is the opener tag +
                    // newline + tool name + `key: "value"` lines. May be closed
                    // by the close tag, a blank line, the next opener, or
                    // stream end (handled by flush).
                    match take_toolcall_body(
                        &mut self.text_buffer,
                        &mut self.tool_tag_body,
                        &mut self.in_tool_tag_string,
                    ) {
                        TagBodyResult::Closed => {
                            if let Some(tc) = parse_toolcall_body(&self.tool_tag_body) {
                                out.extend(self.emit_inline_tool_use(&tc));
                            } else {
                                // Matched the opener but body was malformed.
                                // Emit the accumulated text verbatim (it already
                                // includes the opener tag).
                                let raw = self.tool_tag_body.clone();
                                out.push(self.text_delta_event(self.current_block_index, &raw));
                            }
                            self.in_tool_tag = false;
                            self.tool_tag_body.clear();
                            continue;
                        }
                        TagBodyResult::Pending => {
                            // Need more chunks to reach a terminator.
                            return out;
                        }
                    }
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
        let id = format!(
            "chatcmpl-tool-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..16]
        );
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
            match self.tool_tag_kind {
                ToolTagKind::Square => {
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
                }
                ToolTagKind::Angle => {
                    // The angle form has NO closing tag, so reaching stream end
                    // inside an angle tag is NORMAL — GLM emits `<Name\nkey:
                    // "value"\n…` and stops. Combine the drained body with any
                    // remaining buffer and try to parse it as a tool call. The
                    // opener `<` was left in the buffer and drained into
                    // `tool_tag_body` by take_angle_tag_body, so the body
                    // already starts with `<`.
                    let mut body = std::mem::take(&mut self.tool_tag_body);
                    body.push_str(&self.text_buffer);
                    self.text_buffer.clear();
                    self.in_tool_tag = false;
                    self.in_tool_tag_string = false;
                    if let Some(tc) = parse_angle_tag_body(&body) {
                        out.extend(self.emit_inline_tool_use(&tc));
                    } else if !body.is_empty() {
                        // Not parseable as a tool call — emit verbatim so
                        // nothing is silently dropped.
                        if !self.text_block_open {
                            out.push(self.text_block_start_event(self.current_block_index));
                            self.text_block_open = true;
                            self.text_ever_emitted = true;
                        }
                        out.push(self.text_delta_event(self.current_block_index, &body));
                    }
                }
                ToolTagKind::ToolCall => {
                    // The HF wrapper form is frequently unclosed (GLM emits the
                    // opener + attrs and stops), so reaching stream end inside a
                    // ToolCall tag is NORMAL. Combine the drained body with any
                    // remaining buffer and try to parse it. The opener tag was
                    // drained into `tool_tag_body` by take_toolcall_body, so the
                    // body already starts with the opener.
                    let mut body = std::mem::take(&mut self.tool_tag_body);
                    body.push_str(&self.text_buffer);
                    self.text_buffer.clear();
                    self.in_tool_tag = false;
                    self.in_tool_tag_string = false;
                    if let Some(tc) = parse_toolcall_body(&body) {
                        out.extend(self.emit_inline_tool_use(&tc));
                    } else if !body.is_empty() {
                        if !self.text_block_open {
                            out.push(self.text_block_start_event(self.current_block_index));
                            self.text_block_open = true;
                            self.text_ever_emitted = true;
                        }
                        out.push(self.text_delta_event(self.current_block_index, &body));
                    }
                }
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

    fn text_delta_event(&mut self, index: i32, text: &str) -> String {
        // Record for the turn-summary diagnostic in finish().
        self.text_chars += text.len();
        const PREVIEW_MAX: usize = 200;
        if self.text_preview.len() < PREVIEW_MAX {
            let room = PREVIEW_MAX - self.text_preview.len();
            let take = text
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(text.len()))
                .take_while(|i| *i <= room)
                .last()
                .unwrap_or(0);
            self.text_preview.push_str(&text[..take]);
        }
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
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let args_str = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("{}");
                        let mut input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                        // Drop the zero-argument workaround property so the
                        // client's schema validation still passes.
                        if let Some(obj) = input.as_object_mut() {
                            obj.remove(NOARGS_PLACEHOLDER);
                        }
                        content.push(
                            json!({"type": "tool_use", "id": id, "name": name, "input": input}),
                        );
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
        assert!(
            combined.contains("I'll check that now."),
            "preceding text should pass through, got: {combined}"
        );
        assert!(
            !combined.contains("[Bash"),
            "tool tag should NOT leak into text, got: {combined}"
        );
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
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Bash")
        );
        assert_eq!(
            tool_uses[1].get("name").and_then(|n| n.as_str()),
            Some("Bash")
        );
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
        assert_eq!(
            tool_uses.len(),
            0,
            "non-tool brackets must not become tool_use"
        );
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
        assert!(
            text.contains("[1, 2, 3]"),
            "array literal should pass through, got: {text}"
        );
        assert!(
            text.contains("[see notes]"),
            "prose bracket should pass through, got: {text}"
        );
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
        assert_eq!(
            tool_uses.len(),
            0,
            "unclosed tag should not yield a tool_use"
        );
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
        assert!(
            text.contains("[Bash"),
            "unclosed tag text should pass through, got: {text}"
        );
    }

    // ── Angle-bracket-colon grammar tests ────────────────────────────────
    //
    // GLM-5.2 also emits tool calls in the ANGLE form, e.g.:
    //
    //     <Agent
    //     description: "Check which client commits..."
    //     prompt: "..."
    //     tools: "Bash"
    //
    // (angle bracket + colon separator, quoted values, NO closing tag). These
    // tests mirror the square-bracket suite for the angle grammar and cover the
    // behaviors unique to it: no closing bracket, stream-end flush, blank-line
    // terminator, next-opener terminator, and HTML-tag pass-through.

    /// A complete angle-grammar tool call (terminated by a blank line) emitted
    /// as a single content delta should convert to a tool_use block.
    #[test]
    fn test_angle_tool_call_single_chunk() {
        // Use a literal newline in the JSON content via \n. The blank line
        // (two newlines) after the last attribute terminates the tag.
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"<Agent\ndescription: \"Check commits\"\nprompt: \"find recent client commits\"\ntools: \"Bash\"\n\n"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(
            tool_uses.len(),
            1,
            "expected one tool_use block, got events: {:?}",
            events
        );
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Agent")
        );
        // The tool_use block is at index 1 (after the text block opened for
        // preceding/following prose). Find its input_json_delta.
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["description"], "Check commits");
        assert_eq!(parsed["prompt"], "find recent client commits");
        assert_eq!(parsed["tools"], "Bash");

        // stop_reason must be tool_use even though finish_reason was "stop".
        let delta = find_message_delta(&events).expect("should have message_delta");
        let stop_reason = delta
            .get("delta")
            .and_then(|d| d.get("stop_reason"))
            .and_then(|s| s.as_str());
        assert_eq!(stop_reason, Some("tool_use"));
    }

    /// An angle tool call split across many SSE chunks must still be
    /// reassembled — including splits inside the quoted value and across the
    /// terminating blank line.
    #[test]
    fn test_angle_tool_call_split_across_chunks() {
        let full = "<Agent\ndescription: \"Check commits\"\nprompt: \"find recent client commits\"\ntools: \"Bash\"\n\n";
        let slices: &[&str] = &[
            "<Agent\ndescrip",
            "tion: \"Check commits\"\nprom",
            "pt: \"find recent client com",
            "mits\"\ntools: \"Bash\"\n",
            "\n",
        ];
        assert_eq!(slices.concat(), full);
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
        assert_eq!(
            tool_uses.len(),
            1,
            "split angle tag should yield one tool_use"
        );
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Agent")
        );
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["description"], "Check commits");
        assert_eq!(parsed["prompt"], "find recent client commits");
        assert_eq!(parsed["tools"], "Bash");
    }

    /// The angle form has NO closing tag, so a tag that runs to stream end
    /// (the model stops after the last attribute) must be flushed and parsed at
    /// finish() — NOT dropped or emitted as verbatim text. This is the
    /// realistic case for the end_turn-prematurely bug.
    #[test]
    fn test_angle_tool_call_no_terminator_at_stream_end() {
        // No trailing blank line — the stream just ends after the last attr.
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"<Agent\ndescription: \"Check commits\"\nprompt: \"find recent client commits\""},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(
            tool_uses.len(),
            1,
            "angle tag at stream end should flush as a tool_use, got events: {:?}",
            events
        );
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Agent")
        );
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["description"], "Check commits");
        assert_eq!(parsed["prompt"], "find recent client commits");

        let delta = find_message_delta(&events).expect("should have message_delta");
        let stop_reason = delta
            .get("delta")
            .and_then(|d| d.get("stop_reason"))
            .and_then(|s| s.as_str());
        assert_eq!(stop_reason, Some("tool_use"));
    }

    /// Prose before an angle tool call should pass through as text, and the tag
    /// should NOT leak into the text stream.
    #[test]
    fn test_angle_tool_call_with_preceding_text() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"I'll investigate that now.\n<Agent\ndescription: \"Check commits\"\ntools: \"Bash\"\n\n"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 1);
        let text: String = events
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
        assert!(
            text.contains("I'll investigate that now."),
            "preceding text should pass through, got: {text}"
        );
        assert!(
            !text.contains("<Agent"),
            "angle tag should NOT leak into text, got: {text}"
        );
    }

    /// Two consecutive angle tool calls (each terminated by a blank line) in
    /// one content delta should each become a tool_use block.
    #[test]
    fn test_angle_tool_call_multiple() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"<Agent\ndescription: \"first\"\ntools: \"Bash\"\n\n<Agent\ndescription: \"second\"\ntools: \"Bash\"\n\n"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 2, "expected two tool_use blocks");
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Agent")
        );
        assert_eq!(
            tool_uses[1].get("name").and_then(|n| n.as_str()),
            Some("Agent")
        );
        let j1 = input_json_for(&events, 1).expect("first tool input");
        let p1: Value = serde_json::from_str(&j1).unwrap();
        assert_eq!(p1["description"], "first");
        // Second tool_use block: index advances past the first's text reopen.
        // Find whichever input_json_delta decodes to description=second.
        let mut found_second = false;
        for idx in 2..6 {
            if let Some(j) = input_json_for(&events, idx) {
                if let Ok(p) = serde_json::from_str::<Value>(&j) {
                    if p.get("description").and_then(|d| d.as_str()) == Some("second") {
                        found_second = true;
                        break;
                    }
                }
            }
        }
        assert!(
            found_second,
            "second angle tool call should have its input_json"
        );
    }

    /// An angle tool call whose value spans multiple lines (newlines inside the
    /// quoted string) must parse with the embedded newlines preserved.
    #[test]
    fn test_angle_tool_call_multiline_value() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"<Agent\ndescription: \"Check commits\"\nprompt: \"git status\ngit log -1\"\ntools: \"Bash\"\n\n"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 1);
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["prompt"], "git status\ngit log -1");
        assert_eq!(parsed["description"], "Check commits");
    }

    /// A `<` that begins an HTML/XML-style tag (name followed by whitespace
    /// and an attribute on the SAME line, or `</close>`, or a comparison like
    /// `a < b`) must NOT be treated as an angle tool call — it should pass
    /// through as text. The angle grammar requires the name to be followed by
    /// a newline and a `key:` line.
    #[test]
    fn test_html_tag_passes_through_as_text() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"See <div class=\"x\"> and a < b and <br/> here"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(
            tool_uses.len(),
            0,
            "HTML/comparison < must not become a tool_use"
        );
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
        assert!(
            text.contains("<div"),
            "HTML tag should pass through, got: {text}"
        );
        assert!(
            text.contains("a < b"),
            "comparison should pass through, got: {text}"
        );
        assert!(
            text.contains("<br/>"),
            "self-closing tag should pass through, got: {text}"
        );
    }

    /// A lone `<` or a `<Word` prefix at stream end (no newline + key: yet)
    /// must be held back, not emitted prematurely — and when the stream ends
    /// without completing an opener, it flushes as text.
    #[test]
    fn test_angle_partial_opener_flushed_as_text() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"result is <Ag"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(
            tool_uses.len(),
            0,
            "incomplete opener must not yield a tool_use"
        );
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
        assert!(
            text.contains("<Ag"),
            "partial opener should flush as text, got: {text}"
        );
    }

    /// A square-bracket tool call followed by an angle tool call (both grammars
    /// in one stream) must each be parsed by the correct grammar.
    #[test]
    fn test_mixed_square_and_angle_tool_calls() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"content":"[Bash command=\"git status\" description=\"Status\"]\n<Agent\ndescription: \"follow-up\"\ntools: \"Bash\"\n\n"},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(
            tool_uses.len(),
            2,
            "expected one square + one angle tool_use"
        );
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Bash")
        );
        assert_eq!(
            tool_uses[1].get("name").and_then(|n| n.as_str()),
            Some("Agent")
        );
    }

    // ── HF tool-call-wrapper grammar (the REAL GLM-5.2 leak) ─────────────
    //
    // GLM-5.2 emits tool calls in the HuggingFace/Qwen chat-template wrapper:
    // the literal opener tag (angle-bracket `tool_call` token), a newline, the
    // tool name, then `key: "value"` lines. Frequently UNCLOSED (the model
    // stops after the last attribute). This is the form that caused the
    // "after compaction I said something, it didn't call tools and reached
    // end_turn prematurely" bug in claude-code-acp. These tests use the EXACT
    // bytes captured from a live transcript (60867dfe…) to verify the parser
    // converts them to tool_use + stop_reason=tool_use.

    /// Build the opener tag without a literal angle-bracket-tool token in the
    /// source (keeps editors/tooling from mis-parsing the test file).
    fn tc_open() -> String {
        format!("{}tool_call{}", "\x3c", "\x3e")
    }
    fn tc_close() -> String {
        format!("{}/tool_call{}", "\x3c", "\x3e")
    }

    /// The exact leaked content from transcript 60867dfe line 3959 (an
    /// UNCLOSED wrapper tool call) should convert to a tool_use block, not leak
    /// as text + end_turn.
    #[test]
    fn test_real_leak_unclosed_wrapper() {
        let open = tc_open();
        // Reconstructed from the transcript: prose, then the wrapper opener,
        // tool name "Agent", then colon attributes, NO close tag, NO blank line.
        let content = format!(
            "I'll launch two parallel investigations: (1) check which client commits touch the ACP path, and (2) trace the context-usage growth + post-compaction 50% residual.{}Agent\ndescription: \"Check which client commits touch ACP path vs run path\"\nprompt: \"Check these 7 client commits and tell me which touch the ACP server path (src/acp/acp-server.ts) vs the run CI path (src/cli-run/*). Commits: 0cf8310, 51ee75b, 37e4da8.\"\ntools: \"Bash\"\nagent: \"Explore\"\ntool_input: \"Explore\"",
            open
        );
        let content_json = serde_json::to_string(&content).unwrap();
        let line1 = format!(
            r#"data: {{"id":"1","model":"test","choices":[{{"index":0,"delta":{{"content":{}}}}}]}}"#,
            content_json
        );
        let events = run_stream(&[
            line1.as_str(),
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(
            tool_uses.len(),
            1,
            "real unclosed wrapper leak should yield one tool_use, got events: {:?}",
            events
        );
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Agent")
        );
        // Find the input_json_delta for the tool_use block.
        let mut json = None;
        for idx in 0..6 {
            if let Some(j) = input_json_for(&events, idx) {
                if serde_json::from_str::<Value>(&j).ok().and_then(|v| {
                    v.get("description")
                        .and_then(|d| d.as_str())
                        .map(|s| s.to_string())
                }) == Some("Check which client commits touch ACP path vs run path".to_string())
                {
                    json = Some(j);
                    break;
                }
            }
        }
        let json = json.expect("should find the tool_use input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["tools"], "Bash");
        assert_eq!(parsed["agent"], "Explore");
        assert_eq!(parsed["tool_input"], "Explore");

        // stop_reason MUST be tool_use (this was the bug: it was end_turn).
        let delta = find_message_delta(&events).expect("should have message_delta");
        let stop_reason = delta
            .get("delta")
            .and_then(|d| d.get("stop_reason"))
            .and_then(|s| s.as_str());
        assert_eq!(
            stop_reason,
            Some("tool_use"),
            "unclosed wrapper leak must force stop_reason=tool_use"
        );
    }

    /// The exact leaked content from transcript 60867dfe line 3970 — a CLOSED
    /// wrapper tool call (with the close tag) for a Bash command.
    #[test]
    fn test_real_leak_closed_wrapper() {
        let open = tc_open();
        let close = tc_close();
        let content = format!(
            "{}Bash\ncommand: \"git show --stat 0cf8310 | head -20\"\ndescription: \"Show files changed in commit\"{}",
            open, close
        );
        let content_json = serde_json::to_string(&content).unwrap();
        let line1 = format!(
            r#"data: {{"id":"1","model":"test","choices":[{{"index":0,"delta":{{"content":{}}}}}]}}"#,
            content_json
        );
        let events = run_stream(&[
            line1.as_str(),
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"data: {"id":"1","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
            "data: [DONE]",
        ]);

        let tool_uses = collect_tool_use(&events);
        assert_eq!(
            tool_uses.len(),
            1,
            "closed wrapper leak should yield one tool_use"
        );
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Bash")
        );
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["command"], "git show --stat 0cf8310 | head -20");
        assert_eq!(parsed["description"], "Show files changed in commit");

        let delta = find_message_delta(&events).expect("should have message_delta");
        let stop_reason = delta
            .get("delta")
            .and_then(|d| d.get("stop_reason"))
            .and_then(|s| s.as_str());
        assert_eq!(stop_reason, Some("tool_use"));
    }

    /// A wrapper tool call split across many SSE chunks (including a split
    /// right after the opener tag, before the tool name) must reassemble.
    #[test]
    fn test_wrapper_tool_call_split_across_chunks() {
        let open = tc_open();
        let full = format!(
            "{}Agent\ndescription: \"Check commits\"\nprompt: \"find recent client commits\"\ntools: \"Bash\"\n\n",
            open
        );
        // Split awkwardly: opener tag alone, then name+attrs in pieces.
        let open_len = open.len();
        let slices: Vec<String> = vec![
            full[..open_len].to_string(),             // just the opener tag
            full[open_len..open_len + 7].to_string(), // "Agent\n"
            full[open_len + 7..].to_string(),         // rest
        ];
        assert_eq!(slices.concat(), full);
        let mut lines: Vec<String> = Vec::new();
        for slice in &slices {
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
        assert_eq!(
            tool_uses.len(),
            1,
            "split wrapper tag should yield one tool_use"
        );
        assert_eq!(
            tool_uses[0].get("name").and_then(|n| n.as_str()),
            Some("Agent")
        );
        let json = input_json_for(&events, 1).expect("should have input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("input_json should be valid JSON");
        assert_eq!(parsed["description"], "Check commits");
        assert_eq!(parsed["tools"], "Bash");
    }

    // ── Zero-argument tool workaround (NOARGS_PLACEHOLDER) ─────────────────
    //
    // GLM-5.2-FP8 on vLLM drops any streamed tool call whose arguments object
    // is empty: the turn returns finish_reason="stop" with no tool_calls and
    // only the model's preamble, so the client sees a turn that did nothing.
    // Reproduced against the backend directly (no proxy), and only for GLM —
    // DeepSeek-V4 streams empty-argument tool calls fine.

    /// A tool that declares no properties gets the placeholder for GLM, so the
    /// model emits `{"__noargs": ""}` instead of the dropped `{}`.
    #[test]
    fn test_noargs_tool_gets_placeholder_for_glm() {
        let body = json!({
            "messages": [],
            "tools": [{"name": "TaskList", "description": "List tasks", "input_schema": {"type": "object", "properties": {}, "required": []}}]
        });
        let out = anthropic_to_chat_completions("zai-org/GLM-5.2-FP8", &body, true);
        let props = &out["tools"][0]["function"]["parameters"]["properties"];
        assert!(
            props.get(NOARGS_PLACEHOLDER).is_some(),
            "GLM should get the zero-argument placeholder, got {}",
            props
        );
        assert_eq!(
            out["tools"][0]["function"]["parameters"]["required"],
            json!([NOARGS_PLACEHOLDER]),
            "the placeholder must be required — an optional one is still emitted as {{}} most of the time"
        );
    }

    /// An `input_schema` with no `properties` key at all is also argument-less.
    #[test]
    fn test_noargs_tool_without_properties_key_gets_placeholder() {
        let body = json!({
            "messages": [],
            "tools": [{"name": "TaskList", "input_schema": {"type": "object"}}]
        });
        let out = anthropic_to_chat_completions("zai-org/GLM-5.2-FP8", &body, true);
        assert!(out["tools"][0]["function"]["parameters"]["properties"]
            .get(NOARGS_PLACEHOLDER)
            .is_some());
    }

    /// A tool that already has parameters is left exactly as-is.
    #[test]
    fn test_tool_with_args_is_untouched() {
        let body = json!({
            "messages": [],
            "tools": [{"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
        });
        let out = anthropic_to_chat_completions("zai-org/GLM-5.2-FP8", &body, true);
        let params = &out["tools"][0]["function"]["parameters"];
        assert!(params["properties"].get(NOARGS_PLACEHOLDER).is_none());
        assert_eq!(params["required"], json!(["command"]));
    }

    /// Models that stream empty-argument tool calls correctly must not pay for
    /// the workaround — an unexpected property costs tokens and can tempt the
    /// model into putting real content in it.
    #[test]
    fn test_noargs_placeholder_not_added_for_non_glm() {
        let body = json!({
            "messages": [],
            "tools": [{"name": "TaskList", "input_schema": {"type": "object", "properties": {}}}]
        });
        let out = anthropic_to_chat_completions("deepseek-ai/DeepSeek-V4-Flash-0731", &body, true);
        assert!(out["tools"][0]["function"]["parameters"]["properties"]
            .get(NOARGS_PLACEHOLDER)
            .is_none());
    }

    /// The placeholder must never reach the client: Claude Code validates
    /// tool_use input against its own schema and rejects unknown properties.
    #[test]
    fn test_placeholder_stripped_from_streamed_tool_use() {
        let events = run_stream(&[
            r#"data: {"id":"1","model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"TaskList","arguments":"{\"__noargs\": \"\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            "data: [DONE]",
        ]);
        let tool_uses = collect_tool_use(&events);
        assert_eq!(tool_uses.len(), 1, "expected one tool_use block");
        assert_eq!(tool_uses[0]["name"], "TaskList");
        let json = input_json_for(&events, 0).expect("tool_use should carry input_json_delta");
        let parsed: Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(
            parsed,
            json!({}),
            "placeholder must be stripped, got {}",
            parsed
        );
    }

    /// Stripping must not disturb real arguments sent alongside the placeholder.
    #[test]
    fn test_placeholder_stripped_keeps_real_args() {
        assert_eq!(
            strip_placeholder_arg(r#"{"__noargs":"","command":"ls"}"#),
            r#"{"command":"ls"}"#
        );
    }

    /// Arguments without the placeholder — and malformed arguments — pass
    /// through byte-for-byte so a real tool call is never mangled.
    #[test]
    fn test_strip_placeholder_is_noop_without_placeholder() {
        assert_eq!(
            strip_placeholder_arg(r#"{"command":"ls -la"}"#),
            r#"{"command":"ls -la"}"#
        );
        assert_eq!(strip_placeholder_arg("{not json"), "{not json");
        assert_eq!(strip_placeholder_arg("[1,2]"), "[1,2]");
    }

    /// The non-streaming response path strips the placeholder too.
    #[test]
    fn test_placeholder_stripped_from_non_streaming_tool_use() {
        let response = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "TaskList", "arguments": "{\"__noargs\": \"\"}"}
                    }]
                }
            }]
        });
        let msg = chat_completions_to_anthropic("zai-org/GLM-5.2-FP8", &response);
        let tool_use = msg["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["type"] == "tool_use")
            .expect("expected a tool_use block");
        assert_eq!(tool_use["input"], json!({}));
    }

    /// Stray mid-conversation system messages (injected by Claude Code when
    /// spawning a subagent) must be converted to user messages IN PLACE so the
    /// rendered token prefix stays stable for KV cache reuse. They must NOT be
    /// merged into the leading system message (that rewrote the prefix and
    /// forced a full re-prefill on the next turn).
    #[test]
    fn test_stray_system_message_converted_in_place() {
        let body = json!({
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "Available agent types: Explore"},
                {"role": "user", "content": "now explore"}
            ]
        });
        let out = anthropic_to_chat_completions("test-model", &body, false);
        let msgs = out["messages"].as_array().unwrap();
        // Leading system message untouched (no merge).
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful.");
        // Stray system became user at its original position (index 3).
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "hi");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[3]["role"], "user");
        assert_eq!(msgs[3]["content"], "Available agent types: Explore");
        assert_eq!(msgs[4]["role"], "user");
        assert_eq!(msgs[4]["content"], "now explore");
        assert_eq!(msgs.len(), 5);
    }
}
