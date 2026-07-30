// Tests for the text-only ("no tools") turn discriminator used to fix
// Claude Code CLI's /compact failing with
// "summarization produced empty response".
//
// Claude Code sends its compaction prompt with a literal
// "Do NOT call any tools" instruction while still shipping the full tool
// schemas. GLM-5.x calls a tool anyway, leaving the response with no text
// block. The proxy detects the instruction and strips the tools so the
// backend must answer in prose.

use ericai_proxy::main::is_text_only_turn;
use serde_json::json;

/// The exact preamble Claude Code CLI prepends to a compaction prompt.
const COMPACT_PREAMBLE: &str = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\nYour task is to create a detailed summary of the conversation so far.";

/// The reminder Claude Code appends after the summary instructions.
const COMPACT_REMINDER: &str = "REMINDER: Do NOT call any tools. Respond with plain text only \u{2014} an <analysis> block followed by a <summary> block. Tool calls will be rejected and you will fail the task.";

#[test]
fn detects_compact_preamble_in_string_content() {
    let req = json!({
        "messages": [{ "role": "user", "content": COMPACT_PREAMBLE }]
    });
    assert!(is_text_only_turn(&req));
}

#[test]
fn detects_compact_reminder_in_string_content() {
    let req = json!({
        "messages": [{ "role": "user", "content": COMPACT_REMINDER }]
    });
    assert!(is_text_only_turn(&req));
}

#[test]
fn detects_marker_in_anthropic_block_content() {
    // Anthropic-shaped content: a list of typed blocks rather than a string.
    let req = json!({
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": "some earlier context" },
                { "type": "text", "text": COMPACT_PREAMBLE }
            ]
        }]
    });
    assert!(is_text_only_turn(&req));
}

#[test]
fn detects_marker_in_a_later_user_message() {
    let req = json!({
        "messages": [
            { "role": "user", "content": "add pagination to /users" },
            { "role": "assistant", "content": "done" },
            { "role": "user", "content": COMPACT_PREAMBLE }
        ]
    });
    assert!(is_text_only_turn(&req));
}

#[test]
fn ignores_marker_outside_user_messages() {
    // A model that merely echoes the phrase must not disable tools for the
    // whole conversation.
    let req = json!({
        "messages": [
            { "role": "system", "content": COMPACT_PREAMBLE },
            { "role": "assistant", "content": COMPACT_REMINDER }
        ]
    });
    assert!(!is_text_only_turn(&req));
}

#[test]
fn ordinary_turn_is_not_text_only() {
    let req = json!({
        "messages": [{ "role": "user", "content": "read src/main.rs and fix the bug" }],
        "tools": [{ "type": "function", "function": { "name": "Read" } }]
    });
    assert!(!is_text_only_turn(&req));
}

#[test]
fn thinking_disabled_alone_is_not_text_only() {
    // Running with thinking off (e.g. a non-reasoning model such as
    // Qwen2.5-14B-1M) must keep tools available.
    let req = json!({
        "thinking": { "type": "disabled" },
        "messages": [{ "role": "user", "content": "list the files in src/" }],
        "tools": [{ "type": "function", "function": { "name": "Bash" } }]
    });
    assert!(!is_text_only_turn(&req));
}

#[test]
fn missing_or_malformed_messages_is_not_text_only() {
    assert!(!is_text_only_turn(&json!({})));
    assert!(!is_text_only_turn(&json!({ "messages": "not-an-array" })));
    assert!(!is_text_only_turn(&json!({ "messages": [] })));
    // A user message with no content field at all.
    assert!(!is_text_only_turn(&json!({ "messages": [{ "role": "user" }] })));
}
