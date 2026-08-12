---
name: rust-dev
description: Rust development skill for vLLM-proxy. Use when writing, editing, or reviewing Rust code in this repo. Enforces rustfmt, zero clippy warnings, proper logging via logger module, serde_json patterns, and the translation conventions specific to this proxy.
---

# Rust Development — vLLM-proxy

## Build & Lint

```bash
# Format (MUST pass before commit)
rustfmt src/*.rs

# Lint (MUST be zero warnings)
cargo clippy --release 2>&1 | grep "warning" && echo "FIX ALL WARNINGS" || echo "Clean"

# Build
cargo build --release

# Test
cargo test

# Restart proxy after changes
./start.sh --restart
```

## Code Style

- **Indentation**: 4 spaces, never tabs
- **rustfmt**: always run before committing — `rustfmt src/*.rs`
- **clippy**: zero warnings allowed
- **No `println!`/`eprintln!`**: use the `logger` module exclusively
  ```rust
  // GOOD
  logger::debug("translate_request", &format!("tools: {}", count));
  logger::info("ws", &format!("connection opened: {}", id));

  // BAD
  println!("tools: {}", count);
  ```
- **Error handling**: prefer `unwrap_or()` / `unwrap_or_default()` over `.unwrap()` in production paths
  ```rust
  // GOOD
  let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");

  // BAD
  let name = tool.get("name").and_then(|v| v.as_str()).unwrap();
  ```
- **JSON**: use `serde_json::json!()` for construction, `.get()` / `.as_str()` / `.as_i64()` for access
  ```rust
  let tool = json!({
      "type": "function",
      "function": { "name": name, "description": desc }
  });
  let desc = tool.get("function")
      .and_then(|f| f.get("description"))
      .and_then(|v| v.as_str())
      .unwrap_or("");
  ```

## Repo-Specific Conventions

### Translation Logic (translate_request.rs / translate_response.rs)
- **Custom/freeform tools** (apply_patch, exec): convert to function tools with single `input` string param — vLLM doesn't support grammar-constrained decoding
- **Regular function tools** (wait, exec_command, etc.): pass through with original params
- **Streaming**: chunks arrive in pieces — use `pending` buffers and flush on complete items
- **Think tags**: handle `<think>` / `</think>` across chunk boundaries with suffix-prefix matching
- **Usage**: vLLM sends two chunks — real usage then zero-usage cleanup; never overwrite real usage with zeros

### WebSocket vs HTTP
- Both paths share the same translation functions
- WS connections are reused across turns; cache previous request for `previous_response_id` merging
- Prewarm requests (`generate=false`) don't include full tool set — don't treat as real turns

### KV-Cache
- Track message prefix changes; log `KV-cache break!` when prefix changes between turns
- Incremental requests merge new items into cached request to preserve prefix stability
- vLLM 0.23.0 does NOT return `prompt_tokens_details.cached_tokens` — log will show MISS always

## Common Pitfalls

1. **Forgetting `use crate::logger;`** — if you use `logger::debug()` in a new file, add the import
2. **Indentation after apply_patch** — always run `rustfmt` after patching to fix mixed tabs/spaces
3. **Zero-usage chunk overwriting real usage** — the vLLM cleanup chunk (`prompt_tokens:0`) must not overwrite real usage data
4. **Relative paths in markdown links** — always use `file:///` absolute paths with `?symbol=` before `#L`
5. **Forgetting to restart proxy** — after code changes, run `./start.sh --restart`

## Commit Convention

```bash
# Format: <type>: <description>
# Types: feat, fix, debug, chore, perf
# Lowercase, imperative mood, no trailing period
git commit -m "fix: merge assistant tool_calls into single message for qwen3"
git commit -m "feat: add Code Mode tool support"
git commit -m "debug: log raw vLLM usage object and KV-cache hit/miss per turn"
```
