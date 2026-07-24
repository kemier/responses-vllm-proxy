# AGENTS.md — ericai-proxy

## What This Repo Is

ericai-proxy is a Rust proxy that translates between the OpenAI Responses API (used by Codex CLI) and the Chat Completions API (used by EricAI's vLLM backend). It enables Codex CLI, vscode-acp, and Zed's codex-acp to use EricAI-hosted models (GLM-5.2, Qwen3, Kimi-K2, DeepSeek-V4, etc.) instead of OpenAI's API.

## Related Repositories

| Repo | Path | Description |
|---|---|---|
| **ericli** | `../ericli` | EricAI CLI — a Codex CLI alternative that talks to this proxy. Provides `ericli acp` for ACP stdio. Key files: `src/acp/acp-server.ts` (ACP server), `src/core/tool-executor.ts` (tool flow + streaming), `src/ericai/client.ts` (vLLM streaming client). Install: `bash scripts/install-acp.sh --local`. |
| **vscode-acp** | `../vscode-acp` | VS Code extension implementing the Agent Client Protocol. Renders chat UI, spawns agent binaries via stdio/JSON-RPC. Key files: `src/ui/ChatWebviewProvider.ts` (webview + thought blocks + tool tooltips), `src/core/SessionManager.ts` (session lifecycle, cancel). Deploy: `npm run package && npx vsce package` → install VSIX to `~/.vscode-server/extensions/`. |
| **codex-acp** | `../codex-acp` | ACP adapter for Codex CLI. The bridge between vscode-acp's JSON-RPC and Codex core. Key files: `src/CodexAcpServer.ts` (ACP server), `src/CodexEventHandler.ts` (reasoning/thought event mapping). |
| **codex** | `../codex` | Codex CLI core (Rust). The actual agent engine. Key crate: `codex-rs/app-server/`. |

## Architecture

```
Codex CLI / vscode-acp / Zed codex-acp
  └─ Sends Responses API requests to localhost:8317
      └─ ericai-proxy (Rust/Axum) translates:
          ├─ POST /v1/responses        → HTTP streaming SSE translation
          ├─ GET  /v1/responses        → WebSocket streaming translation
          ├─ POST /v1/chat/completions → passthrough
          |- POST /v1/messages       -> Anthropic Messages API (Claude Code)
          ├─ POST /v1/responses/compact → compaction wrapper
          └─ GET  /v1/models           → model list from config
              └─ EricAI vLLM (https://ray.sero.gic.ericsson.se/backendsso/v1)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/main.rs` | HTTP/WebSocket routes, request handling, incremental request merging, KV-cache prefix tracking |
| `src/translate_request.rs` | Responses API → Chat Completions translation (messages, tools, thinking injection) |
| `src/translate_response.rs` | Chat Completions streaming → Responses API SSE events (content, tool_calls, reasoning, usage) |
| src/translate_anthropic.rs | Anthropic Messages API <-> OpenAI Chat Completions translation (request + streaming response, used by Claude Code) |
| `src/auth.rs` | Microsoft SSO token management (refresh, caching) |
| `src/config.rs` | YAML config parsing |
| `src/logger.rs` | Custom async logger with file rotation and size limits |
| `scripts/update_catalog.sh` | Regenerates `~/.codex/model_catalog.json` from codex binary schema + EricAI backend metadata |
| `start.sh` | Build, start, restart the proxy; regenerates catalog on startup |
| `config.yaml` | Proxy configuration (port, auth, model aliases, logging) |
| `tests/` | Unit tests for translation logic |

## Development Conventions

### Build & Run
- `./start.sh` — start (builds if needed)
- `./start.sh --restart` — stop existing instance and start fresh
- `./start.sh --rebuild` — rebuild from scratch and restart
- `./start.sh --build` — build only, don't start
- `cargo build --release` — manual build
- `cargo test` — run unit tests
- `cargo clippy` — lint (must pass with zero warnings)

### Code Style
- Use `rustfmt` before committing: `rustfmt src/*.rs`
- Zero clippy warnings allowed
- Indentation: 4 spaces (never tabs)
- Logging: use `logger::debug()`, `logger::info()`, `logger::warn()`, `logger::error()` — never `println!` or `eprintln!`
- Log module name as first arg (e.g. `logger::debug("translate_request", &msg)`)
- Error handling: prefer `unwrap_or()` / `unwrap_or_default()` over `.unwrap()` in production paths
- JSON: use `serde_json::json!()` macro for construction, `.get()` / `.as_str()` for access

### Commit Convention
Use conventional commits with gitemojis (Ericsson internal guideline):
- `feat:` new feature (e.g. `feat: add Code Mode tool support`)
- `fix:` bug fix (e.g. `fix: merge assistant tool_calls into single message for qwen3`)
- `debug:` debug logging addition (e.g. `debug: log raw vLLM usage object`)
- `chore:` maintenance (e.g. `chore: add architecture docs`)
- `perf:` performance improvement
- Lowercase, imperative mood, no trailing period

### Translation Logic Conventions
- **Custom/freeform tools** (apply_patch, exec): converted to function tools with single `input` string param — vLLM doesn't support grammar-constrained decoding
- **Regular function tools** (wait, exec_command, etc.): pass through with original params
- **apply_patch tool description**: only 32 chars from codex — the full patch grammar must live in `base_instructions` in `update_catalog.sh`
- **exec tool description**: ~3000 chars, built dynamically by codex — passed through as-is, no extra injection needed
- **Model-specific quirks** (qwen3 thinking, GLM reasoning tags): handled in `translate_request.rs` via `inject_thinking()` and in `translate_response.rs` via tag detection

### Model Catalog
- `scripts/update_catalog.sh` regenerates `~/.codex/model_catalog.json` on startup
- The catalog maps proxy aliases (e.g. `glm-5.2`) to vLLM model IDs (e.g. `zai-org/GLM-5.2-FP8`)
- `base_instructions` in the catalog includes: file link format, code fence discipline, apply_patch grammar
- `tool_mode: "code_mode"` is set for all models to enable Codex Code Mode (exec/wait tools)
- The codex binary at `../codex` is the source of truth for the ModelInfo schema

### Debugging
- Logs are in `./logs/` (main.log + error-*.log)
- Config: `log-level: "debug"` and `log-turn-debug: true` in `config.yaml`
- Check `~/known-issue.md` and `~/troubleshooting-guide.md` for known problems
- The `../codex` repo contains the Codex CLI source code — reference it when investigating protocol issues
- Codex config: `~/.codex/config.toml` (features, model selection) and `~/.codex/model_catalog.json`
- vLLM backend: vLLM 0.23.0, does NOT return `prompt_tokens_details.cached_tokens` in API responses

### WebSocket vs HTTP
- Codex CLI TUI uses HTTP (`POST /v1/responses`)
- vscode-acp and Zed codex-acp use WebSocket (`GET /v1/responses` with upgrade)
- Both paths share the same translation logic (`translate_request` / `translate_response`)
- WebSocket connections are reused across turns; the proxy caches the previous request for `previous_response_id` incremental merging
- Prewarm requests (`generate=false`) are sent by codex-acp before actual turns — these don't include the full tool set

### KV-Cache Considerations
- The proxy tracks message prefix changes and logs `KV-cache break!` when the prefix changes between turns
- Incremental requests (`previous_response_id`) merge new items into cached request to preserve prefix stability
- vLLM 0.23.0 does not expose per-call cache hit rate — only Prometheus `/metrics` (not exposed by EricAI)

### Config in `~/.codex/config.toml`
- `[features]` section: `code_mode = true`, `code_mode_host = true`
- `suppress_unstable_features_warning = true` (top-level, NOT inside `[features]`)
- `model_catalog_json = "/home/<user>/.codex/model_catalog.json"`
- Model provider: `cliproxy` (points to localhost:8317)
