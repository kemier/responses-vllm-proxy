# ericai-proxy

Minimal Rust proxy that translates OpenAI Responses API ↔ Chat Completions for EricAI (vLLM backend).

## Architecture

```
Codex CLI / vscode-acp (codex-acp)
  └─ Sends OpenAI Responses API requests to localhost:8317
      └─ ericai-proxy (Rust) translates:
          ├─ POST /v1/responses → /chat/completions (streaming SSE translation)
          ├─ GET  /v1/responses → WebSocket (same translation over WS)
          ├─ POST /v1/chat/completions → passthrough
          ├─ POST /backend-api/codex/responses/compact → compaction wrapper
          └─ GET  /v1/models → model list from config
              └─ EricAI vLLM (https://ray.sero.gic.ericsson.se/backendsso/v1)
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List available models |
| POST | `/v1/responses` | Main Responses API (streaming + non-streaming) |
| GET | `/v1/responses` | WebSocket Responses API |
| POST | `/v1/chat/completions` | Chat completions passthrough |
| POST | `/backend-api/codex/responses` | Alias for `/v1/responses` |
| GET | `/backend-api/codex/responses` | WebSocket alias |
| POST | `/backend-api/codex/responses/compact` | Remote compaction v2 |

## Build

```bash
cargo build --release
```

## Run

```bash
# Direct
CONFIG_PATH=./config.yaml ./target/release/ericai-proxy

# Via start.sh (builds + token refresh + catalog generation)
./start.sh
```

## Config

See `config.yaml` — minimal config with only EricAI provider settings.

## Features

- **Request translation**: Responses API → Chat Completions (instructions, input array, tools, function calls, reasoning effort)
- **Response translation**: Chat Completions SSE → Responses API SSE (full event sequence)
- **Thinking injection**: DeepSeek (`thinking: true, reasoning_effort`), GLM (`enable_thinking`)
- **Token management**: MSAL OAuth refresh, auto-refresh 5min before expiry
- **Remote compaction**: Wraps chat completion as `response.compaction` object
- **WebSocket support**: Stateful Responses WS transport for Codex — `generate=false` prewarm handshake, connection reuse across turns, and `previous_response_id` incremental-input expansion (rebuilds full history for the stateless vLLM backend)
- **Model aliasing**: Config aliases (e.g. `glm-5.2` → `zai-org/GLM-5.2-FP8`)
