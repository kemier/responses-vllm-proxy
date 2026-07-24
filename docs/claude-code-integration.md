# Claude Code Integration

This guide explains how to use Claude Code (Anthropic's CLI coding agent) with EricAI models via the ericai-proxy.

## Architecture

```
claude (Claude Code CLI)
  └─ Sends Anthropic Messages API requests to localhost:3457
      └─ CCR gateway (Claude Code Router) translates:
          ├─ Anthropic Messages API → OpenAI Chat Completions format
          └─ Routes to ericai provider
              └─ ericai-proxy (localhost:8317)
                  └─ POST /v1/chat/completions (passthrough)
                      └─ EricAI vLLM (GLM-5.2, Qwen3, etc.)
```

The proxy's `/v1/chat/completions` passthrough handler injects `enable_thinking: false` for GLM models. This is required because GLM-5.2 in thinking mode returns output via the `reasoning` field, leaving `content` empty. CCR expects text in `content` — without this fix, CCR receives empty responses and returns 502.

## Prerequisites

- Node.js v22+ and npm
- ericai-proxy running on `localhost:8317` (`./start.sh`)
- EricAI MSAL token valid (refreshed by `start.sh`)

## Installation

### 1. Install Claude Code and Claude Code Router

```bash
npm install -g @anthropic-ai/claude-code
npm install -g @musistudio/claude-code-router
```

Verify:

```bash
claude --version
ccr --version
```

### 2. Configure CCR

Create `~/.claude-code-router/config.json`:

```json
{
  "HOST": "127.0.0.1",
  "PORT": 3456,
  "APIKEY": "ccr-local",
  "Providers": [
    {
      "name": "ericai",
      "api_base_url": "http://127.0.0.1:8317/v1/chat/completions",
      "api_key": "ericai-proxy",
      "api_type": "openai_chat_completions",
      "models": [
        "glm-5.2",
        "glm-5.1",
        "Qwen3-Coder",
        "deepseek-v4-pro",
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "qwen2.5-14b-1m"
      ]
    }
  ],
  "Router": {
    "builtInRules": {
      "claude-code": { "enabled": true }
    },
    "fallback": {
      "mode": "off",
      "models": [],
      "retryCount": 1
    },
    "rules": [
      {
        "matcher": "claude-sonnet-4-20250514",
        "provider": "ericai",
        "model": "glm-5.2"
      },
      {
        "matcher": "claude-sonnet-4",
        "provider": "ericai",
        "model": "glm-5.2"
      },
      {
        "matcher": "claude-opus-4",
        "provider": "ericai",
        "model": "glm-5.2"
      },
      {
        "matcher": "claude-3-5-sonnet",
        "provider": "ericai",
        "model": "glm-5.2"
      },
      {
        "matcher": "claude-3-5-haiku",
        "provider": "ericai",
        "model": "glm-5.2"
      }
    ]
  }
}
```

The `rules` array maps Claude Code's default model names (e.g. `claude-sonnet-4`) to EricAI models. All requests are routed to `glm-5.2` by default. Change the `model` field in each rule to use a different backend.

### 3. Start CCR

```bash
ccr start
```

CCR starts a daemon on port 3456 and a gateway on port 3457. The gateway auto-generates an auth key in `~/.claude-code-router/gateway.config.json` on every restart.

Verify:

```bash
ccr status
```

### 4. Configure Claude Code

Create the API key helper script:

```bash
cat > ~/.local/bin/claude-api-key-helper.sh << 'EOF'
#!/bin/bash
# Returns the current CCR gateway key dynamically
python3 -c "import json; g=json.load(open('/home/ezennin/.claude-code-router/gateway.config.json')); print(g['auth']['staticApiKeys']['keys'][0])" 2>/dev/null
EOF
chmod +x ~/.local/bin/claude-api-key-helper.sh
```

This script reads the gateway key at launch time, so it survives CCR restarts automatically.

Edit `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:3457",
    "ANTHROPIC_MODEL": "glm-5.2",
    "ANTHROPIC_SMALL_FAST_MODEL": "glm-5.2",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_BUG_COMMAND": "1",
    "DISABLE_FEEDBACK_COMMAND": "1",
    "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1"
  },
  "skipWebFetchPreflight": true,
  "apiKeyHelper": "/home/ezennin/.local/bin/claude-api-key-helper.sh"
}
```

- `ANTHROPIC_BASE_URL` points to the CCR gateway (port 3457, not 3456)
- `ANTHROPIC_MODEL` sets the default model Claude Code requests
- `apiKeyHelper` is a script that outputs the CCR gateway key — Claude Code calls this at startup to authenticate. This is required because the gateway key changes on every CCR restart.

### 5. Disable Telemetry and Non-Essential Traffic

By default, Claude Code sends telemetry, error reports, and other non-essential traffic to Anthropic's servers (`api.anthropic.com`). Since we're routing all inference through our own proxy, we disable all of this.

The environment variables in `settings.json` above handle this:

| Variable | What it disables |
|----------|-----------------|
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | Aggregate switch — disables auto-updater, bug command, error reporting, and telemetry together |
| `DISABLE_TELEMETRY` | Usage analytics (latency, reliability, feature usage patterns) |
| `DISABLE_ERROR_REPORTING` | Error stack traces sent to third-party error tracking |
| `DISABLE_BUG_COMMAND` | The `/bug` command (sends conversation history to Anthropic) |
| `DISABLE_FEEDBACK_COMMAND` | The `/feedback` command (sends conversation history to Anthropic) |
| `CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY` | Session quality survey prompts |
| `skipWebFetchPreflight` | WebFetch domain safety check (sends hostname to `api.anthropic.com` before fetching any URL) |

**What is NOT disabled** (cannot be turned off via settings):
- The initial `HEAD /api/hello` health check to `api.anthropic.com` on launch (~152 bytes, no conversation data). This is a connectivity probe built into the CLI binary. To block it entirely, add a hosts entry: `echo "127.0.0.1 api.anthropic.com" | sudo tee -a /etc/hosts` — but this may cause Claude Code to fail if it requires the probe to succeed.

**Important:** `skipWebFetchPreflight: true` means Claude Code will fetch any URL without consulting Anthropic's safety blocklist. Combine with WebFetch permission rules in settings if you need to restrict which domains Claude can reach.

### 5. Launch

```bash
claude
```

Or for one-shot mode:

```bash
claude --print "Say hello"
```

## How It Works

1. **Claude Code** sends Anthropic Messages API requests (model: `glm-5.2`) to `http://127.0.0.1:3457`
2. **CCR gateway** authenticates the request using the auto-generated key, then translates the Anthropic Messages format to OpenAI Chat Completions format
3. **CCR router** matches the model name against the `rules` array and forwards to the `ericai` provider
4. **ericai-proxy** receives the Chat Completions request at `POST /v1/chat/completions`, injects `enable_thinking: false` for GLM models, and forwards to the EricAI vLLM backend
5. **EricAI vLLM** processes the request and returns the response
6. The response flows back through the same chain, with CCR translating OpenAI format back to Anthropic Messages format

### Why `enable_thinking: false` Is Required

GLM-5.2 supports a "thinking mode" where the model outputs its reasoning via a `reasoning` field in the response, leaving the `content` field empty. This is useful for Codex (which has a `translate_response` state machine that parses `reasoning` deltas), but CCR's OpenAI parser only looks at the `content` field. Without disabling thinking, CCR receives empty responses and returns 502 errors.

The fix is in `src/main.rs` in the `handle_chat_completions` function:

```rust
// For GLM models, disable thinking so content is returned in the `content` field
// (not `reasoning`). CCR's OpenAI parser only reads `content`.
if model.contains("glm") {
    if let Some(kw) = body.get_mut("chat_template_kwargs").and_then(|v| v.as_object_mut()) {
        kw.entry("enable_thinking").or_insert(json!(false));
    } else {
        body["chat_template_kwargs"] = json!({"enable_thinking": false});
    }
}
```

This only affects the `/v1/chat/completions` passthrough path. The `/v1/responses` path (used by Codex) has its own thinking injection logic in `translate_request.rs` that is model- and reasoning-effort-aware.

## CCR Restart Procedure

If CCR restarts, the gateway key changes. The `apiKeyHelper` script handles this automatically — no manual intervention needed.

To manually restart CCR:

```bash
ccr stop
ccr start
```

To check the current gateway key:

```bash
python3 -c "import json; g=json.load(open('/home/ezennin/.claude-code-router/gateway.config.json')); print(g['auth']['staticApiKeys']['keys'][0])"
```

## Switching Models

To use a different EricAI model, change `ANTHROPIC_MODEL` in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_MODEL": "Qwen3-Coder"
  }
}
```

Or override at launch time:

```bash
ANTHROPIC_MODEL=Qwen3-Coder claude
```

The model name must match one of the models listed in the CCR `Providers[0].models` array. The CCR router rules map Claude Code's internal model names (`claude-sonnet-4`, etc.) to EricAI models — update the `model` field in the rules to change the default mapping.

## Troubleshooting

### "Not logged in - Please run /login"

Claude Code cannot find an API key. Ensure:
- `apiKeyHelper` in `~/.claude/settings.json` points to a valid executable script
- The script outputs the CCR gateway key (check with `~/.local/bin/claude-api-key-helper.sh`)
- CCR is running (`ccr status`)

### 502 Bad Gateway / Empty responses

The proxy's `enable_thinking: false` injection is not working. Check:
- The proxy is running the latest build (`./start.sh --restart`)
- The model name contains "glm" (the injection is gated on this)
- Proxy logs: `tail -f logs/main.log`

### "All target providers failed"

CCR cannot reach the ericai provider. Check:
- ericai-proxy is running on port 8317 (`curl http://localhost:8317/v1/models`)
- CCR config `api_base_url` is correct (`http://127.0.0.1:8317/v1/chat/completions`)
- CCR is running (`ccr status`)

### "API key is missing"

CCR gateway auth is failing. The `apiKeyHelper` script must output the exact key from `gateway.config.json`. Verify:

```bash
# What the helper outputs
~/.local/bin/claude-api-key-helper.sh

# What CCR expects
python3 -c "import json; g=json.load(open('/home/ezennin/.claude-code-router/gateway.config.json')); print(g['auth']['staticApiKeys']['keys'][0])"
```

These must match. If CCR was restarted, the key changed — the helper script reads it dynamically, so just relaunch `claude`.

## File Reference

| File | Purpose |
|------|---------|
| `~/.claude/settings.json` | Claude Code config — base URL, model, API key helper |
| `~/.claude-code-router/config.json` | CCR config — providers, models, routing rules |
| `~/.claude-code-router/gateway.config.json` | Auto-generated gateway auth config (key changes on restart) |
| `~/.local/bin/claude-api-key-helper.sh` | Script that outputs the current CCR gateway key |
| `src/main.rs` | Proxy `handle_chat_completions` — injects `enable_thinking: false` for GLM models |
