# Claude Code Integration

## Architecture

Claude Code CLI sends Anthropic Messages API requests directly to vLLM-proxy (localhost:8317).
The proxy has a native /v1/messages endpoint that translates Anthropic format to OpenAI Chat
Completions internally, then translates the streaming SSE response back to Anthropic format.

This eliminates CCR (Claude Code Router) entirely, fixing broken streaming.

## Configuration

Edit ~/.claude/settings.json:
- ANTHROPIC_BASE_URL: http://127.0.0.1:8317
- ANTHROPIC_MODEL: glm-5.2
- apiKeyHelper: ~/.local/bin/claude-api-key-direct.sh (outputs vLLM-proxy)

Unset ANTHROPIC_API_KEY env var if set (it overrides apiKeyHelper).

## Key Files

- src/translate_anthropic.rs: Anthropic <-> OpenAI translation
- src/main.rs handle_messages(): /v1/messages HTTP handler
- ~/.claude/settings.json: Claude Code config

## Migration from CCR

ccr stop && npm uninstall -g @musistudio/claude-code-router
