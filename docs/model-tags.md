# Model Tags — Fast Model Routing for Subagents

The vLLM-proxy serves capability `tags` on each model in the `/v1/models`
endpoint. These tags allow downstream clients (codex, claude-code-acp) to select
fast, cost-efficient models for subagent tasks instead of always inheriting
the expensive default session model.

## How Tags Are Served

The `/v1/models` response includes a `tags` field on each model entry:

```json
{
  "id": "gpt-oss-120b",
  "object": "model",
  "owned_by": "vLLM",
  "vLLM_backend_id": "openai/gpt-oss-120b",
  "context_length": 131072,
  "tags": ["fast"]
}
```

### Tag Priority

Tags are resolved with this priority (in `handle_models`, `src/main.rs`):

1. **Live backend metadata** — `fetch_backend_meta()` calls
   `vLLM api models.list` every 5 minutes and extracts
   `routing_tags` from `rayllm_metadata.engine_config`. Cached in a
   `BackendMetaCache` (RwLock HashMap).
2. **config.yaml `tags` field** — static fallback when backend metadata is
   unavailable. Available immediately on proxy startup (no delay waiting
   for the backend fetch).

## Configuring Tags in config.yaml

```yaml
openai-compatibility:
  - name: vLLM
    models:
      - name: openai/gpt-oss-120b
        alias: gpt-oss-120b
        tags: [fast]
      - name: Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
        alias: Qwen3-Coder-480B
        tags: [fast, coding]
      - name: zai-org/GLM-5.2-FP8
        alias: glm-5.2
        tags: [sota, near_sota, thinking]
```

## Available Tags

| Tag | Meaning | Use Case |
|------|---------|----------|
| `fast` | Fast, non-thinking, cost-efficient | Subagents, tool-heavy turns |
| `coding` | Optimized for code generation | Worker subagents, code tasks |
| `sota` | Frontier-grade reasoning | Main session, complex reasoning |
| `near_sota` | Near-frontier reasoning | Alternative to sota |
| `thinking` | Deep thinking (slower) | Complex analysis (avoid for subagents) |
| `multimodal` | Supports vision/images | Image tasks |
| `1M` | 1M token context window | Very large context |

## Codex Model Catalog Integration

The `scripts/update_catalog.sh` script reads tags from the proxy `/v1/models`
response and translates them into the Codex model catalog
(`~/.codex/model_catalog.json`):

1. **Cost-aware descriptions** — Instead of generic `"gpt-oss-120b via
   vLLM"`, the LLM sees `"gpt-oss-120b — Fast and cost-efficient"`. This
   lets the LLM pick fast models for subagents via the `model` param in
   `spawn_agent`.
2. **Service tiers** — Fast-tagged models get `service_tiers: [{id:
   "priority", name: "Fast"}]` so the LLM can pass `service_tier:
   "priority"` when spawning.

The catalog is regenerated on proxy startup and on config.yaml changes.

## Tag Resolution Integration

The client's `resolveTag('tag:fast', models)` function reads the `tags` field
from the proxy response (as a fallback when `rayllm_metadata.routing_tags`
is absent). Proxy-managed models (with `tags` but no `rayllm_metadata`) are
marked `ACTIVE` automatically, since the proxy handles backend routing.

See the client docs for details.

## Adding a New Tagged Model

1. Add the model to `config.yaml` with the appropriate `tags`
2. Copy the updated config to the runtime location:
   `~/.local/share/vLLM-proxy/config.yaml`
3. Restart the proxy: `~/.local/share/vLLM-proxy/start.sh --restart`
4. The catalog script runs automatically and updates
   `~/.codex/model_catalog.json`

## Verifying Tags

```bash
curl -s http://127.0.0.1:8317/v1/models | python3 -c "
import sys, json
d = json.load(sys.stdin)
for m in d['data']:
    tags = m.get('tags', [])
    if tags:
        print(f\"{m['id']:25s} tags={tags}\")
"
```
