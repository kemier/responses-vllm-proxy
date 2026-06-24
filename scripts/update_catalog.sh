#!/bin/bash
# Regenerate Codex model catalog from proxy /v1/models + ericai backend metadata
# Called at startup and on config.yaml changes.
#
# Strategy:
#   1. Extract the canonical schema from `codex debug models` (the codex binary
#      itself defines what fields ModelInfo expects — this is the source of truth)
#   2. Fetch available models from CLIProxyAPI /v1/models
#   3. Fetch backend metadata from `ericai api models.list`
#   4. Build catalog entries matching the canonical schema
#   5. Write atomically to ~/.codex/model_catalog.json
#   6. Validate by running `codex debug models` again and comparing field sets

DIR="$(cd "$(dirname "$0")" && pwd)"
CATALOG="${CATALOG:-$HOME/.codex/model_catalog.json}"
PORT="${PORT:-8317}"
CONFIG_PATH="${CONFIG_PATH:-$DIR/config.yaml}"

# Locate the codex binary (same one codex-acp will spawn)
CODEX_BIN="${CODEX_BIN:-$(find "$HOME/.npm-global/lib/node_modules/@openai" -path "*/codex-linux-x64/vendor/*/bin/codex" -type f 2>/dev/null | head -1)}"
if [ -z "$CODEX_BIN" ] || [ ! -x "$CODEX_BIN" ]; then
    echo "ERROR: codex binary not found. Set CODEX_BIN env var." >&2
    exit 1
fi

# Wait for proxy to be ready (max 5s)
for i in $(seq 1 10); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 0.5
done

CATALOG="$CATALOG" PORT="$PORT" CONFIG_PATH="$CONFIG_PATH" CODEX_BIN="$CODEX_BIN" uv run --project /workspace/git/ezennin/lab python3 << 'PYEOF'
import json, os, subprocess, sys, urllib.request, tempfile

CATALOG = os.environ["CATALOG"]
PORT = os.environ["PORT"]
CONFIG_PATH = os.environ["CONFIG_PATH"]
CODEX_BIN = os.environ["CODEX_BIN"]

# ── Step 1: Extract canonical schema from codex binary ──────────────────────
# `codex debug models` dumps the *current* model catalog as codex sees it.
# If our catalog already exists, this shows us the merged result (our fields +
# codex's auto-filled defaults). If no catalog exists yet, we get the built-in
# models — still useful for seeing the field set.
#
# We extract the full set of field names codex expects. This way, if a new codex
# version adds/removes/renames fields, our script can detect the drift and warn.

def get_codex_schema():
    """Run `codex debug models` and extract the field set + defaults."""
    try:
        proc = subprocess.run(
            [CODEX_BIN, "debug", "models"],
            capture_output=True, text=True, timeout=15
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None, None
        data = json.loads(proc.stdout)
        models = data.get("models", [])
        if not models:
            return None, None
        all_fields = set()
        field_defaults = {}
        for m in models:
            for k, v in m.items():
                all_fields.add(k)
                if k not in field_defaults:
                    field_defaults[k] = v
        return all_fields, field_defaults
    except Exception as e:
        print(f"Warning: could not extract schema from `codex debug models`: {e}")
        return None, None

canonical_fields, field_defaults = get_codex_schema()

if canonical_fields:
    print(f"Schema extracted from codex binary ({len(canonical_fields)} fields):")
    for f in sorted(canonical_fields):
        dv = field_defaults.get(f)
        if isinstance(dv, str) and len(str(dv)) > 60:
            dv = f"{str(dv)[:60]}..."
        print(f"  {f} = {dv!r}")
else:
    # Fallback: hard-coded schema from codex v0.142.0
    canonical_fields = {
        "slug", "display_name", "description",
        "default_reasoning_level", "supported_reasoning_levels",
        "shell_type", "visibility", "supported_in_api", "priority",
        "base_instructions", "supports_reasoning_summaries",
        "default_reasoning_summary", "support_verbosity",
        "apply_patch_tool_type", "web_search_tool_type",
        "truncation_policy", "supports_image_detail_original",
        "context_window", "experimental_supported_tools",
        "input_modalities", "supports_search_tool",
        "supports_parallel_tool_calls",
    }
    field_defaults = {}
    print(f"Using fallback schema ({len(canonical_fields)} fields)")

# Fields codex auto-fills — we don't need to set these in our catalog
AUTO_FILLED_FIELDS = {
    "additional_speed_tiers", "availability_nux", "default_verbosity",
    "effective_context_window_percent", "service_tiers", "upgrade",
    "use_responses_lite",
}

# ── Step 2: Fetch proxy model list ──────────────────────────────────────────
try:
    with urllib.request.urlopen(f"http://localhost:{PORT}/v1/models") as resp:
        proxy_data = json.loads(resp.read())
    proxy_ids = sorted(set(m["id"] for m in proxy_data["data"] if not m["id"].startswith("ericai/")))
except Exception as e:
    print(f"Warning: could not fetch proxy models: {e}")
    proxy_ids = []

# ── Step 3: Map proxy alias → ericai backend model ID ───────────────────────
alias_map = {}
try:
    import yaml
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for provider in (cfg.get("openai-compatibility") or []):
        if provider.get("prefix") == "ericai":
            for model in (provider.get("models") or []):
                alias_map[model.get("alias", model["name"])] = model["name"]
except Exception as e:
    print(f"Warning: could not parse config.yaml: {e}")

# ── Step 4: Fetch ericai backend metadata ───────────────────────────────────
ericai_meta = {}
backend_ids = set(alias_map.values())
try:
    result = subprocess.run(
        ["uv", "run", "ericai", "api", "models.list"],
        capture_output=True, text=True, timeout=30,
        cwd="/workspace/git/ezennin/lab",
        env={**os.environ, "OPENAI_API_KEY": "", "OPENAI_BASE_URL": ""}
    )
    import re
    for obj_text in re.split(r'(?<=\})\s*(?=\{)', result.stdout):
        try:
            m = json.loads(obj_text)
            mid = m.get("id", "")
            if mid not in backend_ids:
                continue
            if m.get("deployment_status", {}).get("status") != "HEALTHY":
                continue
            engine = m.get("rayllm_metadata", {}).get("engine_config", {})
            ericai_meta[mid] = {
                "max_total_tokens": engine.get("max_total_tokens", 200000),
                "lifecycle": engine.get("lifecycle", ""),
                "gpu_type": m.get("rayllm_metadata", {}).get("gpu_type", ""),
                "num_gpus": m.get("rayllm_metadata", {}).get("num_gpus", 0),
                "engine_type": engine.get("type", ""),
                "input_modalities": (engine.get("architecture") or {}).get("input_modalities", ["text"]),
            }
        except json.JSONDecodeError:
            pass
except Exception as e:
    print(f"Warning: could not fetch ericai metadata: {e}")

# ── Step 5: Build catalog entries matching canonical schema ─────────────────

reasoning_profiles = {
    "deepseek": {
        "levels": [
            {"effort": "low", "description": "Quick responses, lighter reasoning (DeepSeek: high)"},
            {"effort": "medium", "description": "Balanced reasoning depth (DeepSeek: high)"},
            {"effort": "high", "description": "Deeper reasoning for complex problems (DeepSeek: high)"},
            {"effort": "xhigh", "description": "Maximum reasoning depth (DeepSeek: max)"},
        ],
        "default": "high",
    },
    "glm": {
        "levels": [
            {"effort": "low", "description": "Disable deep thinking (enable_thinking: false)"},
            {"effort": "high", "description": "Deep thinking enabled (default)"},
        ],
        "default": "high",
    },
}

def get_reasoning(slug):
    for prefix, profile in reasoning_profiles.items():
        if slug.lower().startswith(prefix):
            return profile["levels"], profile["default"]
    return [], None

def normalize_display_name(slug):
    lower = slug.lower()
    if lower.startswith("deepseek-"):
        return f"DeepSeek {slug[9:].upper()}"
    if lower.startswith("glm-"):
        return f"GLM-{slug[4:]}"
    if lower.startswith("gpt-"):
        parts = slug.split("-", 2)
        r = f"GPT-{parts[1]}"
        if len(parts) == 3:
            r += f" {parts[2].title()}"
        return r
    if lower.startswith("kimi-"):
        return f"Kimi {slug[5:].upper()}"
    if lower.startswith("minimax-"):
        return f"MiniMax {slug[8:].upper()}"
    if lower.startswith("gemini-"):
        return f"Gemini {slug[7:]}"
    if lower.startswith("claude-"):
        return f"Claude {slug[7:]}"
    if "/" in slug:
        return slug[slug.rfind("/")+1:]
    return slug

models = []
for i, slug in enumerate(proxy_ids):
    backend_id = alias_map.get(slug, slug)

    if slug in alias_map and backend_id not in ericai_meta:
        print(f"  Skipping {slug} (backend {backend_id} not HEALTHY)")
        continue

    meta = ericai_meta.get(backend_id, {})
    levels, default_effort = get_reasoning(slug)
    context_window = meta.get("max_total_tokens", 200000) or 200000

    # Build entry with ONLY fields codex recognizes.
    # Codex auto-fills missing optional fields (see AUTO_FILLED_FIELDS).
    entry = {
        "slug": slug,
        "display_name": normalize_display_name(slug),
        "description": f"{slug} via EricAI",
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": i,
        "base_instructions": "",
        "supports_reasoning_summaries": bool(levels),
        "support_verbosity": False,
        "default_reasoning_summary": "auto",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": False,
        "input_modalities": meta.get("input_modalities", ["text"]),
        "context_window": context_window,
        "experimental_supported_tools": [],
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text",
        "supports_search_tool": False,
    }

    if levels:
        entry["supported_reasoning_levels"] = levels
        entry["default_reasoning_level"] = default_effort
    else:
        entry["supported_reasoning_levels"] = []
        entry["default_reasoning_level"] = None

    # Detect schema drift: warn if codex added NEW fields we don't know about
    entry_fields = set(entry.keys())
    unknown_fields = canonical_fields - entry_fields - AUTO_FILLED_FIELDS
    if unknown_fields:
        print(f"  ⚠️  {slug} — codex has NEW fields not in our entry: {sorted(unknown_fields)}")
        print(f"     These may be new fields added in a codex update. Check if they need values.")

    models.append(entry)

# ── Step 6: Write atomically ────────────────────────────────────────────────
catalog_dir = os.path.realpath(os.path.dirname(CATALOG))
os.makedirs(catalog_dir, exist_ok=True)
fd, tmp_path = tempfile.mkstemp(dir=catalog_dir, suffix=".tmp")
try:
    with os.fdopen(fd, "w") as f:
        json.dump({"models": models}, f, indent=2)
    os.replace(tmp_path, CATALOG)
except Exception:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    raise

print(f"\nModel catalog updated: {len(models)} models → {CATALOG}")
for m in models:
    cw = m["context_window"]
    efforts = [e["effort"] for e in m.get("supported_reasoning_levels", [])]
    effort_str = f"efforts={','.join(efforts)}" if efforts else "no-effort"
    print(f"  {m['slug']:25s} context_window={cw:>7d} {effort_str}")

# ── Step 7: Validate — run `codex debug models` and verify ──────────────────
try:
    proc = subprocess.run(
        [CODEX_BIN, "debug", "models"],
        capture_output=True, text=True, timeout=15
    )
    if proc.returncode == 0 and proc.stdout.strip():
        codex_models = json.loads(proc.stdout).get("models", [])
        if codex_models:
            codex_fields = set()
            for m in codex_models:
                codex_fields.update(m.keys())
            our_fields = set(models[0].keys()) if models else set()
            auto = codex_fields - our_fields
            stripped = our_fields - codex_fields
            if stripped:
                print(f"\n⚠️  Schema validation: codex STRIPPED these fields (not recognized):")
                for f in sorted(stripped):
                    print(f"    - {f}")
            else:
                print(f"\n✅ Schema validation passed: all {len(our_fields)} fields recognized by codex")
            print(f"   Codex auto-filled {len(auto)} optional fields: {sorted(auto)}")
    else:
        print(f"\n⚠️  Schema validation skipped: `codex debug models` failed")
except Exception as e:
    print(f"\n⚠️  Schema validation skipped: {e}")
PYEOF
