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
LAB_DIR="${1:-${LAB_DIR:-/workspace/git/ezennin/lab}}"

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

# NOTE: We `cd` into $LAB_DIR and use plain `uv run` (without --project) so that
# uv picks up the project's existing .venv. `uv run --project "$LAB_DIR"` from a
# different cwd ignores .venv and provisions a bare cpython (no pyyaml installed).
CATALOG="$CATALOG" PORT="$PORT" CONFIG_PATH="$CONFIG_PATH" CODEX_BIN="$CODEX_BIN" LAB_DIR="$LAB_DIR" sh -c 'cd "$LAB_DIR" && uv run python3' << 'PYEOF'
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
    "service_tiers", "upgrade",
    "use_responses_lite",
    "include_skills_usage_instructions",
}

# ── Step 2: Fetch proxy model list ──────────────────────────────────────────
try:
    with urllib.request.urlopen(f"http://localhost:{PORT}/v1/models") as resp:
        proxy_data = json.loads(resp.read())
    proxy_ids = sorted(set(m["id"] for m in proxy_data["data"] if not m["id"].startswith("ericai/")))
    # Build a map of model-id -> context_length from the proxy config.
    # This is the authoritative source (set per-model in config.yaml),
    # preferred over the backend's max_total_tokens which can be wrong
    # (e.g. GLM-5.2 reports 196608 which is the correct usable context).
    proxy_context_lengths = {}
    proxy_tags = {}
    for m in proxy_data.get("data", []):
        cl = m.get("context_length")
        if isinstance(cl, int) and cl > 0:
            proxy_context_lengths[m["id"]] = cl
        tags = m.get("tags", [])
        if isinstance(tags, list) and tags:
            proxy_tags[m["id"]] = [str(t) for t in tags]
except Exception as e:
    print(f"Warning: could not fetch proxy models: {e}")
    proxy_ids = []
    proxy_context_lengths = {}
    proxy_tags = {}

# ── Base instructions: teach LLM to emit clickable link formats ─────────────
BASE_INSTRUCTIONS = """
── AGENTS.md spec ────────────────────────────────────────────────────
- Repos often contain AGENTS.md files. These files can appear anywhere within the repository.
- These files are a way for humans to give you (the agent) instructions or tips for working within the container.
- Some examples might be: coding conventions, info about how code is organized, or instructions for how to run or test code.
- Instructions in AGENTS.md files:
    - The scope of an AGENTS.md file is the entire directory tree rooted at the folder that contains it.
    - For every file you touch in the final patch, you must obey instructions in any AGENTS.md file whose scope includes that file.
    - Instructions about code style, structure, naming, etc. apply only to code within the AGENTS.md file's scope, unless the file states otherwise.
    - More-deeply-nested AGENTS.md files take precedence in the case of conflicting instructions.
    - Direct system/developer/user instructions (as part of a prompt) take precedence over AGENTS.md instructions.
- The contents of the AGENTS.md file at the root of the repo and any directories from the CWD up to the root are included with the developer message and don't need to be re-read. When working in a subdirectory of CWD, or a directory outside the CWD, check for any AGENTS.md files that may be applicable.

── Reference code or symbol locations ────────────────────────────────────────────
When referencing code locations, use clickable links so the user can jump directly to source. Links must work in BOTH Zed and VS Code.

## Link Format (MANDATORY — read carefully, Zed has NO line correction)

Zed jumps to the exact #L line you provide. It does NOT use LSP or text search to correct wrong lines. VS Code corrects ±200 lines using ?symbol=. Therefore you MUST:

WORKFLOW (do this BEFORE writing your summary):
1. Identify all symbols you will reference in your answer
2. Run grep -n for EACH symbol to get the exact line number: grep -n 'symbol_name' file
3. Collect all the verified file:line pairs
4. ONLY THEN write your summary, using the verified line numbers in links

Do NOT write your summary first and guess line numbers — you will get them wrong. Verify first, then write.

### URL structure (STRICT ORDER — RFC 3986):
file:///{absolute_path}?symbol={symbol_name}#L{line_number}

The ?symbol= MUST come before #L. If reversed, Zed swallows the query into the fragment and every link jumps to line 1.

### Examples:
[responses_to_chat_completions](file:///workspace/git/ezennin/ericai-proxy/src/translate_request.rs?symbol=responses_to_chat_completions#L5)
[handle_content](file:///workspace/git/ezennin/ericai-proxy/src/translate_response.rs?symbol=handle_content#L316)
[fen_adapter_init](file:///workspace/git/ezennin/ssr/pkt/sw/src/fen/fen_adapter.c?symbol=fen_adapter_init#L142)

### Self-check before emitting ANY link:
- Does the URL have ?symbol= BEFORE #L? If not, fix it.
- Did you run grep -n to verify the line number? If not, run it now.
- Is ?symbol= present on EVERY file:// link? Not just some — ALL of them.

### For sibling/cross-workspace repos:
Same format. MUST verify the line with: grep -n 'symbol_name' file
[icrlib_client_send_client_msg()](file:///workspace/git/ezennin/ssr/pkt/sw/include/icrlib/icrlib_api.h?symbol=icrlib_client_send_client_msg#L525)

### Last resort — @symbol: (VS Code only):
[nats_msgbus_create](@symbol:nats_msgbus_create)
Only use when you cannot determine the file path. Zed users see plain text.

## Rules

1. EVERY file:// link MUST have ?symbol=symbol_name BEFORE #L: file:///path?symbol=name#L42
2. NEVER emit a file:// link without ?symbol= — not even one. Zed cannot correct it.
3. NEVER put #L before ?symbol=: file:///path#L42?symbol=name is BROKEN.
4. ALWAYS verify line numbers with grep -n. Zed jumps to exact #L with zero correction.
5. NEVER use relative paths in markdown links. [main.rs](src/main.rs) is BROKEN in Zed. Use [main.rs](file:///workspace/.../src/main.rs?symbol=main.rs#L42) instead.
6. Link text = bare symbol name WITHOUT backticks: [process_line](file:///...) NOT [`process_line`](file:///...)
7. NEVER use Type::method syntax. Use the bare method name in both link text and ?symbol=.
8. ALWAYS use ABSOLUTE paths starting with file:///
9. In C headers, function name is often on line AFTER "extern return_type". Point to the function name line.
10. For file references (not symbols), still include ?symbol= with the filename and #L with a line number.

── Code fence discipline ────────────────────────────────────────────────────
Code fences (```...```) are for code and diagrams ONLY.
- NEVER put prose, explanations, or narrative text inside a code fence. If it is not executable code, a shell command, a data structure, or an ASCII/PlantUML diagram, it does not belong in a code fence.
- When drawing ASCII art diagrams, close the fence immediately after the diagram ends. Do NOT continue writing explanations inside the same fence.
- If you need to explain a diagram, close the code fence first, then write the explanation as normal markdown.

── Sharing progress updates ──────────────────────────────────────────────────
- For especially longer tasks that you work on (i.e. requiring many tool calls, or a plan with multiple steps), you should provide progress updates back to the user at reasonable intervals. These updates should be structured as a concise sentence or two (no more than 8-10 words long) recapping progress so far in plain language: this update demonstrates your understanding of what needs to be done, progress so far (i.e. files explores, subtasks complete), and where you're going next.
- Before doing large chunks of work that may incur latency as experienced by the user (i.e. writing a new file), you should send a concise message to the user with an update indicating what you're about to do to ensure they know what you're spending time on. Don't start editing or writing large files before informing the user what you are doing and why.
- The messages you send before tool calls should describe what is immediately about to be done next in very concise language. If there was previous work done, this preamble message should also include a note about the work done so far to bring the user along.
"""

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
        cwd=os.environ.get("LAB_DIR", "/workspace/git/ezennin/lab"),
        env={**os.environ, "OPENAI_API_KEY": "", "OPENAI_BASE_URL": ""}
    )
    import re
    for obj_text in re.split(r'(?<=\})\s*(?=\{)', result.stdout):
        try:
            m = json.loads(obj_text)
            mid = m.get("id", "")
            if mid not in backend_ids:
                continue
            status = m.get("deployment_status", {}).get("status", "")
            demand = m.get("rayllm_metadata", {}).get("demand_status", {})
            running = m.get("deployment_status", {}).get("running_replicas", 0)
            loaded = demand.get("loaded", False)
            # Only include models that are actually loaded and have running replicas.
            # Models that are UPSCALING/DOWNSCALING with 0 replicas will hang on request.
            if status != "HEALTHY" or not loaded or running < 1:
                continue
            engine = m.get("rayllm_metadata", {}).get("engine_config", {})
            # Architecture can be at engine_config level OR top-level model
            arch = engine.get("architecture") or m.get("architecture") or {}
            ericai_meta[mid] = {
                "max_total_tokens": engine.get("max_total_tokens", 200000),
                "lifecycle": engine.get("lifecycle", ""),
                "gpu_type": m.get("rayllm_metadata", {}).get("gpu_type", ""),
                "num_gpus": m.get("rayllm_metadata", {}).get("num_gpus", 0),
                "engine_type": engine.get("type", ""),
                "input_modalities": arch.get("input_modalities", ["text"]),
                "status": status,
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
            {"effort": "medium", "description": "Standard thinking mode"},
            {"effort": "high", "description": "Deep thinking enabled (default)"},
        ],
        "default": "high",
    },
    "kimi": {
        "levels": [
            {"effort": "low", "description": "Disable thinking (thinking: false)"},
            {"effort": "high", "description": "Enable thinking (default)"},
        ],
        "default": "high",
    },
    "mimo": {
        "levels": [
            {"effort": "low", "description": "Disable thinking (enable_thinking: false)"},
            {"effort": "high", "description": "Enable thinking (default)"},
        ],
        "default": "high",
    },
}

def get_reasoning(slug):
    for prefix, profile in reasoning_profiles.items():
        if slug.lower().startswith(prefix):
            return profile["levels"], profile["default"]
    # Non-thinking models (e.g. qwen3-coder-instruct): advertise a single
    # "none" reasoning level. This makes codex's `single_supported_effort`
    # check true so the model picker closes immediately via
    # `dismiss_on_select` instead of looping back to the same dialogue.
    return [{"effort": "none", "description": "Non-reasoning model"}], "none"

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
    if lower.startswith("qwen"):
        return f"Qwen {slug[4:].upper()}"
    if lower.startswith("gemini-"):
        return f"Gemini {slug[7:]}"
    if lower.startswith("claude-"):
        return f"Claude {slug[7:]}"
    if "/" in slug:
        return slug[slug.rfind("/")+1:]
    return slug

# ── Tag-based model descriptions and service tiers ──────────────────────────
# The proxy serves capability tags (fast, coding, sota, near_sota, thinking,
# multimodal) from config.yaml / backend metadata. We translate these into
# human-readable descriptions the LLM sees in the spawn_agent tool schema, so
# it can pick fast/cheap models for subagents instead of always inheriting the
# expensive default.

TAG_DESCRIPTIONS = {
    "fast": "Fast and cost-efficient",
    "coding": "optimized for code generation",
    "sota": "frontier-grade reasoning",
    "near_sota": "near-frontier reasoning",
    "thinking": "deep thinking (slower)",
    "multimodal": "supports vision/images",
    "1M": "1M token context",
}

def build_description(slug, tags):
    """Build a cost/speed-aware description from tags."""
    if not tags:
        return f"{slug} via EricAI"
    parts = []
    for tag in tags:
        desc = TAG_DESCRIPTIONS.get(tag)
        if desc:
            parts.append(desc)
    if not parts:
        return f"{slug} via EricAI"
    return f"{slug} — {', '.join(parts)}"

def build_service_tiers(tags):
    """Fast-tagged models get a 'priority' (Fast) service tier so the LLM
    can pass service_tier='priority' when spawning subagents."""
    if tags and "fast" in tags:
        return [{"id": "priority", "name": "Fast", "description": "Fast, cost-efficient tier for subagents"}]
    return []


models = []
for i, slug in enumerate(proxy_ids):
    backend_id = alias_map.get(slug, slug)

    meta = ericai_meta.get(backend_id, {})
    if slug in alias_map and backend_id not in ericai_meta:
        print(f"  ⚠️  {slug} (backend {backend_id} not loaded/healthy) — including with defaults")
    levels, default_effort = get_reasoning(slug)
    # Prefer the proxy config's context_length (authoritative, per-model),
    # fall back to the backend metadata, then to 200000.
    context_window = (
        proxy_context_lengths.get(slug)
        or meta.get("max_total_tokens")
        or 200000
    )

    # Build entry with ONLY fields codex recognizes.
    # Codex auto-fills missing optional fields (see AUTO_FILLED_FIELDS).
    entry = {
        "slug": slug,
        "display_name": normalize_display_name(slug),
        "description": build_description(slug, proxy_tags.get(slug, [])),
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": i,
        "base_instructions": BASE_INSTRUCTIONS,
        # Only models with real thinking (low/high/etc.) get reasoning summaries.
        # "none"-only models are non-reasoning and must not advertise summaries.
        "supports_reasoning_summaries": any(e["effort"] != "none" for e in levels),
        "support_verbosity": False,
        "default_reasoning_summary": "auto",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": False,
       "input_modalities": meta.get("input_modalities", ["text"]),
       "context_window": context_window,
        # Use the full context window (no 5% headroom reserved by default).
        # Codex default auto-compaction (90% of context_window) is sufficient.
        "effective_context_window_percent": 100,
       "experimental_supported_tools": [],
        "apply_patch_tool_type": "freeform",
       "web_search_tool_type": "text",
       "supports_search_tool": False,
   }

    # Advertise Code Mode support. The proxy translates exec/wait freeform
    # tools to Chat Completions function tools, so all models can use Code
    # Mode. Setting tool_mode="code_mode" suppresses codex's
    # "model does not advertise Code Mode support" warning and enables
    # the Code Mode tool surface (exec + wait) instead of direct tools.
    # Add service tiers for fast-tagged models so the LLM sees them
    # in the spawn_agent tool schema and can pick them for subagents.
    tiers = build_service_tiers(proxy_tags.get(slug, []))
    if tiers:
        entry["service_tiers"] = tiers

    entry["tool_mode"] = "code_mode"

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
