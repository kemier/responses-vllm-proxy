#!/bin/bash
# Check if any model in the catalog became unhealthy (exit 1 = needs rebuild)

DIR="$(cd "$(dirname "$0")" && pwd)"
CATALOG="${CATALOG:-$HOME/.codex/model_catalog.json}"
CONFIG_PATH="${CONFIG_PATH:-$DIR/config.yaml}"

uv run --project /workspace/git/ezennin/lab python3 << 'PYEOF'
import json, os, subprocess, sys, re

CATALOG = os.environ.get("CATALOG", os.path.expanduser("~/.codex/model_catalog.json"))
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")

# Load current catalog
try:
    with open(CATALOG) as f:
        catalog = json.load(f)
except:
    sys.exit(0)  # no catalog yet, nothing to check

catalog_backends = {m["ericai_backend_id"] for m in catalog.get("models", []) if m.get("ericai_backend_id")}
if not catalog_backends:
    sys.exit(0)

# Get healthy backends
healthy = set()
try:
    result = subprocess.run(
        ["uv", "run", "ericai", "api", "models.list"],
        capture_output=True, text=True, timeout=30,
        cwd="/workspace/git/ezennin/lab",
        env={**os.environ, "OPENAI_API_KEY": "", "OPENAI_BASE_URL": ""}
    )
    for obj_text in re.split(r'(?<=\})\s*(?=\{)', result.stdout):
        try:
            m = json.loads(obj_text)
            if m.get("deployment_status", {}).get("status") == "HEALTHY":
                healthy.add(m.get("id", ""))
        except json.JSONDecodeError:
            pass
except:
    sys.exit(0)  # can't reach backend, don't remove models

# Check if any catalog model is no longer healthy
unhealthy = catalog_backends - healthy
if unhealthy:
    for mid in sorted(unhealthy):
        print(f"  Model unhealthy, removing from catalog: {mid}")
    sys.exit(1)

# Also check if new models became healthy that weren't in catalog
import yaml
try:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    configured = set()
    for provider in (cfg.get("openai-compatibility") or []):
        if provider.get("prefix") == "ericai":
            for model in (provider.get("models") or []):
                configured.add(model["name"])
except:
    sys.exit(0)

newly_healthy = (configured & healthy) - catalog_backends
if newly_healthy:
    for mid in sorted(newly_healthy):
        print(f"  Model now healthy, adding to catalog: {mid}")
    sys.exit(1)

sys.exit(0)
PYEOF
