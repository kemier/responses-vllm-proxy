#!/bin/bash
# Start ericai-proxy (Rust) and regenerate Codex model catalog on startup + config changes
#
# Usage:
#   ./start.sh --rebuild  Rebuild then restart
#   ./start.sh            Start the proxy (build if needed)
#   ./start.sh --restart  Stop any running instance and start fresh
#   ./start.sh --build    Build only (do not start the proxy)

DIR="$(cd "$(dirname "$0")" && pwd)"
CATALOG="$HOME/.codex/model_catalog.json"
PORT=8317
PIDFILE="/tmp/ericai-proxy-$USER.pid"

# ─── Parse args ─────────────────────────────────────────────────────────────
RESTART=false
BUILD_ONLY=false
REBUILD=false
for arg in "$@"; do
    case "$arg" in
        --restart)  RESTART=true ;;
        --build)    BUILD_ONLY=true ;;
        --rebuild)  REBUILD=true ;;
        --help|-h)
            echo "Usage: $0 [--restart] [--build] [--rebuild]"
            echo ""
            echo "Options:"
            echo "  --restart   Stop any running instance and start fresh"
            echo "  --build     Build only, do not start the proxy"
            echo "  --rebuild   Rebuild then restart"
            echo "  --help      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (use --help)"
            exit 1
            ;;
    esac
done

# --rebuild implies --restart
if [ "$REBUILD" = true ]; then
    RESTART=true
fi

# ─── Stop existing instance if --restart ────────────────────────────────────
if [ "$RESTART" = true ]; then
    echo "Restarting ericai-proxy..."

    # Kill via PID file
    if [ -f "$PIDFILE" ]; then
        OLD_PID=$(cat "$PIDFILE" 2>/dev/null)
        if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Stopping existing instance (PID $OLD_PID)..."
            kill "$OLD_PID" 2>/dev/null
            # Wait for it to exit (max 5s)
            for i in $(seq 1 10); do
                kill -0 "$OLD_PID" 2>/dev/null || break
                sleep 0.5
            done
            # Force kill if still alive
            if kill -0 "$OLD_PID" 2>/dev/null; then
                echo "Force killing PID $OLD_PID..."
                kill -9 "$OLD_PID" 2>/dev/null
                sleep 1
            fi
        fi
        rm -f "$PIDFILE"
    fi

    # Also kill any stray processes on the port
    STRAY_PID=$(lsof -ti :$PORT 2>/dev/null)
    if [ -n "$STRAY_PID" ]; then
        echo "Killing stray process on port $PORT (PID $STRAY_PID)..."
        kill "$STRAY_PID" 2>/dev/null
        sleep 1
        kill -9 "$STRAY_PID" 2>/dev/null
    fi

    echo "Stopped."
fi

# ─── Check if already running ───────────────────────────────────────────────
if [ -f "$PIDFILE" ]; then
    EXISTING_PID=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "ericai-proxy is already running (PID $EXISTING_PID)."
        echo "Use --restart to restart, or kill $EXISTING_PID first."
        exit 1
    fi
    rm -f "$PIDFILE"
fi

# ─── Build ──────────────────────────────────────────────────────────────────
NEED_BUILD=false
if [ "$REBUILD" = true ] || [ "$BUILD_ONLY" = true ] || [ ! -f "$DIR/target/release/ericai-proxy" ]; then
    NEED_BUILD=true
fi

if [ "$NEED_BUILD" = true ]; then
    echo "Building ericai-proxy (release)..."
    cd "$DIR" && cargo build --release || { echo "Build failed!"; exit 1; }
fi

# --build means: build only, don't start
if [ "$BUILD_ONLY" = true ]; then
    echo "Build complete."
    exit 0
fi

# ─── Refresh EricAI token ───────────────────────────────────────────────────
echo "Refreshing EricAI token..."
(cd /workspace/git/ezennin/lab && OPENAI_API_KEY="" OPENAI_BASE_URL="" uv run python3 -c "
import json, os, time
from ericai.sso_auth import get_token
token = get_token(failOnInteractiveLogin=True)
path = os.path.expanduser('~/.cli-proxy-api/ericai-ericai-user.json')
os.makedirs(os.path.dirname(path), exist_ok=True)
try:
    with open(path) as f:
        data = json.load(f)
except:
    data = {}
data['access_token'] = token
data['expires_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(time.time() + 3600))
data['last_refresh'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
print(f'Token refreshed (len={len(token)})')
" 2>/dev/null) && echo "Token synced" || echo "Warning: token refresh failed, may need --ericai-login"

# ─── Start proxy ────────────────────────────────────────────────────────────
export CONFIG_PATH="$DIR/config.yaml"
"$DIR/target/release/ericai-proxy" &
PID=$!
echo "$PID" > "$PIDFILE"
echo "ericai-proxy started (PID $PID)"

# Wait for proxy ready
for i in $(seq 1 10); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 0.5
done

# ─── Initial catalog generation ─────────────────────────────────────────────
export CATALOG PORT CONFIG_PATH="$DIR/config.yaml"
"$DIR/scripts/update_catalog.sh"
echo "Catalog written to $CATALOG"

# ─── Watch config + monitor health ──────────────────────────────────────────
(
    LAST_HASH=$(sha256sum "$DIR/config.yaml" 2>/dev/null | cut -d' ' -f1)
    HEALTH_COUNTER=0
    while kill -0 $PID 2>/dev/null; do
        sleep 3
        REBUILD=0

        # Check config changes
        NEW_HASH=$(sha256sum "$DIR/config.yaml" 2>/dev/null | cut -d' ' -f1)
        if [ "$NEW_HASH" != "$LAST_HASH" ]; then
            LAST_HASH="$NEW_HASH"
            echo "Config change detected, updating catalog..."
            sleep 1
            REBUILD=1
        fi

        # Check model health every 30s (10 iterations x 3s)
        HEALTH_COUNTER=$((HEALTH_COUNTER + 1))
        if [ $HEALTH_COUNTER -ge 10 ]; then
            HEALTH_COUNTER=0
            "$DIR/scripts/check_health.sh" || REBUILD=1
        fi

        if [ $REBUILD -eq 1 ]; then
            "$DIR/scripts/update_catalog.sh"
            echo "Catalog written to $CATALOG"
        fi
    done
) &
WATCHER_PID=$!

# ─── Cleanup on exit ────────────────────────────────────────────────────────
cleanup() {
    kill $WATCHER_PID 2>/dev/null
    rm -f "$PIDFILE"
}
trap cleanup EXIT INT TERM

# Keep proxy in foreground
wait $PID
