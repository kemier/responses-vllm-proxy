#!/bin/bash
# Start vllm-proxy (Rust) — a standard no-auth vLLM OpenAI-compatible proxy.
#
# Usage:
#   ./start.sh --rebuild  Rebuild then restart
#   ./start.sh            Start the proxy (build if needed)
#   ./start.sh --restart  Stop any running instance and start fresh
#   ./start.sh --build    Build only (do not start the proxy)

DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8317
PIDFILE="/tmp/vllm-proxy-$USER.pid"

# Resolve the binary: prefer a release build in the source tree (dev mode),
# otherwise fall back to an installed copy in $DIR/bin (installed mode).
if [ -f "$DIR/target/release/vllm-proxy" ]; then
    BIN_PATH="$DIR/target/release/vllm-proxy"
else
    BIN_PATH="$DIR/bin/vllm-proxy"
fi

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
    echo "Restarting vllm-proxy..."

    # Kill via PID file first
    if [ -f "$PIDFILE" ]; then
        OLD_PID=$(cat "$PIDFILE" 2>/dev/null)
        if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Stopping existing instance (PID $OLD_PID)..."
            kill "$OLD_PID" 2>/dev/null
            # Wait for it to exit (max 15s — graceful shutdown needs time
            # to close WebSocket connections and drain in-flight requests)
            for i in $(seq 1 30); do
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

    # Kill ANY process still LISTENING on the port (stray orphans, multiple PIDs)
    STRAY_PIDS=$(lsof -ti :$PORT -sTCP:LISTEN 2>/dev/null | sort -u | tr '
' ' ')
    if [ -n "$STRAY_PIDS" ]; then
        echo "Killing stray listeners on port $PORT (PIDs: $STRAY_PIDS)..."
        for pid in $STRAY_PIDS; do
            kill "$pid" 2>/dev/null
        done
        sleep 2
        # Force kill survivors
        for pid in $STRAY_PIDS; do
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
        done
        sleep 1
    fi

    # Wait for port listener to be fully released (max 10s)
    for i in $(seq 1 20); do
        lsof -ti :$PORT -sTCP:LISTEN 2>/dev/null | head -1 | grep -q . || break
        sleep 0.5
    done

    # Final check
    if lsof -ti :$PORT -sTCP:LISTEN 2>/dev/null | head -1 | grep -q .; then
        echo "WARNING: Port $PORT still in use after kill attempts!"
        echo "  Listeners: $(lsof -ti :$PORT -sTCP:LISTEN 2>/dev/null | tr '
' ' ')"
        echo "  Manual fix: fuser -k $PORT/tcp"
    fi

    echo "Stopped."
fi

# ─── Check if already running ───────────────────────────────────────────────
if [ -f "$PIDFILE" ]; then
    EXISTING_PID=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "vllm-proxy is already running (PID $EXISTING_PID)."
        echo "Use --restart to restart, or kill $EXISTING_PID first."
        exit 1
    fi
    rm -f "$PIDFILE"
fi

# ─── Build ──────────────────────────────────────────────────────────────────
NEED_BUILD=false
if [ "$REBUILD" = true ] || [ "$BUILD_ONLY" = true ] || [ ! -f "$BIN_PATH" ]; then
    NEED_BUILD=true
fi

if [ "$NEED_BUILD" = true ]; then
    echo "Building vllm-proxy (release)..."
    cd "$DIR" && cargo build --release || { echo "Build failed!"; exit 1; }
fi

# --build means: build only, don't start
if [ "$BUILD_ONLY" = true ]; then
    echo "Build complete."
    exit 0
fi

# ─── Upstream keys ──────────────────────────────────────────────────────────
# Providers read their upstream key from env (api-key-env in config.yaml) so
# the tracked config never carries a secret. If the caller didn't export it,
# resolve it from ~/.claude/settings.json — ANTHROPIC_AUTH_TOKEN is the key
# this machine's sessions already authenticate the Ubuntu llama-server with.
if [ -z "${STRATA_UBUNTU_KEY:-}" ]; then
    STRATA_UBUNTU_KEY=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["env"]["ANTHROPIC_AUTH_TOKEN"])' "$HOME/.claude/settings.json" 2>/dev/null)
    if [ -n "$STRATA_UBUNTU_KEY" ]; then
        export STRATA_UBUNTU_KEY
        echo "STRATA_UBUNTU_KEY resolved from $HOME/.claude/settings.json"
    else
        echo "WARNING: STRATA_UBUNTU_KEY unset and not resolvable — keyed providers will 401 upstream"
    fi
fi

# ─── Start proxy ────────────────────────────────────────────────────────────
export CONFIG_PATH="$DIR/config.yaml"
(cd "$DIR" && "$BIN_PATH") &
PID=$!
echo "$PID" > "$PIDFILE"
echo "vllm-proxy started (PID $PID)"

# Wait for proxy ready
for i in $(seq 1 10); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 0.5
done

# ─── Cleanup on exit ────────────────────────────────────────────────────────
cleanup() {
    # Wait for the proxy to drain connections after receiving SIGINT/SIGTERM.
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "Waiting for proxy (PID $PID) to drain connections..."
        for i in $(seq 1 30); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing proxy (PID $PID)..."
            kill -9 "$PID" 2>/dev/null
        fi
    fi
    rm -f "$PIDFILE"
}
trap cleanup EXIT INT TERM

# Keep proxy in foreground
wait $PID