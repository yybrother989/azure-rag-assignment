#!/bin/bash
# Start the Chainlit web UI (background, headless).
#
# Defaults
#   port: 8000     (override with CHAINLIT_PORT env var)
#   logs: /tmp/chainlit.log
#
# Usage
#   ./scripts/start_chainlit.sh        # start on :8000, open browser
#   CHAINLIT_PORT=9000 ./scripts/start_chainlit.sh
#
# Stop with `./scripts/stop_chainlit.sh`.

set -e

PORT="${CHAINLIT_PORT:-8000}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="/tmp/chainlit.log"
VENV_CHAINLIT="$REPO_ROOT/.venv/bin/chainlit"

if [ ! -x "$VENV_CHAINLIT" ]; then
    echo "ERROR: $VENV_CHAINLIT not found. Activate or build the venv first."
    exit 1
fi

if lsof -i :"$PORT" -t >/dev/null 2>&1; then
    echo "Port $PORT is already in use. Run ./scripts/stop_chainlit.sh first, then retry."
    exit 1
fi

cd "$REPO_ROOT"
nohup "$VENV_CHAINLIT" run src/app.py --headless --port "$PORT" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Started Chainlit (PID $PID) on http://localhost:$PORT"
echo "Logs:  tail -f $LOG_FILE"
echo "Stop:  ./scripts/stop_chainlit.sh"

# Sanity check — make sure the process didn't die immediately.
sleep 3
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo ""
    echo "ERROR: Chainlit process died. Last 20 log lines:"
    tail -20 "$LOG_FILE"
    exit 1
fi

# Wait until the port responds, then open browser (macOS).
for _ in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/" -o /dev/null; then
        if command -v open >/dev/null 2>&1; then
            open "http://localhost:$PORT"
        fi
        echo "Ready. Browser opened."
        exit 0
    fi
    sleep 1
done

echo "Started but http://localhost:$PORT didn't respond within 30s. Check $LOG_FILE."
