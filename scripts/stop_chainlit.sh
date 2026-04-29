#!/bin/bash
# Stop Chainlit by killing whatever is listening on the Chainlit port.
#
# Defaults to :8000 — override with CHAINLIT_PORT env var.
#
# Usage
#   ./scripts/stop_chainlit.sh
#   CHAINLIT_PORT=9000 ./scripts/stop_chainlit.sh

PORT="${CHAINLIT_PORT:-8000}"
PIDS="$(lsof -i :"$PORT" -t 2>/dev/null || true)"

if [ -z "$PIDS" ]; then
    echo "Nothing listening on port $PORT — already stopped."
    exit 0
fi

echo "Killing PID(s) on port $PORT: $PIDS"
echo "$PIDS" | xargs kill -9 2>/dev/null || true
sleep 1

if lsof -i :"$PORT" -t >/dev/null 2>&1; then
    echo "WARNING: port $PORT still in use after kill. Try again or check manually."
    exit 1
fi

echo "Stopped. Port $PORT is free."
