#!/usr/bin/env bash

# DocuSense Server Startup Script
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "          DocuSense FastAPI Web Server            "
echo "=================================================="

# Check if port 8000 is occupied by a stale process and free it
if command -v lsof >/dev/null 2>&1 && lsof -i :8000 >/dev/null 2>&1; then
    echo "[WARN] Port 8000 is currently occupied. Terminating stale process..."
    fuser -k 8000/tcp >/dev/null 2>&1 || true
    sleep 1
fi

# Check if uv is installed
if command -v uv >/dev/null 2>&1; then
    echo "[INFO] Using uv dependency manager..."
    if [ ! -d ".venv" ]; then
        echo "[INFO] Installing dependencies with uv sync..."
        uv sync
    fi
    CMD="uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload"
elif [ -f ".venv/bin/uvicorn" ]; then
    echo "[INFO] Using local .venv Python environment..."
    CMD=".venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --reload"
else
    echo "[INFO] Using system uvicorn..."
    CMD="uvicorn api:app --host 0.0.0.0 --port 8000 --reload"
fi

echo ""
echo "🚀 Server is starting up!"
echo "--------------------------------------------------"
echo " 🌐 Web Dashboard: http://localhost:8000"
echo " ⚙️  Settings Page: http://localhost:8000 (Settings Tab)"
echo " 📖 API Docs:       http://localhost:8000/docs"
echo "--------------------------------------------------"
echo "Press Ctrl+C to stop the server."
echo ""

exec $CMD
