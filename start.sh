#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON="$DIR/.venv/bin/python"
elif [ -f "/Users/dickyeka/.gemini/antigravity/scratch/face-recognition-mcp/.venv/bin/python" ]; then
    PYTHON="/Users/dickyeka/.gemini/antigravity/scratch/face-recognition-mcp/.venv/bin/python"
else
    PYTHON="python3"
fi

echo "=== Starting Google Drive Face Finder Web App ==="
echo "URL: http://localhost:8000"
echo "Press Ctrl+C to stop."

PYTHONPATH="$DIR" exec "$PYTHON" -m uvicorn app:app --app-dir "$DIR" --host 0.0.0.0 --port 8000 --reload
