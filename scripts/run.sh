#!/usr/bin/env bash
# Start SignalScope (API + built frontend) on http://localhost:${PORT:-8000}
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
if [ "${1:-}" = "--dev" ]; then
  uvicorn app.backend.main:app --reload --port 8010 &
  (cd app/frontend && npm run dev)
else
  [ -f app/frontend/dist/index.html ] || (cd app/frontend && npm run build)
  echo "SignalScope -> http://localhost:$PORT"
  exec uvicorn app.backend.main:app --host 0.0.0.0 --port "$PORT"
fi
