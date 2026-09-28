#!/bin/sh
# Starts CAMEL Sentinel inside the container. MODE selects what runs:
#   all (default) - backend in the background + Streamlit UI in the foreground
#   api           - FastAPI backend only (port 8000)
#   ui            - Streamlit UI only (port 8501), calling $CAMEL_BACKEND_URL
#
# Security (docs/09 F6): in MODE=all the UI calls the API over localhost, so the API
# listens on 127.0.0.1 only and is NOT reachable from outside the container, even if
# port 8000 is published. Set CAMEL_API_PUBLIC=1 to expose it (e.g. to demo /docs).
set -e
cd /app

API_HOST=127.0.0.1
if [ "${CAMEL_API_PUBLIC:-0}" = "1" ]; then API_HOST=0.0.0.0; fi

case "${MODE:-all}" in
  api)
    exec python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000
    ;;
  ui)
    exec python -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
    ;;
  all)
    python -m uvicorn main:app --app-dir backend --host "$API_HOST" --port 8000 &
    API_PID=$!
    # If the UI exits, take the API down with it so the container stops cleanly.
    trap 'kill $API_PID 2>/dev/null' EXIT INT TERM
    python -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
    ;;
  *)
    echo "Unknown MODE '$MODE' (use all, api or ui)" >&2
    exit 64
    ;;
esac
