#!/bin/sh
# Container health = the service this MODE is responsible for actually answers.
# all/api: the API must report the model loaded (not just "some port is open").
# ui:      Streamlit's own health endpoint.
# Uses Python instead of curl so the image doesn't need curl (fewer CVEs, docs/09 §4).
case "${MODE:-all}" in
  ui)
    python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
    ;;
  *)
    python -c "import json, urllib.request as u; r = json.load(u.urlopen('http://localhost:8000/health', timeout=4)); raise SystemExit(0 if r.get('model_loaded') else 1)"
    ;;
esac
