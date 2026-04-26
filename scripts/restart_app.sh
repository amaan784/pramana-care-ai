#!/usr/bin/env bash
# Free Edition apps auto-stop 24h after start. Run this within 23h of judging.
set -euo pipefail

TARGET="${1:-prod}"
APP_NAME="pramana-${TARGET}"

echo "→ Stopping $APP_NAME (ok if already stopped)"
databricks apps stop "$APP_NAME" || true

echo "→ Starting $APP_NAME"
databricks apps start "$APP_NAME"

echo "→ Waiting for app to come ONLINE"
for _ in $(seq 1 30); do
  STATE=$(databricks apps get "$APP_NAME" -o json | python -c "import sys, json; print(json.load(sys.stdin).get('compute_status', {}).get('state', '?'))")
  echo "    state=$STATE"
  [[ "$STATE" == "ACTIVE" || "$STATE" == "RUNNING" ]] && break
  sleep 10
done

databricks apps get "$APP_NAME"
