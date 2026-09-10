#!/usr/bin/env bash
# Pre-warm the streaming demo cache: trigger processing for every catalogue
# video that has no cached event log yet, one at a time (the registry dedups,
# and sequential keeps GPU contention sane). Live entries are skipped.
#   bash scripts/prewarm.sh [host]
set -euo pipefail
HOST="${1:-http://localhost:8031}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAX_WAIT_S=1800   # a failed job writes no cache — never wait forever

# Python on Windows writes \r\n: strip \r, or every id carries it and the
# cache check / wait loop look for a file name that can never exist
ids=$(curl -s "$HOST/api/videos" | .venv/Scripts/python.exe -c \
  "import json,sys; print('\n'.join(k for k, v in json.load(sys.stdin).items() if not v.get('live')))" \
  | tr -d '\r')

for id in $ids; do
  if [ -f "$ROOT/out_stream/$id.events.json" ]; then
    echo "cached: $id"
    continue
  fi
  echo "processing: $id"
  # opening the SSE endpoint starts the job; drop the connection after 3s
  curl -s -N --max-time 3 "$HOST/api/stream/$id" >/dev/null || true
  waited=0
  until [ -f "$ROOT/out_stream/$id.events.json" ] || [ "$waited" -ge "$MAX_WAIT_S" ]; do
    sleep 15
    waited=$((waited + 15))
  done
  if [ -f "$ROOT/out_stream/$id.events.json" ]; then
    echo "done: $id"
  else
    echo "timeout: $id (job failed or stalled — skipping)"
  fi
done
echo "catalogue pre-warm finished"
