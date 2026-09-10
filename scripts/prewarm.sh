#!/usr/bin/env bash
# Pre-warm the streaming demo cache: trigger processing for every catalogue
# video that has no cached event log yet, one at a time (the registry dedups,
# and sequential keeps GPU contention sane).
#   bash scripts/prewarm.sh [host]
set -euo pipefail
HOST="${1:-http://localhost:8031}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ids=$(curl -s "$HOST/api/videos" | .venv/Scripts/python.exe -c \
  "import json,sys; print('\n'.join(json.load(sys.stdin).keys()))")

for id in $ids; do
  if [ -f "$ROOT/out_stream/$id.events.json" ]; then
    echo "cached: $id"
    continue
  fi
  echo "processing: $id"
  # opening the SSE endpoint starts the job; drop the connection after 3s
  curl -s -N --max-time 3 "$HOST/api/stream/$id" >/dev/null || true
  until [ -f "$ROOT/out_stream/$id.events.json" ]; do sleep 15; done
  echo "done: $id"
done
echo "catalogue fully pre-warmed"
