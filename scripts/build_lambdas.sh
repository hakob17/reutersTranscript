#!/usr/bin/env bash
# Stage Lambda bundles into build/lambdas/<name>/ for Terraform's archive_file.
# Run from the repo root before `terraform plan`:
#   bash scripts/build_lambdas.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build/lambdas"
rm -rf "$BUILD"

for fn in ingest attribute notify_review review_callback publish; do
  mkdir -p "$BUILD/$fn"
  cp "$ROOT/services/lambdas/$fn/handler.py" "$BUILD/$fn/"
done

# attribute needs the shared package + the anthropic SDK (boto3 ships with the
# runtime). cv2 is NOT needed: detection runs on the GPU task, and chyron.py
# imports it lazily.
cp -r "$ROOT/speaker_attribution" "$BUILD/attribute/speaker_attribution"
find "$BUILD/attribute/speaker_attribution" -name __pycache__ -type d -exec rm -rf {} +
python -m pip install --quiet --target "$BUILD/attribute" \
  --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.12 \
  "anthropic>=0.40"

echo "lambda bundles staged in $BUILD"
