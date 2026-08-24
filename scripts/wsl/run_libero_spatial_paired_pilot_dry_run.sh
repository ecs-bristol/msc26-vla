#!/usr/bin/env bash
set -Eeuo pipefail

# This launcher intentionally has no rollout mode. It never downloads or loads a model.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must name the dry-run artifact directory}"
BASE_SNAPSHOT_PATH="${BASE_SNAPSHOT_PATH:?BASE_SNAPSHOT_PATH must name the local frozen SmolVLA snapshot}"
VLM_SNAPSHOT_PATH="${VLM_SNAPSHOT_PATH:?VLM_SNAPSHOT_PATH must name the local frozen SmolVLM2 snapshot}"

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/analysis/libero_spatial_paired_pilot.py" \
  --dry-run \
  --output-dir "$OUTPUT_DIR" \
  --base-snapshot-path "$BASE_SNAPSHOT_PATH" \
  --vlm-snapshot-path "$VLM_SNAPSHOT_PATH"
