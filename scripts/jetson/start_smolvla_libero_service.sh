#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
export CHECKPOINT MODEL_REVISION

case "$MODE" in
  bootstrap)
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
    ;;
  offline)
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    ;;
  *)
    printf 'Usage: %s {bootstrap|offline}\n' "$0" >&2
    exit 2
    ;;
esac

printf 'Starting Jetson SmolVLA service: checkpoint=%s revision=%s mode=%s endpoint=http://0.0.0.0:8081\n' \
  "$CHECKPOINT" "$MODEL_REVISION" "$MODE"

exec "$(dirname "$0")/run_container.sh" python3 -m libero_platform serve-policy \
  --policy smolvla_libero \
  --checkpoint "$CHECKPOINT" \
  --revision "$MODEL_REVISION" \
  --precision fp16 \
  --host 0.0.0.0 \
  --port 8081
