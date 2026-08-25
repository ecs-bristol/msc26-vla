#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
NUM_STEPS="${NUM_STEPS:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
QUANT_METHOD="${QUANT_METHOD:-none}"
QUANT_SCOPE="${QUANT_SCOPE:-language}"
VISION_BITS="${VISION_BITS:-4}"
CONNECTOR_BITS="${CONNECTOR_BITS:-8}"
TEXT_BITS="${TEXT_BITS:-8}"
export CHECKPOINT MODEL_REVISION NUM_STEPS N_ACTION_STEPS CHUNK_SIZE \
  QUANT_METHOD QUANT_SCOPE VISION_BITS CONNECTOR_BITS TEXT_BITS

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

printf 'Starting Jetson SmolVLA service: checkpoint=%s revision=%s mode=%s num_steps=%s n_action_steps=%s chunk_size=%s quant=%s/%s endpoint=http://0.0.0.0:8081\n' \
  "$CHECKPOINT" "$MODEL_REVISION" "$MODE" "$NUM_STEPS" "$N_ACTION_STEPS" \
  "$CHUNK_SIZE" "$QUANT_METHOD" "$QUANT_SCOPE"

exec "$(dirname "$0")/run_container.sh" python3 -m libero_platform serve-policy \
  --policy smolvla_libero \
  --checkpoint "$CHECKPOINT" \
  --revision "$MODEL_REVISION" \
  --precision fp16 \
  --num-steps "$NUM_STEPS" \
  --n-action-steps "$N_ACTION_STEPS" \
  --chunk-size "$CHUNK_SIZE" \
  --quant-method "$QUANT_METHOD" \
  --quant-scope "$QUANT_SCOPE" \
  --vision-bits "$VISION_BITS" \
  --connector-bits "$CONNECTOR_BITS" \
  --text-bits "$TEXT_BITS" \
  --host 0.0.0.0 \
  --port 8081
