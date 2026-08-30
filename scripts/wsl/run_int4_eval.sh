#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment B: quantized SmolVLA on the official PC-local harness via
# the lerobot_policy_smolvla_int4 plugin.
#
# Usage:
#   QUANT_METHOD=int8_groupwise NUM_STEPS=10 SUITE=libero_spatial N_EPISODES=1 \
#     MODEL_REVISION=<rev> bash scripts/wsl/run_int4_eval.sh
#   QUANT_SCOPE=backbone ... # optional: language (default) or backbone

MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
HF_HOME="${HF_HOME:-$HOME/vla/hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/vla/results}"
N_EPISODES="${N_EPISODES:-1}"
SUITE="${SUITE:-libero_spatial}"
EPISODE_LENGTH="${EPISODE_LENGTH:-280}"
QUANT_METHOD="${QUANT_METHOD:-int4_groupwise}"
QUANT_SCOPE="${QUANT_SCOPE:-}"
GROUP_SIZE="${GROUP_SIZE:-}"
VISION_BITS="${VISION_BITS:-}"
CONNECTOR_BITS="${CONNECTOR_BITS:-}"
TEXT_BITS="${TEXT_BITS:-}"
NUM_STEPS="${NUM_STEPS:-}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SCOPE_TAG="${QUANT_SCOPE:-language}"
STEPS_TAG="${NUM_STEPS:-default}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${SUITE}_int4_${QUANT_METHOD}_${SCOPE_TAG}_ns${STEPS_TAG}_$RUN_STAMP}"

export HF_HOME MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUTPUT_DIR"

printf 'SmolVLA int4 evaluation: suite=%s episodes_per_task=%s quant=%s num_steps=%s output=%s\n' \
  "$SUITE" "$N_EPISODES" "$QUANT_METHOD" "${NUM_STEPS:-checkpoint-default}" "$OUTPUT_DIR"

EXTRA_ARGS=()
if [ -n "$NUM_STEPS" ]; then
  EXTRA_ARGS+=(--policy.num_steps="$NUM_STEPS")
fi
if [ -n "$QUANT_SCOPE" ]; then
  EXTRA_ARGS+=(--policy.quant_scope="$QUANT_SCOPE")
fi
if [ -n "$GROUP_SIZE" ]; then
  EXTRA_ARGS+=(--policy.group_size="$GROUP_SIZE")
fi
if [ -n "$VISION_BITS" ]; then
  EXTRA_ARGS+=(--policy.vision_bits="$VISION_BITS")
fi
if [ -n "$CONNECTOR_BITS" ]; then
  EXTRA_ARGS+=(--policy.connector_bits="$CONNECTOR_BITS")
fi
if [ -n "$TEXT_BITS" ]; then
  EXTRA_ARGS+=(--policy.text_bits="$TEXT_BITS")
fi

exec lerobot-eval \
  --policy.type=smolvla_int4 \
  --policy.checkpoint="$CHECKPOINT" \
  --policy.revision="$MODEL_REVISION" \
  --policy.quant_method="$QUANT_METHOD" \
  "${EXTRA_ARGS[@]}" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.episode_length="$EPISODE_LENGTH" \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$OUTPUT_DIR"
