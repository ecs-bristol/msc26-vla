#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
HF_HOME="${HF_HOME:-$HOME/vla/hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/vla/results}"
N_EPISODES="${N_EPISODES:-1}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/libero_spatial_pc_local_$RUN_STAMP}"

export HF_HOME MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUTPUT_DIR"

printf 'Official PC-local evaluation: suite=libero_spatial episodes_per_task=%s output=%s\n' \
  "$N_EPISODES" "$OUTPUT_DIR"

exec lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.pretrained_revision="$MODEL_REVISION" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.episode_length=280 \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$OUTPUT_DIR"
