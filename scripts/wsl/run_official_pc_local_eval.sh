#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
HF_HOME="${HF_HOME:-$HOME/vla/hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/vla/results}"
N_EPISODES="${N_EPISODES:-1}"
SUITE="${SUITE:-libero_spatial}"
EPISODE_LENGTH="${EPISODE_LENGTH:-280}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${SUITE}_pc_local_$RUN_STAMP}"

export HF_HOME MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUTPUT_DIR"

printf 'Official PC-local evaluation: suite=%s episodes_per_task=%s output=%s\n' \
  "$SUITE" "$N_EPISODES" "$OUTPUT_DIR"

exec lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.pretrained_revision="$MODEL_REVISION" \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.episode_length="$EPISODE_LENGTH" \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$OUTPUT_DIR"
