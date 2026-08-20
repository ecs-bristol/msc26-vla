#!/usr/bin/env bash
set -Eeuo pipefail

JETSON_ENDPOINT="${JETSON_ENDPOINT:?JETSON_ENDPOINT must be set}"
MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
HF_HOME="${HF_HOME:-$HOME/vla/hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/vla/results}"
N_EPISODES="${N_EPISODES:-1}"
SUITE="${SUITE:-libero_spatial}"
EPISODE_LENGTH="${EPISODE_LENGTH:-280}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${SUITE}_jetson_remote_$RUN_STAMP}"

export HF_HOME MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUTPUT_DIR"

printf 'Official LeRobot evaluation: suite=%s episodes_per_task=%s endpoint=%s output=%s\n' \
  "$SUITE" "$N_EPISODES" "$JETSON_ENDPOINT" "$OUTPUT_DIR"

exec lerobot-eval \
  --policy.type=remote_jetson \
  --policy.endpoint="$JETSON_ENDPOINT" \
  --policy.checkpoint="$CHECKPOINT" \
  --policy.revision="$MODEL_REVISION" \
  --policy.precision=fp16 \
  --policy.device=cpu \
  --policy.suite="$SUITE" \
  --policy.telemetry_path="$OUTPUT_DIR/remote_transport.jsonl" \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.episode_length="$EPISODE_LENGTH" \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$OUTPUT_DIR"
