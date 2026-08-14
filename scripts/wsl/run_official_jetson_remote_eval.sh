#!/usr/bin/env bash
set -Eeuo pipefail

JETSON_ENDPOINT="${JETSON_ENDPOINT:?JETSON_ENDPOINT must be set}"
MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
HF_HOME="${HF_HOME:-$HOME/vla/hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/vla/results}"
N_EPISODES="${N_EPISODES:-1}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/libero_spatial_jetson_remote_$RUN_STAMP}"

export HF_HOME MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUTPUT_DIR"

printf 'Official LeRobot evaluation: suite=libero_spatial episodes_per_task=%s endpoint=%s output=%s\n' \
  "$N_EPISODES" "$JETSON_ENDPOINT" "$OUTPUT_DIR"

exec lerobot-eval \
  --policy.type=remote_jetson \
  --policy.endpoint="$JETSON_ENDPOINT" \
  --policy.checkpoint="$CHECKPOINT" \
  --policy.revision="$MODEL_REVISION" \
  --policy.precision=fp16 \
  --policy.device=cpu \
  --policy.telemetry_path="$OUTPUT_DIR/remote_transport.jsonl" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.episode_length=280 \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$OUTPUT_DIR"
