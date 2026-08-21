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
EVAL_SEED="${EVAL_SEED:-1000}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${SUITE}_jetson_remote_$RUN_STAMP}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "$SUITE" != "libero_spatial" ]]; then
  printf 'The reportable Jetson protocol supports only libero_spatial, got %s\n' "$SUITE" >&2
  exit 2
fi
if [[ "$EPISODE_LENGTH" != "280" ]]; then
  printf 'The reportable Jetson protocol requires episode_length=280, got %s\n' "$EPISODE_LENGTH" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

python3 "$PROJECT_ROOT/scripts/analysis/capture_official_eval_provenance.py" \
  --output-dir "$OUTPUT_DIR" \
  --project-root "$PROJECT_ROOT" \
  --suite "$SUITE" \
  --episodes-per-task "$N_EPISODES" \
  --seed "$EVAL_SEED" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$MODEL_REVISION" \
  --episode-length "$EPISODE_LENGTH" \
  --write-launcher-resolved-config

printf 'Official LeRobot evaluation: suite=%s episodes_per_task=%s endpoint=%s output=%s\n' \
  "$SUITE" "$N_EPISODES" "$JETSON_ENDPOINT" "$OUTPUT_DIR"

exec lerobot-eval \
  --policy.type=remote_jetson \
  --policy.endpoint="$JETSON_ENDPOINT" \
  --policy.checkpoint="$CHECKPOINT" \
  --policy.revision="$MODEL_REVISION" \
  --policy.precision=fp16 \
  --policy.device=cpu \
  --policy.suite=libero_spatial \
  "--policy.seed=$EVAL_SEED" \
  --policy.telemetry_path="$OUTPUT_DIR/remote_transport.jsonl" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.episode_length=280 \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  "--seed=$EVAL_SEED" \
  --output_dir="$OUTPUT_DIR"
