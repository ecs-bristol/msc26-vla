#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
HF_HOME="${HF_HOME:-$HOME/vla/hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/vla/results}"
N_EPISODES="${N_EPISODES:-1}"
SUITE="${SUITE:-libero_spatial}"
EPISODE_LENGTH="${EPISODE_LENGTH:-280}"
NUM_STEPS="${NUM_STEPS:-}"
N_ACTION_STEPS="${N_ACTION_STEPS:-}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TAG=""
if [ -n "$NUM_STEPS" ]; then
  TAG="${TAG}_ns${NUM_STEPS}"
fi
if [ -n "$N_ACTION_STEPS" ]; then
  TAG="${TAG}_na${N_ACTION_STEPS}"
fi
if [ -n "$CHUNK_SIZE" ]; then
  TAG="${TAG}_cs${CHUNK_SIZE}"
fi
if [ -n "$TAG" ]; then
  OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${SUITE}_pc_local${TAG}_$RUN_STAMP}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${SUITE}_pc_local_$RUN_STAMP}"
fi

export HF_HOME MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUTPUT_DIR"

printf 'Official PC-local evaluation: suite=%s episodes_per_task=%s num_steps=%s n_action_steps=%s chunk_size=%s output=%s\n' \
  "$SUITE" "$N_EPISODES" "${NUM_STEPS:-checkpoint-default}" \
  "${N_ACTION_STEPS:-checkpoint-default}" "${CHUNK_SIZE:-checkpoint-default}" "$OUTPUT_DIR"

NUM_STEPS_ARGS=()
if [ -n "$NUM_STEPS" ]; then
  NUM_STEPS_ARGS=(--policy.num_steps="$NUM_STEPS")
fi
N_ACTION_STEPS_ARGS=()
if [ -n "$N_ACTION_STEPS" ]; then
  N_ACTION_STEPS_ARGS=(--policy.n_action_steps="$N_ACTION_STEPS")
fi
CHUNK_SIZE_ARGS=()
if [ -n "$CHUNK_SIZE" ]; then
  CHUNK_SIZE_ARGS=(--policy.chunk_size="$CHUNK_SIZE")
fi

exec lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.pretrained_revision="$MODEL_REVISION" \
  "${NUM_STEPS_ARGS[@]}" \
  "${N_ACTION_STEPS_ARGS[@]}" \
  "${CHUNK_SIZE_ARGS[@]}" \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.episode_length="$EPISODE_LENGTH" \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$OUTPUT_DIR"
