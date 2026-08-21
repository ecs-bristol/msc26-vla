#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
HF_HOME="${HF_HOME:-$HOME/vla/hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/vla/results}"
N_EPISODES="${N_EPISODES:-1}"
SUITE="${SUITE:-libero_spatial}"
EPISODE_LENGTH="${EPISODE_LENGTH:-280}"
NUM_STEPS="${NUM_STEPS:-2}"
EVAL_SEED="${EVAL_SEED:-1000}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${SUITE}_pc_local_$RUN_STAMP}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$PROJECT_ROOT/scripts/analysis/capture_official_eval_provenance.py" ]]; then
  printf 'Missing provenance helper under PROJECT_ROOT=%s\n' "$PROJECT_ROOT" >&2
  exit 2
fi

if [[ "$SUITE" != "libero_spatial" ]]; then
  printf 'The paired manifest generator currently supports only libero_spatial, got %s\n' "$SUITE" >&2
  exit 2
fi

cmd=(
  lerobot-eval
  "--policy.path=$CHECKPOINT"
  "--policy.pretrained_revision=$MODEL_REVISION"
  "--policy.num_steps=$NUM_STEPS"
  --env.type=libero
  "--env.task=$SUITE"
  "--env.episode_length=$EPISODE_LENGTH"
  "--eval.n_episodes=$N_EPISODES"
  --eval.batch_size=1
  --env.max_parallel_tasks=1
  "--seed=$EVAL_SEED"
  "--output_dir=$OUTPUT_DIR"
)

printf 'Official PC-local evaluation: suite=%s episodes_per_task=%s num_steps=%s seed=%s output=%s\n' \
  "$SUITE" "$N_EPISODES" "$NUM_STEPS" "$EVAL_SEED" "$OUTPUT_DIR"
printf '%q ' "${cmd[@]}" > "$OUTPUT_DIR/command.sh"
printf '\n' >> "$OUTPUT_DIR/command.sh"
chmod 600 "$OUTPUT_DIR/command.sh"

python3 "$PROJECT_ROOT/scripts/analysis/capture_official_eval_provenance.py" \
  --output-dir "$OUTPUT_DIR" \
  --project-root "$PROJECT_ROOT" \
  --suite "$SUITE" \
  --episodes-per-task "$N_EPISODES" \
  --seed "$EVAL_SEED" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$MODEL_REVISION" \
  --num-steps "$NUM_STEPS" \
  --episode-length "$EPISODE_LENGTH"

set +e
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/stdout_stderr.log"
status="${PIPESTATUS[0]}"
set -e

# LeRobot v0.6.1 logs its parsed EvalPipelineConfig at startup. Retain the
# unfiltered source log under this explicit name so the resolved config can be audited.
cp "$OUTPUT_DIR/stdout_stderr.log" "$OUTPUT_DIR/resolved_config.log"

python3 "$PROJECT_ROOT/scripts/analysis/capture_official_eval_provenance.py" \
  --output-dir "$OUTPUT_DIR" \
  --project-root "$PROJECT_ROOT" \
  --suite "$SUITE" \
  --episodes-per-task "$N_EPISODES" \
  --seed "$EVAL_SEED" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-revision "$MODEL_REVISION" \
  --num-steps "$NUM_STEPS" \
  --episode-length "$EPISODE_LENGTH" \
  --exit-code "$status"

exit "$status"
