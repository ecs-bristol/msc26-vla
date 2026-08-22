#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
POLICY_MODE="${POLICY_MODE:?POLICY_MODE must be native_h1, native_h20, or adaptive_fixed_h20}"
HF_HOME="${HF_HOME:-$HOME/vla/hf-cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/vla/results}"
N_EPISODES="${N_EPISODES:-1}"
SUITE="${SUITE:-libero_spatial}"
EPISODE_LENGTH="${EPISODE_LENGTH:-280}"
NUM_STEPS="${NUM_STEPS:-2}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
EVAL_SEED="${EVAL_SEED:-1000}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${SUITE}_pc_local_$RUN_STAMP}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAIRED_SEED_MANIFEST="${PAIRED_SEED_MANIFEST:-$OUTPUT_DIR/paired_seed_manifest.json}"

export HF_HOME HF_HUB_CACHE MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$PROJECT_ROOT/scripts/analysis/capture_official_eval_provenance.py" ]]; then
  printf 'Missing provenance helper under PROJECT_ROOT=%s\n' "$PROJECT_ROOT" >&2
  exit 2
fi

if [[ "$SUITE" != "libero_spatial" ]]; then
  printf 'The paired manifest generator currently supports only libero_spatial, got %s\n' "$SUITE" >&2
  exit 2
fi
if [[ "$NUM_STEPS" != "2" || "$CHUNK_SIZE" != "50" ]]; then
  printf 'Frozen protocol requires NUM_STEPS=2 and CHUNK_SIZE=50, got %s and %s\n' "$NUM_STEPS" "$CHUNK_SIZE" >&2
  exit 2
fi

policy_args=()
policy_type=""
native_n_action_steps=""
fixed_h=""
plugin_distribution=""
case "$POLICY_MODE" in
  native_h1|native_h20)
    policy_type="smolvla"
    native_n_action_steps="${POLICY_MODE#native_h}"
    policy_args=(
      "--policy.path=$CHECKPOINT"
      "--policy.pretrained_revision=$MODEL_REVISION"
      "--policy.n_action_steps=$native_n_action_steps"
      "--policy.num_steps=$NUM_STEPS"
      "--policy.chunk_size=$CHUNK_SIZE"
    )
    ;;
  adaptive_fixed_h20)
    policy_type="smolvla_adaptive"
    fixed_h="20"
    plugin_distribution="lerobot_policy_smolvla_adaptive"
    BASE_SNAPSHOT_PATH="${BASE_SNAPSHOT_PATH:?BASE_SNAPSHOT_PATH must name the frozen SmolVLA snapshot}"
    VLM_SNAPSHOT_PATH="${VLM_SNAPSHOT_PATH:?VLM_SNAPSHOT_PATH must name the frozen SmolVLM2 snapshot}"
    policy_args=(
      "--policy.type=$policy_type"
      "--policy.base_checkpoint=$CHECKPOINT"
      "--policy.base_revision=$MODEL_REVISION"
      "--policy.base_snapshot_path=$BASE_SNAPSHOT_PATH"
      "--policy.vlm_snapshot_path=$VLM_SNAPSHOT_PATH"
      "--policy.fixed_h=$fixed_h"
      "--policy.num_steps=$NUM_STEPS"
      --policy.precision=fp16
      "--policy.base_cache_dir=$HF_HUB_CACHE"
      --policy.local_files_only=true
      --policy.device=cuda
      --policy.use_amp=true
    )
    ;;
  *)
    printf 'Unsupported POLICY_MODE=%s; expected native_h1, native_h20, or adaptive_fixed_h20\n' "$POLICY_MODE" >&2
    exit 2
    ;;
esac

if [[ -n "$native_n_action_steps" ]]; then
  provenance_horizon_args=(--native-n-action-steps "$native_n_action_steps")
else
  provenance_horizon_args=(--fixed-h "$fixed_h" --plugin-distribution "$plugin_distribution")
fi

cmd=(
  lerobot-eval
  "${policy_args[@]}"
  --env.type=libero
  "--env.task=$SUITE"
  "--env.episode_length=$EPISODE_LENGTH"
  "--eval.n_episodes=$N_EPISODES"
  --eval.batch_size=1
  --env.max_parallel_tasks=1
  "--seed=$EVAL_SEED"
  "--output_dir=$OUTPUT_DIR"
)

printf 'Official PC-local evaluation: mode=%s policy_type=%s suite=%s episodes_per_task=%s num_steps=%s chunk_size=%s seed=%s output=%s\n' \
  "$POLICY_MODE" "$policy_type" "$SUITE" "$N_EPISODES" "$NUM_STEPS" "$CHUNK_SIZE" "$EVAL_SEED" "$OUTPUT_DIR"
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
  --policy-mode "$POLICY_MODE" \
  --policy-type "$policy_type" \
  --chunk-size "$CHUNK_SIZE" \
  --num-steps "$NUM_STEPS" \
  --episode-length "$EPISODE_LENGTH" \
  --manifest-path "$PAIRED_SEED_MANIFEST" \
  "${provenance_horizon_args[@]}" \
  --write-launcher-resolved-config

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
  --policy-mode "$POLICY_MODE" \
  --policy-type "$policy_type" \
  --chunk-size "$CHUNK_SIZE" \
  --num-steps "$NUM_STEPS" \
  --episode-length "$EPISODE_LENGTH" \
  --manifest-path "$PAIRED_SEED_MANIFEST" \
  "${provenance_horizon_args[@]}" \
  --exit-code "$status"

exit "$status"
