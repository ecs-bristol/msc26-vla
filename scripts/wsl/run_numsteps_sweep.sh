#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment A: flow-matching num_steps sweep on the official PC-local harness.
#
# Usage:
#   SUITE=libero_spatial N_EPISODES=1 MODEL_REVISION=<rev> \
#     bash scripts/wsl/run_numsteps_sweep.sh
#
# Set NUM_STEPS_VALUES to override the default sweep grid.

SUITE="${SUITE:-libero_spatial}"
N_EPISODES="${N_EPISODES:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-280}"
MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
VALUES="${NUM_STEPS_VALUES:-10 8 5 3 2}"

for n in $VALUES; do
  echo "==================== num_steps=$n ===================="
  NUM_STEPS="$n" \
    SUITE="$SUITE" \
    N_EPISODES="$N_EPISODES" \
    EPISODE_LENGTH="$EPISODE_LENGTH" \
    MODEL_REVISION="$MODEL_REVISION" \
    bash scripts/wsl/run_official_pc_local_eval.sh
done
