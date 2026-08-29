#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/13636/Documents/Codex/2026-08-15/sh/outputs/libero_lerobot_adapter_v2
source "$HOME/lerobot/.venv/bin/activate"

# The router automatically excludes seed 44 from its historical evidence, so
# these three runs share initial conditions without leaking baseline outcomes
# into the routing decision.
python3 interactive_grasp.py --all-tasks --attempts 3 --batch-size 1 --strategy native --seed 44 --run
python3 interactive_grasp.py --all-tasks --attempts 3 --batch-size 1 --strategy smooth --seed 44 --run
python3 interactive_grasp.py --all-tasks --attempts 3 --batch-size 1 --strategy router --seed 44 --run

# Hybrid recovery is kept separate because it adds recovery rollouts and is not
# part of the three-way matched success comparison.
if [[ "${RUN_HYBRID:-0}" == "1" ]]; then
  python3 interactive_grasp.py --all-tasks --attempts 3 --batch-size 1 --strategy hybrid --seed 44 --run
fi
