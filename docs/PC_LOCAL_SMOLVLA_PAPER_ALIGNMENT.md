# Official PC-Local SmolVLA Baseline

The previous Windows custom-runner workflow is retired. New benchmark evidence
must be produced by the official LeRobot evaluator in WSL/Linux.

## Fixed protocol

- evaluator: official `lerobot-eval`
- model: `HuggingFaceVLA/smolvla_libero`
- revision: `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
- suite: `libero_spatial`, all ten tasks
- episodes: one per task for the paired deployment experiment
- maximum steps: 280 with official early-success termination
- batch size and parallel tasks: one
- MuJoCo: 3.3.2 with EGL

## Run

In Ubuntu/WSL:

```bash
source ~/vla/lerobot-libero/bin/activate
export PROJECT_ROOT="/mnt/d/Bristol_IOT_with_AI/Capstone Project/.worktrees/libero-yaml-cli-v1/Final_Project/LIBERO_Benchmark_Platform"
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
cd "$PROJECT_ROOT"
bash scripts/wsl/run_official_pc_local_eval.sh
```

Results are written under `~/vla/results/libero_spatial_pc_local_<UTC timestamp>`.
These official outputs are the PC-local baseline paired with the Jetson remote
run documented in `docs/OFFICIAL_LEROBOT_JETSON_REMOTE.md`.
