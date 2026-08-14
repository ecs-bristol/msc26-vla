# Current Experiment Baseline

This folder is the only retained experiment evidence for the current project direction.

## PC-Local Official Baseline

- Runtime: WSL Ubuntu with the official `lerobot-eval` CLI.
- Simulator: LIBERO, one episode for each of ten tasks per suite.
- Checkpoint: `HuggingFaceVLA/smolvla_libero`.
- Revision: `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
- Precision: FP16.
- Episode length: 280 steps.

| Suite | Success | Failed tasks | Primary record |
|---|---|---|---|
| libero_spatial | 8/10 (80%) | 2, 7 | `pc_local/libero_spatial_eval_info.json` |
| libero_object | 9/10 (90%) | 7 | `pc_local/libero_object_eval_info.json` |
| libero_goal | 8/10 (80%) | 3, 4 | `pc_local/libero_goal_eval_info.json` |

## PC Simulator With Jetson Remote Inference

- Simulator: the same WSL Ubuntu official `lerobot-eval` LIBERO Spatial protocol.
- Policy execution: Jetson Orin Nano over HTTP at `http://10.42.0.2:8081`.
- Checkpoint, revision, precision, and episode length: identical to the PC-local baseline.
- Result: 9 / 10 tasks successful (90%).
- Primary records: `jetson_remote/eval_info.json` and `jetson_remote/remote_transport.jsonl`.

## Retained Operational Path

1. Start the Jetson policy service with `scripts/jetson/start_smolvla_libero_service.sh`.
2. Run PC-local evaluation with `scripts/wsl/run_official_pc_local_eval.sh`.
3. Run PC simulation plus Jetson inference with `scripts/wsl/run_official_jetson_remote_eval.sh`.

The legacy YAML experiment runner and its historical outputs are intentionally retired. New results should be stored outside the retired `outputs/` directory and copied into this evidence folder only when they are selected as a baseline.
