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

## Software Optimization: Flow-Matching Step Sweep (Experiment A)

- Harness: PC-local official `lerobot-eval`, `libero_spatial`, 5 episodes per task.
- Method: sweep `--policy.num_steps` over 10, 8, 5, 3, 2 with the same fp16 checkpoint.
- Success rate (5 ep/task, 50 episodes): 10 steps 72.0%, 5 steps 72.0%, 2 steps 72.0%.
- Mean episode time: 90.8s (10) -> 63.1s (5) -> 46.8s (2), a 48% reduction at 2 steps.
- Records: `pc_local/num_steps/num_steps_summary.csv`, `num_steps_per_task.csv`, and figures.

## Software Optimization: VLM Weight Quantization (Experiment B)

- Harness: PC-local official `lerobot-eval` through the `smolvla_int4` LeRobot plugin.
- Method: self-contained per-group weight quantization of the SmolVLA language transformer.
- int8 language-only (5 ep/task): 10 steps 78.0%, 2 steps 80.0% vs fp16 72.0%.
- Negative result: uniform 4-bit PTQ degrades success (backbone 30%, language-only 10%, single-ep smoke).
- Memory: weights 1217.9 -> 929.4 MB (-23.7%); peak VRAM 1280.8 -> 992.9 MB (-22.5%).
- Latency: int8 is slightly faster than fp16 at every num_steps; ns2 mean 133ms.
- Records: `pc_local/int4/quant_summary.csv`, `quant_bench.csv`, and figures.

## Retained Operational Path

1. Start the Jetson policy service with `scripts/jetson/start_smolvla_libero_service.sh`.
2. Run PC-local evaluation with `scripts/wsl/run_official_pc_local_eval.sh`.
3. Run PC simulation plus Jetson inference with `scripts/wsl/run_official_jetson_remote_eval.sh`.

The legacy YAML experiment runner and its historical outputs are intentionally retired. New results should be stored outside the retired `outputs/` directory and copied into this evidence folder only when they are selected as a baseline.
