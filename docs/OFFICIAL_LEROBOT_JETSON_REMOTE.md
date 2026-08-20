# Official LeRobot Evaluation with Jetson Inference

This is the only supported Jetson benchmark path for this project.

- WSL/Linux owns LIBERO, MuJoCo, the 280-step rollout, success detection, videos, and aggregate metrics.
- Jetson owns SmolVLA inference only.
- `lerobot-eval` is the benchmark runner.
- `remote_jetson` is a thin LeRobot policy plugin. It does not implement a second benchmark loop.

The paired PC-local baseline also uses official `lerobot-eval`; see
`docs/PC_LOCAL_SMOLVLA_PAPER_ALIGNMENT.md`.

## 1. Start the Jetson service

From Windows PowerShell:

```powershell
ssh msc26vla@10.42.0.2
```

On first connection, type `yes` when asked to trust the host key.

On Jetson:

```bash
cd ~/vla/project
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
./scripts/jetson/start_smolvla_libero_service.sh offline
```

Keep that terminal open. The ready line is:

```text
policy service listening on http://0.0.0.0:8081
```

## 2. Prepare WSL

Open a separate Ubuntu/WSL terminal:

```bash
source ~/vla/lerobot-libero/bin/activate
export PROJECT_ROOT="/mnt/d/Bristol_IOT_with_AI/Capstone Project/.worktrees/libero-yaml-cli-v1/Final_Project/LIBERO_Benchmark_Platform"
cd "$PROJECT_ROOT"
bash scripts/wsl/install_remote_jetson_policy.sh
```

Set the immutable experiment identity and verify the service:

```bash
export JETSON_ENDPOINT=http://10.42.0.2:8081
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
bash scripts/wsl/run_jetson_remote_preflight.sh
```

## 3. Run the official evaluation

One episode for each of the ten LIBERO Spatial tasks:

```bash
N_EPISODES=1 bash scripts/wsl/run_official_jetson_remote_eval.sh
```

The script invokes official `lerobot-eval` with one environment at a time. LIBERO's official 280-step limit and early success termination remain under the official evaluator.

Results are written under:

```text
~/vla/results/libero_spatial_jetson_remote_<UTC timestamp>/
```

The official LeRobot output and videos are the primary benchmark evidence. `remote_transport.jsonl` records endpoint latency and server metadata as supplementary deployment evidence.

## Retired path

The custom YAML benchmark runner is retired for Jetson benchmark claims. Historical source and outputs remain for provenance, but no active Jetson remote YAML configuration is kept.
