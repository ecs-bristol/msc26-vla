# LIBERO SmolVLA Edge Benchmark

Evaluate [HuggingFaceVLA/smolvla_libero](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
on the LIBERO Spatial benchmark under two deployments:

1. **PC-Local** — simulation and policy inference both run on a workstation (WSL Ubuntu + NVIDIA GPU).
2. **PC simulation + Jetson remote inference** — the simulator runs on the workstation while the
   policy runs on a Jetson Orin Nano, served over HTTP.

The benchmark uses the **official `lerobot-eval`** CLI. The custom YAML runner in this repository is
retired and kept only for historical provenance.

## Results (LIBERO Spatial, 1 episode per task, 280 steps, FP16)

| Deployment | Success |
|---|---|
| PC-Local | 8/10 (80%) |
| PC simulation + Jetson remote inference | 9/10 (90%) |

Detailed per-task records are in [`evidence/latest/`](evidence/latest/).

## Architecture

- **WSL Ubuntu** owns the LIBERO simulator, MuJoCo, rollout execution, success detection, video
  recording, and metric aggregation.
- **Jetson Orin Nano** owns SmolVLA inference only and exposes an HTTP policy service.
- `remote_jetson` (see [`plugins/`](plugins/lerobot_policy_remote_jetson/)) is a thin LeRobot policy
  plugin that forwards observations to the Jetson and returns actions; it does not implement a second
  benchmark loop.

## Fixed experiment identity

```text
CHECKPOINT     = HuggingFaceVLA/smolvla_libero
MODEL_REVISION = 6721902bc4d61e50a3bfdb11dfb4cb626f05d102
PRECISION      = fp16
SUITE          = libero_spatial (10 tasks)
EPISODE_LENGTH = 280 steps (official limit, early success termination)
MUJOCO_GL      = egl
```

## Requirements

- Workstation: WSL2 Ubuntu with Python 3.12, an NVIDIA GPU, and EGL-capable MuJoCo.
- Jetson: Orin Nano with Docker and NVIDIA Container Toolkit (`--runtime nvidia`).
- Network access to Hugging Face on first run (models and datasets are cached under `HF_HOME`).

## Quick start

### 1. WSL Ubuntu (simulation & evaluation)

```bash
python3.12 -m venv ~/vla/lerobot-libero
source ~/vla/lerobot-libero/bin/activate

git clone --branch v0.6.1 --depth 1 https://github.com/huggingface/lerobot.git ~/vla/lerobot
cd ~/vla/lerobot
python -m pip install -e '.[libero,smolvla,evaluation]'

export HF_HOME=~/vla/hf-cache
```

### 2. PC-Local baseline

```bash
source ~/vla/lerobot-libero/bin/activate
export PROJECT_ROOT="<path-to-this-repo>"
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
cd "$PROJECT_ROOT"
bash scripts/wsl/run_official_pc_local_eval.sh
```

Results are written to `~/vla/results/libero_spatial_pc_local_<UTC timestamp>/`.

### 3. PC simulation + Jetson remote inference

**Terminal 1 — Jetson (start the policy service):**

```bash
cd <this-repo-on-jetson>
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
./scripts/jetson/start_smolvla_libero_service.sh offline   # or "bootstrap" on first run
```

Wait for `policy service listening on http://0.0.0.0:8081`.

**Terminal 2 — WSL (install plugin, preflight, evaluate):**

```bash
source ~/vla/lerobot-libero/bin/activate
export PROJECT_ROOT="<path-to-this-repo>"
cd "$PROJECT_ROOT"
bash scripts/wsl/install_remote_jetson_policy.sh   # once per environment

export JETSON_ENDPOINT=http://10.42.0.2:8081
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
bash scripts/wsl/run_jetson_remote_preflight.sh
N_EPISODES=1 bash scripts/wsl/run_official_jetson_remote_eval.sh
```

Results are written to `~/vla/results/libero_spatial_jetson_remote_<UTC timestamp>/`.

## Repository layout

```text
configs/                 Legacy YAML experiment configs (retired, historical)
docker/jetson/           Jetson Docker image definition
docs/                    Protocols and quick-start guide (START_GUIDE.md)
evidence/latest/         Current benchmark baseline evidence
plugins/                 lerobot_policy_remote_jetson (HTTP policy plugin)
scripts/
  jetson/                Jetson service scripts (Docker container)
  wsl/                   WSL evaluation scripts (PC-local and Jetson remote)
  analysis/              Success-rate analysis utilities
src/libero_platform/     Legacy benchmark platform source (retired runner)
tests/                   Unit tests
```

## Documentation

- `docs/START_GUIDE.md` — full setup and experiment walkthrough (Chinese)
- `docs/PC_LOCAL_SMOLVLA_PAPER_ALIGNMENT.md` — PC-local protocol
- `docs/OFFICIAL_LEROBOT_JETSON_REMOTE.md` — Jetson remote protocol
- `README_ZH.md` — Chinese project description
