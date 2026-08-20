# Official LeRobot Eval With Jetson Remote Policy Implementation Plan

> **Execution:** Implement task by task with tests before production code and verification after each task.

**Goal:** Replace the active custom LIBERO runner with official `lerobot-eval` plus a thin Jetson remote-policy plugin.

**Architecture:** WSL runs the official evaluator and LIBERO environment. An installed `lerobot_policy_remote_jetson` plugin translates official observations to the existing Jetson HTTP service. Jetson performs SmolVLA preprocessing, FP16 inference, and postprocessing. Official LeRobot outputs are the benchmark evidence.

**Tech stack:** Python 3.12, LeRobot 0.6.1 plugin API, hf-libero 0.1.4, MuJoCo 3.3.2, Torch, requests, Pillow, pytest, Bash.

## Task 1: Freeze The Architecture And Legacy Boundary

**Files:**
- Create: `docs/superpowers/specs/2026-08-13-official-lerobot-eval-jetson-remote-policy-design.md`
- Create: `docs/superpowers/plans/2026-08-13-official-lerobot-eval-jetson-remote-policy.md`

**Implementation:** Record that official `lerobot-eval` owns all benchmark semantics, the plugin owns transport only, and the custom runner is legacy evidence only.

**Verification:**
```bash
grep -n "only authority\|legacy evidence" docs/superpowers/specs/2026-08-13-official-lerobot-eval-jetson-remote-policy-design.md
```

## Task 2: Add The Installable LeRobot Policy Plugin Skeleton

**Files:**
- Create: `plugins/lerobot_policy_remote_jetson/pyproject.toml`
- Create: `plugins/lerobot_policy_remote_jetson/src/lerobot_policy_remote_jetson/__init__.py`
- Create: `plugins/lerobot_policy_remote_jetson/src/lerobot_policy_remote_jetson/configuration_remote_jetson.py`
- Create: `plugins/lerobot_policy_remote_jetson/src/lerobot_policy_remote_jetson/modeling_remote_jetson.py`
- Create: `plugins/lerobot_policy_remote_jetson/src/lerobot_policy_remote_jetson/processor_remote_jetson.py`
- Create: `plugins/lerobot_policy_remote_jetson/tests/test_registration.py`

**Complete registration code:**
```python
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("remote_jetson")
@dataclass
class RemoteJetsonConfig(PreTrainedConfig):
    endpoint: str = "http://10.42.0.2:8081"
    checkpoint: str = "HuggingFaceVLA/smolvla_libero"
    revision: str = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    precision: str = "fp16"
    request_timeout_s: float = 180.0
    telemetry_path: Path | None = None
    seed: int = 42
```

Implement all abstract config methods with no training optimizer or scheduler support, because this policy is inference-only.

**Verification:**
```bash
python -m pip install -e ./plugins/lerobot_policy_remote_jetson
python -c "from lerobot.configs.policies import PreTrainedConfig; import lerobot_policy_remote_jetson; print(PreTrainedConfig.get_choice_class('remote_jetson'))"
```

## Task 3: Implement Observation Serialization And Response Validation

**Files:**
- Create: `plugins/lerobot_policy_remote_jetson/src/lerobot_policy_remote_jetson/transport.py`
- Create: `plugins/lerobot_policy_remote_jetson/tests/test_transport.py`

**Complete behavior:**
```python
def serialize_observation(observation: dict[str, object]) -> dict[str, object]:
    return {
        "instruction": one_task_string(observation["task"]),
        "images": {
            "agentview": encode_png(observation["observation.images.image"]),
            "robot0_eye_in_hand": encode_png(observation["observation.images.image2"]),
        },
        "proprioception": one_state_vector(observation["observation.state"], expected=8),
    }
```

Accept NumPy arrays and Torch tensors in HWC or CHW form, with or without one batch dimension. Convert float images in `[0, 1]` back to uint8 exactly once. Reject missing cameras, batch sizes other than one, state shapes other than eight, non-finite values, and response actions other than seven finite values.

**Verification:**
```bash
python -m pytest plugins/lerobot_policy_remote_jetson/tests/test_transport.py -v
```

## Task 4: Implement The Inference-Only Policy And Identity Processors

**Files:**
- Modify: `plugins/lerobot_policy_remote_jetson/src/lerobot_policy_remote_jetson/modeling_remote_jetson.py`
- Modify: `plugins/lerobot_policy_remote_jetson/src/lerobot_policy_remote_jetson/processor_remote_jetson.py`
- Create: `plugins/lerobot_policy_remote_jetson/tests/test_policy.py`

**Complete policy flow:**
```python
def reset(self) -> None:
    self._needs_remote_reset = True

@torch.inference_mode()
def select_action(self, batch: dict[str, object]) -> torch.Tensor:
    payload = serialize_observation(batch)
    if self._needs_remote_reset:
        self._client.reset(instruction=payload["instruction"], seed=self.config.seed)
        self._needs_remote_reset = False
    response = self._client.predict(payload)
    return torch.tensor([validate_action(response)], dtype=torch.float32)
```

`forward` raises `NotImplementedError` because this plugin does not train. `get_optim_params` returns an empty list. The processor factory returns pipelines that do not apply checkpoint normalization or action unnormalization; those remain on Jetson.

**Verification:**
```bash
python -m pytest plugins/lerobot_policy_remote_jetson/tests/test_policy.py -v
```

## Task 5: Add Official Preflight And Evaluation Scripts

**Files:**
- Create: `scripts/wsl/install_remote_jetson_policy.sh`
- Replace: `scripts/wsl/run_jetson_remote_preflight.sh`
- Create: `scripts/wsl/run_official_jetson_remote_eval.sh`
- Create: `tests/test_official_remote_scripts.py`

**Official evaluation command generated by the script:**
```bash
lerobot-eval \
  --policy.type=remote_jetson \
  --policy.endpoint="$JETSON_ENDPOINT" \
  --policy.checkpoint=HuggingFaceVLA/smolvla_libero \
  --policy.revision="$MODEL_REVISION" \
  --policy.precision=fp16 \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$OUTPUT_DIR"
```

Preflight calls `/health`, verifies schema, checkpoint, revision, and precision, then verifies LeRobot plugin discovery. It does not create a LIBERO environment or run the custom CLI validator.

**Verification:**
```bash
python -m pytest tests/test_official_remote_scripts.py -v
bash scripts/wsl/run_jetson_remote_preflight.sh
```

## Task 6: Align Jetson Episode Reset With Official Policy Behavior

**Files:**
- Modify: `src/libero_platform/deployment/policy_service.py`
- Modify: `src/libero_platform/policies/smolvla_policy.py`
- Modify: `tests/test_policy_service.py`
- Modify: `tests/test_smolvla_policy.py`

**Implementation:** The official-service path must clear SmolVLA's action queue at episode reset and must not add project-specific clipping or action smoothing. Preserve the existing checkpoint preprocessor/postprocessor and return exactly one seven-dimensional postprocessed action. Keep deterministic diagnostics behind an explicit non-default flag.

**Verification:**
```bash
python -m pytest tests/test_policy_service.py tests/test_smolvla_policy.py -v
```

## Task 7: Retire The Custom Runner From The Active Workflow

**Files:**
- Delete: `configs/experiments/libero_spatial_jetson_remote_smolvla_smoke.yaml`
- Delete if present: `configs/experiments/libero_spatial_jetson_remote_smolvla_pilot.yaml`
- Delete if present: `configs/experiments/libero_spatial_jetson_remote_smolvla_formal.yaml`
- Modify: `README_ZH.md`
- Modify: `docs/superpowers/specs/2026-08-11-wsl-libero-jetson-remote-smolvla-design.md`

**Implementation:** Mark the prior design superseded and remove active commands that invoke `python -m libero_platform run`. Keep historical outputs and source modules until official remote verification is complete.

**Verification:**
```bash
grep -R "python -m libero_platform run" README_ZH.md scripts/wsl || true
test ! -e configs/experiments/libero_spatial_jetson_remote_smolvla_smoke.yaml
```

## Task 8: Run Official Discovery, Smoke, And Paired Verification

**Files:**
- Evidence only: `~/vla/results/libero_spatial_jetson_remote_smoke/`
- Evidence only: `~/vla/results/libero_spatial_jetson_remote_10task/`

**Verification sequence:**
```bash
python -m pip install -e ./plugins/lerobot_policy_remote_jetson
bash scripts/wsl/run_jetson_remote_preflight.sh
TASK_IDS=0 OUTPUT_DIR="$HOME/vla/results/libero_spatial_jetson_remote_smoke" bash scripts/wsl/run_official_jetson_remote_eval.sh
OUTPUT_DIR="$HOME/vla/results/libero_spatial_jetson_remote_10task" bash scripts/wsl/run_official_jetson_remote_eval.sh
```

The task is complete only when plugin discovery passes, the one-task official run writes valid official evidence, and the ten-task run is directly comparable with the 8/10 PC-local reference.
