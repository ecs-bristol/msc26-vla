# PC-Local SmolVLA Paper Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one canonical PC-local YAML configuration run 100 LIBERO Spatial trials under explicit SmolVLA simulation-inference settings.

**Architecture:** A strict `SmolVLAInferenceSpec` is nested under `PolicySpec` and persists in the resolved experiment YAML. The CLI passes it to the local SmolVLA adapter, whose runtime applies it to the loaded checkpoint configuration before model creation. The existing official-alignment YAML becomes the single formal Spatial configuration.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, pytest, LeRobot 0.6.1, LIBERO, MuJoCo.

## Global Constraints

- Retain the canonical PC runtime at `D:\Bristol_IOT_with_AI\Capstone Project\.pc-smolvla`; create no Python environment.
- Modify the existing `libero_spatial_pc_local_smolvla_official_alignment.yaml`; do not add another formal Spatial alignment YAML.
- Formal protocol: ten tasks, ten benchmark initial states per task, one episode per state, 600 steps, zero settle steps, FP16, identity action control, no recordings of frames or video.
- SmolVLA simulation inference must use one executed action per observation and ten Flow Matching integration steps.
- Preserve existing action validation and raw/transformed action evidence.

---

### Task 1: Make SmolVLA Inference Settings Explicit in the Schema

**Files:**
- Modify: `src/libero_platform/spec.py:36-55`
- Modify: `tests/test_spec.py:1-91`

**Interfaces:**
- Produces `SmolVLAInferenceSpec(n_action_steps: int = 1, num_steps: int = 10)`.
- Produces `PolicySpec.smolvla_inference: SmolVLAInferenceSpec`.
- Later tasks consume `spec.policy.smolvla_inference`.

- [ ] **Step 1: Write the failing schema test**

```python
def test_smolvla_inference_settings_round_trip_through_resolved_spec(tmp_path: Path) -> None:
    payload = {
        **VALID,
        "policy": {
            **VALID["policy"],
            "smolvla_inference": {"n_action_steps": 1, "num_steps": 10},
        },
    }

    spec = ExperimentSpec.model_validate(payload)

    assert spec.policy.smolvla_inference.n_action_steps == 1
    assert spec.policy.smolvla_inference.num_steps == 10
```

- [ ] **Step 2: Run the test and verify it fails because `smolvla_inference` is forbidden**

Run:

```powershell
& 'D:\Bristol_IOT_with_AI\Capstone Project\.pc-smolvla\Scripts\python.exe' -m pytest tests\test_spec.py::test_smolvla_inference_settings_round_trip_through_resolved_spec -q
```

Expected: failure that identifies `policy.smolvla_inference` as an extra field.

- [ ] **Step 3: Add the strict settings model and PolicySpec field**

```python
class SmolVLAInferenceSpec(StrictModel):
    n_action_steps: StrictInt = Field(default=1, ge=1, le=50)
    num_steps: StrictInt = Field(default=10, ge=1, le=100)


class PolicySpec(StrictModel):
    key: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    precision: Literal["none", "fp32", "fp16", "bf16", "int8", "int4"]
    quantization: Literal["none", "int8", "int4"]
    action_control: ActionControlSpec = Field(default_factory=ActionControlSpec)
    smolvla_inference: SmolVLAInferenceSpec = Field(
        default_factory=SmolVLAInferenceSpec
    )
```

- [ ] **Step 4: Run the focused schema tests**

Run:

```powershell
& 'D:\Bristol_IOT_with_AI\Capstone Project\.pc-smolvla\Scripts\python.exe' -m pytest tests\test_spec.py -q
```

Expected: all tests pass.

### Task 2: Bind the Configuration to the Local SmolVLA Runtime

**Files:**
- Modify: `src/libero_platform/policies/smolvla_policy.py:1-185`
- Modify: `src/libero_platform/cli.py:286-314`
- Modify: `tests/test_smolvla_policy.py:1-220`
- Modify: `tests/test_cli.py:1-260`

**Interfaces:**
- `SmolVLAPolicyAdapter(..., smolvla_inference: SmolVLAInferenceSpec | None = None)` forwards settings to `LeRobotSmolVLARuntime`.
- `LeRobotSmolVLARuntime(..., smolvla_inference: SmolVLAInferenceSpec | None = None)` applies the fields after `PreTrainedConfig.from_pretrained`.
- `_apply_smolvla_inference_settings(config, settings)` raises `ValueError` if the requested action steps exceed the checkpoint chunk size.

- [ ] **Step 1: Write failing runtime-setting tests**

```python
def test_smolvla_runtime_applies_paper_simulation_inference_settings() -> None:
    module = importlib.import_module("libero_platform.policies.smolvla_policy")
    config = SimpleNamespace(chunk_size=50, n_action_steps=50, num_steps=30)
    settings = SmolVLAInferenceSpec(n_action_steps=1, num_steps=10)

    module._apply_smolvla_inference_settings(config, settings)

    assert config.n_action_steps == 1
    assert config.num_steps == 10


def test_smolvla_runtime_rejects_more_executed_actions_than_checkpoint_chunk() -> None:
    module = importlib.import_module("libero_platform.policies.smolvla_policy")
    config = SimpleNamespace(chunk_size=4, n_action_steps=1, num_steps=10)

    with pytest.raises(ValueError, match="chunk_size"):
        module._apply_smolvla_inference_settings(
            config, SmolVLAInferenceSpec(n_action_steps=5, num_steps=10)
        )
```

- [ ] **Step 2: Run the focused tests and verify they fail because the helper is missing**

Run:

```powershell
& 'D:\Bristol_IOT_with_AI\Capstone Project\.pc-smolvla\Scripts\python.exe' -m pytest tests\test_smolvla_policy.py::test_smolvla_runtime_applies_paper_simulation_inference_settings tests\test_smolvla_policy.py::test_smolvla_runtime_rejects_more_executed_actions_than_checkpoint_chunk -q
```

Expected: failure that reports `_apply_smolvla_inference_settings` is absent.

- [ ] **Step 3: Add the helper and thread settings through the adapter and CLI**

```python
def _apply_smolvla_inference_settings(config: object, settings: SmolVLAInferenceSpec) -> None:
    chunk_size = int(getattr(config, "chunk_size"))
    if settings.n_action_steps > chunk_size:
        raise ValueError(
            "SmolVLA n_action_steps must not exceed checkpoint chunk_size"
        )
    config.n_action_steps = settings.n_action_steps
    config.num_steps = settings.num_steps
```

Replace the existing direct `config.n_action_steps = 1` assignment with this helper. Pass `spec.policy.smolvla_inference` from `_build_policy` into `SmolVLAPolicyAdapter`.

- [ ] **Step 4: Run policy and CLI regression tests**

Run:

```powershell
& 'D:\Bristol_IOT_with_AI\Capstone Project\.pc-smolvla\Scripts\python.exe' -m pytest tests\test_smolvla_policy.py tests\test_cli.py -q
```

Expected: all tests pass.

### Task 3: Upgrade the Canonical Spatial Configuration and Evidence Guide

**Files:**
- Modify: `configs/experiments/libero_spatial_pc_local_smolvla_official_alignment.yaml:1-29`
- Modify: `tests/test_experiment_configs.py:90-122`
- Create: `docs/PC_LOCAL_SMOLVLA_PAPER_ALIGNMENT.md`

**Interfaces:**
- The canonical YAML resolves to exactly `10 * 10 * 1 = 100` trials.
- The guide exposes one validation command and one run command using the canonical PC interpreter.

- [ ] **Step 1: Write the failing canonical-config test**

```python
def test_pc_local_smolvla_official_alignment_is_a_one_hundred_trial_protocol(
    catalog_root: Path,
) -> None:
    path = catalog_root / "experiments" / "libero_spatial_pc_local_smolvla_official_alignment.yaml"
    spec = load_experiment_spec(path)

    assert spec.benchmark.task_ids == list(range(10))
    assert spec.benchmark.initial_state_ids == list(range(10))
    assert spec.execution.episodes_per_initial_state == 1
    assert spec.benchmark.max_steps == 600
    assert spec.benchmark.settle_steps == 0
    assert spec.benchmark.initial_state_source == "benchmark"
    assert spec.policy.smolvla_inference.n_action_steps == 1
    assert spec.policy.smolvla_inference.num_steps == 10
    assert spec.recording.save_frames is False
    assert spec.recording.save_video is False
```

- [ ] **Step 2: Run the test and verify it fails against the old two-task configuration**

Run:

```powershell
& 'D:\Bristol_IOT_with_AI\Capstone Project\.pc-smolvla\Scripts\python.exe' -m pytest tests\test_experiment_configs.py::test_pc_local_smolvla_official_alignment_is_a_one_hundred_trial_protocol -q
```

Expected: assertion failure because the file currently schedules only two task IDs.

- [ ] **Step 3: Replace the YAML content with the canonical protocol**

```yaml
schema_version: 1
name: libero_spatial_pc_local_smolvla_official_alignment
benchmark:
  backend: libero
  suite: libero_spatial
  task_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  initial_state_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  max_steps: 600
  settle_steps: 0
  initial_state_source: benchmark
policy:
  key: smolvla_libero
  checkpoint: catalog:default
  precision: fp16
  quantization: none
  action_control:
    mode: identity
    translation_scale: 1.0
    rotation_scale: 1.0
  smolvla_inference:
    n_action_steps: 1
    num_steps: 10
deployment:
  mode: pc_local
  profile: pc_default
execution:
  episodes_per_initial_state: 1
  warmup_episodes: 0
  seed: 42
  on_episode_failure: continue
viewer:
  enabled: false
recording:
  save_frames: false
  save_video: false
  frame_stride: 10
  save_steps: true
```

- [ ] **Step 4: Write the operating guide**

The guide must state the 100-trial envelope, distinguish `n_action_steps=1` from `chunk_size=50`, and provide these commands:

```powershell
$ProjectRoot = 'D:\Bristol_IOT_with_AI\Capstone Project\.worktrees\libero-yaml-cli-v1\Final_Project\LIBERO_Benchmark_Platform'
$Python = 'D:\Bristol_IOT_with_AI\Capstone Project\.pc-smolvla\Scripts\python.exe'
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\..\vendor\LIBERO\libero;$env:PYTHONPATH"
$env:LIBERO_CONFIG_PATH = 'D:\Bristol_IOT_with_AI\Capstone Project\.worktrees\libero-yaml-cli-v1\.libero'
& $Python -m libero_platform validate configs\experiments\libero_spatial_pc_local_smolvla_official_alignment.yaml
& $Python -m libero_platform run configs\experiments\libero_spatial_pc_local_smolvla_official_alignment.yaml
```

- [ ] **Step 5: Run the configuration and focused integration tests**

Run:

```powershell
& 'D:\Bristol_IOT_with_AI\Capstone Project\.pc-smolvla\Scripts\python.exe' -m pytest tests\test_spec.py tests\test_experiment_configs.py tests\test_smolvla_policy.py tests\test_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Validate the formal YAML through the real CLI**

Run:

```powershell
& 'D:\Bristol_IOT_with_AI\Capstone Project\.pc-smolvla\Scripts\python.exe' -m libero_platform validate configs\experiments\libero_spatial_pc_local_smolvla_official_alignment.yaml
```

Expected: `valid: libero_spatial_pc_local_smolvla_official_alignment`.

