# WSL LIBERO Jetson Remote SmolVLA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the exact WSL-validated `HuggingFaceVLA/smolvla_libero` model on Jetson Orin Nano and collect reproducible remote-inference LIBERO Spatial evidence while WSL owns MuJoCo simulation.

**Architecture:** Keep two result tracks: native WSL `lerobot-eval` for paper-protocol results and the project runner for PC-simulator/Jetson-inference deployment evidence. Jetson serves policy actions on `http://10.42.0.2:8081`; WSL must reject rollout unless health reports the expected checkpoint, immutable Hub revision, fp16, CUDA, and ready state.

**Tech Stack:** Python 3.12, LeRobot 0.6.1, HF-LIBERO 0.1.4, MuJoCo 3.3.2 EGL, PyTorch CUDA, Docker/NVIDIA runtime, YAML, pytest.

## Amendment: Single 280-Step Protocol-Alignment Run

The staged `600`-step smoke, pilot, and formal experiment matrices are superseded. Retain only `libero_spatial_jetson_remote_smolvla_smoke.yaml`, configured as 10 Spatial tasks, one episode per task, and `max_steps: 280`. The preflight/debug configuration remains non-reportable operational tooling. Task 5 and the staged portions of Task 6 are historical and must not be executed.

## Global Constraints

- Model identity is always `HuggingFaceVLA/smolvla_libero`, an explicit Hub revision, and `fp16`.
- No Windows Python runtime is recreated. WSL runs simulator/evaluation/results; Jetson runs inference only.
- No silent PC-local fallback is permitted for remote policy or service failure.
- Record model identity, action validity, service latency, transport latency, RTT, outcomes, videos, logs, and device telemetry.
- Experiment gates are 1 x 10 smoke, 3 x 10 pilot, then 10 x 10 x 3 formal seeds; retain 600 max steps.
- Do not stage/revert/delete unrelated dirty-worktree files.

---

## File Structure

- `src/libero_platform/policies/base.py`, `smolvla_policy.py`: model identity contract.
- `src/libero_platform/deployment/policy_service.py`, `cli.py`: versioned service health and explicit revision CLI.
- `src/libero_platform/policies/remote_http.py`, `validator.py`, `spec.py`, `catalog.py`: strict remote preflight.
- `src/libero_platform/runner.py`, `result_schema.py`, `recorder.py`: distinct service/transport/RTT metrics.
- `scripts/jetson/start_smolvla_libero_service.sh`, `scripts/wsl/run_jetson_remote_preflight.sh`: reproducible operators' entry points.
- `configs/experiments/libero_spatial_jetson_remote_smolvla_{smoke,pilot,formal}.yaml`: fixed stage definitions.
- `docs/runbooks/wsl-jetson-remote-smolvla.md`: handoff and acceptance commands.

### Task 1: Policy identity and service health contract

**Files:**
- Modify: `src/libero_platform/policies/base.py`
- Modify: `src/libero_platform/policies/smolvla_policy.py`
- Modify: `src/libero_platform/deployment/policy_service.py`
- Modify: `src/libero_platform/cli.py`
- Test: `tests/test_policy_service.py`, `tests/test_smolvla_policy.py`, `tests/test_cli.py`

**Interfaces:**
- Produces `PolicyAdapter.identity() -> dict[str, object]` with `model_key`, `checkpoint`, `revision`, `precision`, `device`, `ready`.
- Produces `GET /health -> {"schema_version": 1, "status": "ok", "policy": identity}`.
- Adds `serve-policy --revision REVISION`.

- [ ] **Step 1: Write failing tests**

```python
def test_health_exposes_loaded_policy_identity(client):
    assert client.get("/health").json() == {
        "schema_version": 1, "status": "ok",
        "policy": {
            "model_key": "smolvla_libero",
            "checkpoint": "HuggingFaceVLA/smolvla_libero",
            "revision": "0123456789abcdef",
            "precision": "fp16", "device": "cuda", "ready": True,
        },
    }

def test_smolvla_identity_uses_revision(fake_runtime):
    policy = SmolVLAPolicyAdapter(
        "smolvla_libero", "HuggingFaceVLA/smolvla_libero", "fp16",
        revision="0123456789abcdef", runtime=fake_runtime,
    )
    policy.load()
    assert policy.identity()["revision"] == "0123456789abcdef"
```

- [ ] **Step 2: Run focused tests**

Run in WSL after `source ~/vla/lerobot-libero/bin/activate`: `python -m pytest tests/test_policy_service.py tests/test_smolvla_policy.py tests/test_cli.py -q`

Expected: FAIL because the identity interface, health payload, and CLI argument do not exist.

- [ ] **Step 3: Implement the contract**

```python
# base.py
def identity(self) -> dict[str, object]:
    return {
        "model_key": self.model_key, "checkpoint": None, "revision": None,
        "precision": None, "device": "unavailable", "ready": False,
    }

# smolvla_policy.py
def identity(self) -> dict[str, object]:
    return {
        "model_key": self.model_key, "checkpoint": self._checkpoint,
        "revision": self._revision, "precision": self._precision,
        "device": self._runtime.device_name(), "ready": self._loaded,
    }

# policy_service.py
return {"schema_version": 1, "status": "ok", "policy": self._policy.identity()}
```

Add `revision: str | None = None` to SmolVLA construction, pass it through `_build_service_policy`, and include the same identity in `/metadata`.

- [ ] **Step 4: Verify and commit**

Run in WSL after `source ~/vla/lerobot-libero/bin/activate`: `python -m pytest tests/test_policy_service.py tests/test_smolvla_policy.py tests/test_cli.py -q`

Expected: PASS; `python -m libero_platform serve-policy --help` lists `--revision`.

```powershell
git add src/libero_platform/policies/base.py src/libero_platform/policies/smolvla_policy.py src/libero_platform/deployment/policy_service.py src/libero_platform/cli.py tests/test_policy_service.py tests/test_smolvla_policy.py tests/test_cli.py
git commit -m "feat: expose Jetson policy identity"
```

### Task 2: Strict WSL-to-Jetson preflight

**Files:**
- Modify: `src/libero_platform/policies/remote_http.py`
- Modify: `src/libero_platform/validator.py`
- Modify: `src/libero_platform/spec.py`
- Modify: `src/libero_platform/catalog.py`
- Test: `tests/test_remote_http.py`, `tests/test_validator.py`, `tests/test_cli.py`

**Interfaces:**
- Produces `probe_remote_policy(endpoint: str, timeout_s: float = 2.0) -> dict[str, object]`.
- Adds `PolicySpec.revision: str | None` and propagates it into the resolved spec.
- Rejects absent/unready services and wrong checkpoint/revision/precision.

- [ ] **Step 1: Write failing tests**

```python
def test_validator_rejects_wrong_checkpoint(requests_mock, resolved):
    requests_mock.get("http://10.42.0.2:8081/health", json={
        "schema_version": 1, "status": "ok",
        "policy": {"checkpoint": "lerobot/smolvla_libero", "revision": "old",
                   "precision": "fp16", "ready": True},
    })
    with pytest.raises(ValidationError, match="checkpoint mismatch"):
        validate_config(resolved, check_network=True)

def test_probe_rejects_not_ready_server(requests_mock):
    requests_mock.get("http://jetson:8081/health", json={
        "schema_version": 1, "status": "ok", "policy": {"ready": False},
    })
    with pytest.raises(RemotePolicyUnavailable, match="not ready"):
        probe_remote_policy("http://jetson:8081")
```

- [ ] **Step 2: Run red tests**

Run in WSL after `source ~/vla/lerobot-libero/bin/activate`: `python -m pytest tests/test_remote_http.py tests/test_validator.py tests/test_cli.py -q`

Expected: FAIL because the existing validator only accepts a successful HTTP status.

- [ ] **Step 3: Implement exact comparison**

```python
def probe_remote_policy(endpoint: str, timeout_s: float = 2.0) -> dict[str, object]:
    response = requests.get(f"{endpoint.rstrip('/')}/health", timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    policy = payload.get("policy")
    if payload.get("schema_version") != 1 or payload.get("status") != "ok":
        raise RemotePolicyUnavailable("remote health response is not ready")
    if not isinstance(policy, dict) or policy.get("ready") is not True:
        raise RemotePolicyUnavailable("remote policy is not ready")
    return policy
```

In `_validate_endpoint`, compare each expected field and include expected/observed values in `ValidationError`. Do not call `/predict` or create a local policy.

- [ ] **Step 4: Verify and commit**

Run in WSL after `source ~/vla/lerobot-libero/bin/activate`: `python -m pytest tests/test_remote_http.py tests/test_validator.py tests/test_cli.py -q`

Expected: PASS including matching-server acceptance.

```powershell
git add src/libero_platform/policies/remote_http.py src/libero_platform/validator.py src/libero_platform/spec.py src/libero_platform/catalog.py tests/test_remote_http.py tests/test_validator.py tests/test_cli.py
git commit -m "feat: validate remote Jetson policy identity"
```

### Task 3: Correct remote latency evidence

**Files:**
- Modify: `src/libero_platform/runner.py`
- Modify: `src/libero_platform/result_schema.py`
- Modify: `src/libero_platform/recorder.py`
- Test: `tests/test_runner.py`, `tests/test_result_schema.py`

**Interfaces:**
- Consumes `PolicyResponse.metadata["service_latency_ms"]` and `["remote_round_trip_ms"]`.
- Adds `StepRecord.service_latency_ms: float | None`.
- Writes `transport_latency_ms=max(rtt-service, 0)` and `end_to_end_ms=rtt`.

- [ ] **Step 1: Write failing evidence test**

```python
def test_runner_records_remote_service_transport_and_rtt(tmp_path, remote_dependencies):
    remote_dependencies.policy.predict.return_value = PolicyResponse(
        action=np.zeros(7, dtype=np.float32), inference_ms=12.5,
        model_key="remote_http_policy", device="remote",
        metadata={"service_latency_ms": 12.5, "remote_round_trip_ms": 43.0},
    )
    run_experiment(remote_spec, remote_dependencies)
    step = read_steps(tmp_path)[0]
    assert step["service_latency_ms"] == 12.5
    assert step["transport_latency_ms"] == 30.5
    assert step["end_to_end_ms"] == 43.0
```

- [ ] **Step 2: Run red tests**

Run in WSL after `source ~/vla/lerobot-libero/bin/activate`: `python -m pytest tests/test_runner.py tests/test_result_schema.py -q`

Expected: FAIL because runner currently writes zero transport latency and copies inference time to RTT.

- [ ] **Step 3: Implement schema and calculation**

```python
service_latency = _finite_metadata_float(response.metadata, "service_latency_ms")
round_trip = _finite_metadata_float(response.metadata, "remote_round_trip_ms")
policy_latency = service_latency if service_latency is not None else float(response.inference_ms)
transport_latency = max(round_trip - service_latency, 0.0) if (
    round_trip is not None and service_latency is not None
) else None
end_to_end = round_trip if round_trip is not None else policy_latency
```

Place optional `service_latency_ms` immediately after `policy_latency_ms` in the step CSV; old CSV readers map a missing value to `None`. Apply this to successful and policy-failure records when metadata exists.

- [ ] **Step 4: Verify and commit**

Run in WSL after `source ~/vla/lerobot-libero/bin/activate`: `python -m pytest tests/test_runner.py tests/test_result_schema.py -q`

Expected: PASS with 12.5 ms service, 30.5 ms transport, and 43.0 ms RTT.

```powershell
git add src/libero_platform/runner.py src/libero_platform/result_schema.py src/libero_platform/recorder.py tests/test_runner.py tests/test_result_schema.py
git commit -m "feat: record remote policy latency evidence"
```

### Task 4: Reproducible service and network scripts

**Files:**
- Create: `scripts/jetson/start_smolvla_libero_service.sh`
- Create: `scripts/wsl/run_jetson_remote_preflight.sh`
- Modify: `scripts/jetson/run_container.sh`
- Test: `tests/test_jetson_container_assets.py`, `tests/test_wsl_jetson_scripts.py`

**Interfaces:**
- Consumes `MODEL_REVISION`, optional `CHECKPOINT`, and `JETSON_ENDPOINT`.
- Produces one-time online `bootstrap` and normal cached `offline` modes.

- [ ] **Step 1: Write failing script checks**

```python
def test_jetson_service_script_has_modes(text):
    assert 'MODE="${1:-offline}"' in text
    assert "HF_HUB_OFFLINE=0" in text and "HF_HUB_OFFLINE=1" in text
    assert '--revision "$MODEL_REVISION"' in text

def test_wsl_preflight_uses_network_validation(text):
    assert "JETSON_ENDPOINT" in text
    assert "libero_platform validate" in text
```

- [ ] **Step 2: Run red tests**

Run in WSL after `source ~/vla/lerobot-libero/bin/activate`: `python -m pytest tests/test_jetson_container_assets.py tests/test_wsl_jetson_scripts.py -q`

Expected: FAIL because these scripts do not exist.

- [ ] **Step 3: Implement scripts**

```bash
#!/usr/bin/env bash
# scripts/jetson/start_smolvla_libero_service.sh
set -euo pipefail
MODE="${1:-offline}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
MODEL_REVISION="${MODEL_REVISION:?Set MODEL_REVISION to the WSL cache revision}"
if [[ "$MODE" == bootstrap ]]; then
  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
elif [[ "$MODE" == offline ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
else
  echo "Usage: $0 [bootstrap|offline]" >&2; exit 2
fi
exec "$(dirname "$0")/run_container.sh" python3 -m libero_platform serve-policy \
  --policy smolvla_libero --checkpoint "$CHECKPOINT" --revision "$MODEL_REVISION" \
  --precision fp16 --host 0.0.0.0 --port 8081
```

```bash
#!/usr/bin/env bash
# scripts/wsl/run_jetson_remote_preflight.sh
set -euo pipefail
export JETSON_ENDPOINT="${JETSON_ENDPOINT:-http://10.42.0.2:8081}"
export MODEL_REVISION="${MODEL_REVISION:?Set MODEL_REVISION before preflight}"
python -m libero_platform validate \
  configs/experiments/libero_spatial_jetson_remote_smolvla_smoke.yaml --check-network
```

Pass `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `CHECKPOINT`, and `MODEL_REVISION` into Docker. Preserve the current read-only project mount and never mount PC simulation files to Jetson.

- [ ] **Step 4: Verify and commit**

Run:
```powershell
bash -n scripts/jetson/start_smolvla_libero_service.sh
bash -n scripts/wsl/run_jetson_remote_preflight.sh
source ~/vla/lerobot-libero/bin/activate
python -m pytest tests/test_jetson_container_assets.py tests/test_wsl_jetson_scripts.py -q
```

Expected: all commands exit 0.

```powershell
git add scripts/jetson/run_container.sh scripts/jetson/start_smolvla_libero_service.sh scripts/wsl/run_jetson_remote_preflight.sh tests/test_jetson_container_assets.py tests/test_wsl_jetson_scripts.py
git commit -m "feat: add reproducible Jetson SmolVLA launch"
```

### Task 5: Fixed 10-task smoke, pilot, and formal YAMLs

**Files:**
- Create: `configs/experiments/libero_spatial_jetson_remote_smolvla_smoke.yaml`
- Create: `configs/experiments/libero_spatial_jetson_remote_smolvla_pilot.yaml`
- Create: `configs/experiments/libero_spatial_jetson_remote_smolvla_formal.yaml`
- Modify: `configs/experiments/libero_spatial_jetson_remote_smolvla.yaml`
- Test: `tests/test_experiment_configs.py`, `tests/test_validator.py`

**Interfaces:**
- Consumes profile `jetson_remote_client_direct` and Task 2 validation.
- Produces 10/30/100 episodes per seed, exactly Spatial task IDs 0 through 9.

- [ ] **Step 1: Write failing config tests**

```python
@pytest.mark.parametrize(("name", "episodes"), [
    ("libero_spatial_jetson_remote_smolvla_smoke.yaml", 1),
    ("libero_spatial_jetson_remote_smolvla_pilot.yaml", 3),
    ("libero_spatial_jetson_remote_smolvla_formal.yaml", 10),
])
def test_remote_spatial_configs_are_complete(name, episodes):
    spec = load_experiment_config(CONFIGS / name)
    assert spec.policy.checkpoint == "HuggingFaceVLA/smolvla_libero"
    assert spec.policy.revision == "${MODEL_REVISION}"
    assert spec.policy.precision == "fp16"
    assert spec.benchmark.task_ids == list(range(10))
    assert spec.benchmark.episodes_per_task == episodes
    assert spec.benchmark.max_steps == 600
```

- [ ] **Step 2: Run red tests**

Run in WSL after `source ~/vla/lerobot-libero/bin/activate`: `python -m pytest tests/test_experiment_configs.py tests/test_validator.py -q`

Expected: FAIL because staged YAMLs do not exist.

- [ ] **Step 3: Add explicit YAML policy and deployment**

```yaml
policy:
  key: remote_http_policy
  checkpoint: HuggingFaceVLA/smolvla_libero
  revision: ${MODEL_REVISION}
  precision: fp16
  quantization: none
deployment:
  mode: remote
  device_profile: jetson_remote_client_direct
benchmark:
  suite: libero_spatial
  task_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  initial_state_ids: [0]
  max_steps: 600
recording:
  save_frames: false
  save_video: true
```

Set `episodes_per_task` to 1, 3, and 10. Mark the existing single-task YAML as legacy debugging, not reportable staged evidence.

- [ ] **Step 4: Verify and commit**

Run in WSL after `source ~/vla/lerobot-libero/bin/activate`: `python -m pytest tests/test_experiment_configs.py tests/test_validator.py -q`

Expected: PASS; smoke resolves to 10 episodes and formal to 100 per seed.

```powershell
git add configs/experiments/libero_spatial_jetson_remote_smolvla.yaml configs/experiments/libero_spatial_jetson_remote_smolvla_smoke.yaml configs/experiments/libero_spatial_jetson_remote_smolvla_pilot.yaml configs/experiments/libero_spatial_jetson_remote_smolvla_formal.yaml tests/test_experiment_configs.py tests/test_validator.py
git commit -m "feat: add staged Jetson remote Spatial configs"
```

### Task 6: Runbook, full regression gate, and hardware acceptance

**Files:**
- Create: `docs/runbooks/wsl-jetson-remote-smolvla.md`
- Modify: `README.md`
- Test: `tests/test_policy_service.py`, `tests/test_remote_http.py`, `tests/test_runner.py`

**Interfaces:**
- Consumes all Tasks 1-5.
- Produces an operator guide and fake-server test proving that valid remote inference generates complete evidence without a fallback.

- [ ] **Step 1: Write failing end-to-end test**

```python
def test_matching_remote_service_records_valid_evidence(tmp_path, http_server):
    http_server.health(
        checkpoint="HuggingFaceVLA/smolvla_libero", revision="abc", precision="fp16"
    )
    http_server.predict(action=[0.0] * 7, service_latency_ms=20.0)
    result = run_experiment(remote_smoke_spec(revision="abc"), dependencies_for(http_server))
    assert result.trials[0].action_valid is True
    assert read_steps(tmp_path)[0]["service_latency_ms"] == 20.0
    assert read_steps(tmp_path)[0]["end_to_end_ms"] >= 20.0
```

- [ ] **Step 2: Run full software gate**

Run:
```powershell
source ~/vla/lerobot-libero/bin/activate
python -m pytest tests/test_policy_service.py tests/test_remote_http.py tests/test_validator.py tests/test_runner.py tests/test_result_schema.py tests/test_experiment_configs.py -q
git diff --check
```

Expected: PASS and no whitespace errors.

- [ ] **Step 3: Write exact operator commands in the runbook**

```bash
# WSL: use the immutable revision of the known-good 8/10 baseline.
export HF_HOME="$HOME/vla/hf-cache"
cat "$HF_HOME/hub/models--HuggingFaceVLA--smolvla_libero/refs/main"
export MODEL_REVISION="<REVISION>"
export MUJOCO_GL=egl
```

```powershell
# Windows PowerShell: transfer a tested project snapshot to Jetson.
tar -czf C:\tmp\libero-jetson-remote.tar.gz -C "$ProjectRoot" .
scp C:\tmp\libero-jetson-remote.tar.gz msc26vla@10.42.0.2:~/vla/
```

```bash
# Jetson: bootstrap online once, then later start cached/offline.
rm -rf ~/vla/project/*
tar -xzf ~/vla/libero-jetson-remote.tar.gz -C ~/vla/project
cd ~/vla/project
export MODEL_REVISION="<REVISION>"
./scripts/jetson/start_smolvla_libero_service.sh bootstrap
# Later boots:
./scripts/jetson/start_smolvla_libero_service.sh offline
```

```bash
# WSL: preflight first, then only the 1 x 10 smoke.
source ~/vla/lerobot-libero/bin/activate
export MODEL_REVISION="<REVISION>"
export JETSON_ENDPOINT="http://10.42.0.2:8081"
./scripts/wsl/run_jetson_remote_preflight.sh
python -m libero_platform run configs/experiments/libero_spatial_jetson_remote_smolvla_smoke.yaml
```

Document that native WSL `lerobot-eval` and custom remote harness results must be reported in separate tables.

- [ ] **Step 4: Hardware acceptance sequence**

1. Preserve WSL official native Spatial evidence: current one-episode-per-task smoke is 8/10.
2. Bootstrap Jetson once, retain the service log, restart in offline mode, then check `/health`.
3. From WSL require matching checkpoint/revision/fp16/CUDA/ready before smoke.
4. Archive `metadata.json`, `trials.csv`, `steps.csv`, `failures.csv`, videos, and Jetson logs for 1 x 10.
5. Advance to 3 x 10 only with zero service/transport failures and 100% valid actions; report success and p50/p95 latency.
6. Advance to 10 x 10 x 3 only if pilot is stable; report it as a deployment-system condition, not native `lerobot-eval`.

- [ ] **Step 5: Commit**

```powershell
git add docs/runbooks/wsl-jetson-remote-smolvla.md README.md tests/test_policy_service.py tests/test_remote_http.py tests/test_runner.py
git commit -m "docs: add WSL Jetson remote SmolVLA runbook"
```

## Self-Review

- **Spec coverage:** Tasks 1-2 prevent model drift, Task 3 produces deployment metrics, Task 4 makes startup repeatable, Task 5 fixes sample sizes and task coverage, and Task 6 separates official WSL baseline evidence from Jetson deployment evidence.
- **Placeholder scan:** No TBD/TODO or unspecified implementation step remains; each task contains exact paths, concrete tests, commands, expected outcomes, and commit scope.
- **Type consistency:** `PolicyAdapter.identity`, `PolicySpec.revision`, `probe_remote_policy`, `service_latency_ms`, `remote_round_trip_ms`, and `MODEL_REVISION` use consistent names.
