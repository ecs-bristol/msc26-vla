# PC-Jetson SmolVLA Action-Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one reproducible CLI diagnostic that compares PC-local and Jetson-remote SmolVLA actions for one deterministic LIBERO observation.

**Architecture:** A focused `policy_parity` module owns deterministic observation capture, adapter invocation, identity validation, numeric comparison, and JSON/CSV evidence writing. The existing CLI only parses the command and delegates to this module. Existing policy adapters and LIBERO backend remain the single sources of inference and environment behavior.

**Tech Stack:** Python 3.12, NumPy, existing LIBERO backend, existing SmolVLA and HTTP policy adapters, pytest.

## Global Constraints

- Fixed evidence input defaults are `libero_spatial`, task 0, initial state 0, seed 42, FP16, and `http://10.42.0.2:8081`.
- Reportable runs must pass `--revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
- The parity command never calls `episode.step()` and never changes the active 280-step experiment YAML.
- Remote inference has no local fallback.
- Alignment threshold is maximum absolute action delta `<= 1e-4`.

---

### Task 1: Add Pure Parity Comparison and Evidence Tests

**Files:**
- Create: `src/libero_platform/policy_parity.py`
- Create: `tests/test_policy_parity.py`

**Interfaces:**
- Consumes: `PolicyRequest`, `PolicyResponse`, `PolicyAdapter`, NumPy arrays.
- Produces: `compare_actions(local: PolicyResponse, remote: PolicyResponse, threshold: float = 1e-4) -> dict[str, object]` and `write_parity_evidence(output_directory: Path, summary: dict[str, object]) -> tuple[Path, Path]`.

- [ ] **Step 1: Write the failing comparison test**

```python
def test_compare_actions_marks_small_delta_as_aligned() -> None:
    local = PolicyResponse(np.zeros(7, dtype=np.float32), 1.0, "local", "cuda")
    remote = PolicyResponse(np.full(7, 0.00001, dtype=np.float32), 2.0, "remote", "remote")

    result = compare_actions(local, remote)

    assert result["status"] == "aligned"
    assert result["max_abs_delta"] == pytest.approx(0.00001)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policy_parity.py::test_compare_actions_marks_small_delta_as_aligned -v`

Expected: FAIL because `libero_platform.policy_parity` does not exist.

- [ ] **Step 3: Write the failing divergence and evidence test**

```python
def test_write_parity_evidence_records_paired_actions(tmp_path: Path) -> None:
    summary = {"local_action": [0.0] * 7, "remote_action": [0.1] * 7, "status": "diverged"}

    summary_path, csv_path = write_parity_evidence(tmp_path, summary)

    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "diverged"
    assert "local_action_0" in csv_path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Implement the minimal pure module**

```python
def compare_actions(local: PolicyResponse, remote: PolicyResponse, threshold: float = 1e-4) -> dict[str, object]:
    delta = np.asarray(remote.action, dtype=np.float32) - np.asarray(local.action, dtype=np.float32)
    max_abs_delta = float(np.abs(delta).max())
    return {"status": "aligned" if max_abs_delta <= threshold else "diverged", "max_abs_delta": max_abs_delta}
```

Implement `write_parity_evidence` using `json.dumps(..., indent=2, sort_keys=True)` and `csv.DictWriter` with `local_action_0` through `remote_action_6` columns.

- [ ] **Step 5: Run focused tests to verify they pass**

Run: `python -m pytest tests/test_policy_parity.py -v`

Expected: PASS.

### Task 2: Capture One Deterministic Observation and Expose the CLI

**Files:**
- Modify: `src/libero_platform/policy_parity.py`
- Modify: `src/libero_platform/cli.py`
- Modify: `tests/test_policy_parity.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `LiberoBackend.open_episode(...)`, `PolicyAdapter.begin_episode(...)`, `PolicyAdapter.predict(...)`, and `probe_remote_policy(endpoint)`.
- Produces: CLI command `policy-parity` and exit code 0 for a completed report, 3 for unavailable/mismatched dependencies, and 4 for an invalid policy result.

- [ ] **Step 1: Write the failing deterministic observation test**

```python
def test_run_policy_parity_uses_one_reset_observation_for_both_adapters(tmp_path: Path) -> None:
    result = run_policy_parity(spec, local_adapter, remote_adapter, backend, tmp_path)

    assert local_adapter.requests == remote_adapter.requests
    assert backend.episode.step_calls == 0
    assert result.summary["status"] in {"aligned", "diverged"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policy_parity.py::test_run_policy_parity_uses_one_reset_observation_for_both_adapters -v`

Expected: FAIL because `run_policy_parity` is not defined.

- [ ] **Step 3: Implement the minimal runner**

```python
episode = backend.open_episode(spec.suite, spec.task_id, spec.initial_state_id, 1, spec.seed)
observation = episode.reset()
request = PolicyRequest(...)
context = EpisodeContext(...)
local.begin_episode(context)
remote.begin_episode(context)
local_response = local.predict(request)
remote_response = remote.predict(request)
```

Create `PolicyParitySpec` as a frozen dataclass containing suite, task ID, initial state ID, seed, checkpoint, revision, precision, endpoint, and output root. Validate remote identity fields before prediction and close the episode/adapters in `finally` blocks.

- [ ] **Step 4: Write the failing CLI dispatch test**

```python
def test_main_policy_parity_delegates_to_parity_runner(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "run_policy_parity", fake_runner)

    result = cli.main(["policy-parity", "--revision", "abc"], output_root=tmp_path)

    assert result == 0
    assert captured["revision"] == "abc"
```

- [ ] **Step 5: Run CLI test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_main_policy_parity_delegates_to_parity_runner -v`

Expected: FAIL because `policy-parity` is not a CLI command.

- [ ] **Step 6: Add CLI parser and dispatch**

```python
parity = subcommands.add_parser("policy-parity")
parity.add_argument("--suite", default="libero_spatial")
parity.add_argument("--task-id", type=int, default=0)
parity.add_argument("--initial-state-id", type=int, default=0)
parity.add_argument("--seed", type=int, default=42)
parity.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
parity.add_argument("--revision", required=True)
parity.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
parity.add_argument("--endpoint", default="http://10.42.0.2:8081")
```

Dispatch to `run_policy_parity` before the default `run` branch and place evidence below `<output_root>/policy_parity/`.

- [ ] **Step 7: Run focused tests to verify they pass**

Run: `python -m pytest tests/test_policy_parity.py tests/test_cli.py -v`

Expected: PASS.

### Task 3: Verify the Diagnostic in WSL Against the Live Jetson Service

**Files:**
- Modify: `docs/superpowers/specs/2026-08-11-wsl-libero-jetson-remote-smolvla-design.md`

**Interfaces:**
- Consumes: a ready Jetson service at `http://10.42.0.2:8081` with the pinned revision.
- Produces: one WSL evidence directory under `outputs/policy_parity/` and an interpretation statement in the deployment design.

- [ ] **Step 1: Run static tests**

Run: `PYTHONPATH=src python -m pytest tests/test_policy_parity.py tests/test_cli.py -v`

Expected: PASS.

- [ ] **Step 2: Start or verify the Jetson service**

Run on Jetson: `./scripts/jetson/start_smolvla_libero_service.sh offline`

Expected: `policy service listening on http://0.0.0.0:8081`.

- [ ] **Step 3: Run the WSL parity command**

```bash
python -m libero_platform policy-parity \
  --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --endpoint http://10.42.0.2:8081
```

Expected: one `summary.json` and one `actions.csv` are written; terminal prints `aligned` or `diverged` plus MAE and maximum delta.

- [ ] **Step 4: Record the outcome**

Add the observed status, revision, action error metrics, and evidence path to the deployment design. Do not treat either status as a benchmark score.

- [ ] **Step 5: Run regression tests**

Run: `PYTHONPATH=src python -m pytest tests/test_remote_http.py tests/test_policy_service.py tests/test_policy_parity.py tests/test_cli.py -v`

Expected: PASS.
