from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "analysis" / "libero_spatial_paired_pilot.py"
SMOLVLA_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SMOLVLM2_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
SUMMARY_FIELDS = {
    "condition",
    "task_id",
    "seed",
    "initial_state_id",
    "environment_seed",
    "inference_seed",
    "success_at_280",
    "success_step",
    "executed_env_steps",
    "wall_time_to_terminal_s",
    "model_invocations",
    "model_inference_time_s",
    "range_violations",
    "range_clips",
    "buffer_discards",
    "mean_actual_horizon",
    "action_trace_sha256",
    "termination_reason",
    "git_sha",
    "resolved_config_path",
}


def _pilot_module():
    spec = importlib.util.spec_from_file_location("paired_pilot_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(tmp_path: Path, revision: str) -> Path:
    path = tmp_path / "hf" / "hub" / "models--example" / "snapshots" / revision
    path.mkdir(parents=True)
    return path


def _command(output_dir: Path, base_snapshot: Path, vlm_snapshot: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--dry-run",
        "--output-dir",
        str(output_dir),
        "--base-snapshot-path",
        str(base_snapshot),
        "--vlm-snapshot-path",
        str(vlm_snapshot),
        *extra,
    ]


def test_dry_run_materializes_six_strictly_paired_conditions(tmp_path: Path) -> None:
    output_dir = tmp_path / "pilot"
    base_snapshot = _snapshot(tmp_path, SMOLVLA_REVISION)
    vlm_snapshot = _snapshot(tmp_path, SMOLVLM2_REVISION)

    subprocess.run(_command(output_dir, base_snapshot, vlm_snapshot), check=True)

    manifest = json.loads((output_dir / "paired_manifest.json").read_text())
    assert manifest["pairing_key"] == ["task_id", "seed", "initial_state_id"]
    assert len(manifest["trials"]) == 50
    assert manifest["trials"][0] == {
        "task_id": 0,
        "seed": 1000,
        "initial_state_id": 0,
        "episode_index": 0,
    }
    assert manifest["trials"][-1] == {
        "task_id": 9,
        "seed": 1004,
        "initial_state_id": 4,
        "episode_index": 4,
    }

    resolved = json.loads((output_dir / "resolved_config.json").read_text())
    assert resolved["suite"] == "libero_spatial"
    assert resolved["episode_cap"] == 280
    assert resolved["batch_size"] == 1
    assert resolved["model"]["local_files_only"] is True
    assert resolved["model"]["num_steps"] == 2
    assert resolved["model"]["chunk_size"] == 50
    assert resolved["model"]["base_revision"] == SMOLVLA_REVISION
    assert resolved["model"]["vlm_revision"] == SMOLVLM2_REVISION
    assert [condition["name"] for condition in resolved["conditions"]] == [
        "Static-H1",
        "Static-H5",
        "Static-H10",
        "Static-H20",
        "Static-H50",
        "Adaptive-H20→H1",
    ]

    episode_files = list((output_dir / "episodes").rglob("*.json"))
    assert len(episode_files) == 300
    adaptive = json.loads(
        (output_dir / "episodes" / "adaptive-h20-to-h1" / "task_00_seed_1000_state_0.json").read_text()
    )
    assert adaptive["condition_config"]["replan_after_safety_violation"] is True
    assert adaptive["termination_reason"] == "not_started_dry_run"
    assert adaptive["resolved_config_path"] == str((output_dir / "resolved_config.json").resolve())
    assert adaptive["environment_seed"] == 1000
    assert isinstance(adaptive["inference_seed"], int)
    assert adaptive["action_trace_sha256"] is None
    for condition in (
        "static-h1",
        "static-h5",
        "static-h10",
        "static-h20",
        "static-h50",
        "adaptive-h20-to-h1",
    ):
        paired = json.loads(
            (output_dir / "episodes" / condition / "task_00_seed_1000_state_0.json").read_text()
        )
        assert paired["inference_seed"] == adaptive["inference_seed"]
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["environment_seed_source"] == "paired_manifest.trials[].seed"
    assert provenance["pairing_seeds"][0] == {
        "task_id": 0,
        "initial_state_id": 0,
        "environment_seed": 1000,
        "inference_seed": adaptive["inference_seed"],
    }

    with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert set(rows[0]) == SUMMARY_FIELDS
    assert len(rows) == 300
    assert {row["condition"] for row in rows} == {
        "Static-H1",
        "Static-H5",
        "Static-H10",
        "Static-H20",
        "Static-H50",
        "Adaptive-H20→H1",
    }


def test_execute_requires_an_existing_paired_manifest_before_any_rollout(tmp_path: Path) -> None:
    base_snapshot = _snapshot(tmp_path, SMOLVLA_REVISION)
    vlm_snapshot = _snapshot(tmp_path, SMOLVLM2_REVISION)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--execute",
            "--output-dir",
            str(tmp_path / "pilot"),
            "--base-snapshot-path",
            str(base_snapshot),
            "--vlm-snapshot-path",
            str(vlm_snapshot),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "paired_manifest.json" in completed.stderr


class _FakeObservation:
    images = {
        "agentview": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    proprioception = np.zeros(8, dtype=np.float32)
    instruction = "fake task"


class _FakeStep:
    def __init__(self, *, success: bool) -> None:
        self.observation = _FakeObservation()
        self.done = success
        self.success = success


class _FakeEpisode:
    def __init__(self, counter: dict[str, int]) -> None:
        self._counter = counter
        self.reset_evidence = None

    def reset(self):
        return _FakeObservation()

    def step(self, action):
        assert np.asarray(action).shape == (7,)
        self._counter["steps"] += 1
        return _FakeStep(success=True)

    def close(self) -> None:
        self._counter["closes"] += 1


class _FakeBackend:
    def __init__(self, counter: dict[str, int]) -> None:
        self._counter = counter

    def open_episode(self, suite, task_id, initial_state_id, max_steps, seed):
        assert suite == "libero_spatial"
        assert task_id == 0
        assert initial_state_id in {0, 1}
        assert max_steps == 280
        assert seed == 1000 + initial_state_id
        self._counter["opens"] += 1
        return _FakeEpisode(self._counter)


class _FakePolicy:
    def __init__(self) -> None:
        self.telemetry: list[dict[str, object]] = []
        self.model_inference_time_s = 0.0

    def reset(self) -> None:
        # Match the real wrapper: reset clears the action buffer but retains
        # auditable episode history, so the executor must slice its telemetry.
        return None

    def select_action(self, observation):
        assert observation.instruction == "fake task"
        self.telemetry.extend(
            [
                {"event": "refill", "planned_horizon": 20},
                {
                    "event": "action_release",
                    "model_invoked": True,
                    "actual_horizon": 20,
                    "range_violation": True,
                    "range_clipped": True,
                    "buffer_discarded": True,
                },
            ]
        )
        self.model_inference_time_s += 0.25
        value = (random.random() + float(np.random.random())) / 4.0
        return np.full(7, value, dtype=np.float32)

    def close(self) -> None:
        return None


def _mock_inference_seed_setter(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))


def test_inference_seed_sets_python_numpy_and_torch_rngs(monkeypatch) -> None:
    module = _pilot_module()
    calls: list[tuple[str, int]] = []

    class _FakeCuda:
        @staticmethod
        def manual_seed_all(seed: int) -> None:
            calls.append(("cuda", seed))

    class _FakeTorch:
        cuda = _FakeCuda()

        @staticmethod
        def manual_seed(seed: int) -> None:
            calls.append(("cpu", seed))

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    module._set_inference_seed(12345)

    assert calls == [("cpu", 12345), ("cuda", 12345)]


def test_executor_reads_manifest_and_resume_does_not_repeat_completed_episodes(tmp_path: Path) -> None:
    module = _pilot_module()
    output_dir = tmp_path / "pilot"
    base_snapshot = _snapshot(tmp_path, SMOLVLA_REVISION)
    vlm_snapshot = _snapshot(tmp_path, SMOLVLM2_REVISION)
    module.materialize_dry_run(
        config_path=PROJECT_ROOT / "configs" / "evaluation" / "libero_spatial_paired_pilot.yaml",
        output_dir=output_dir,
        base_snapshot_path=str(base_snapshot),
        vlm_snapshot_path=str(vlm_snapshot),
    )
    counter = {"opens": 0, "steps": 0, "closes": 0}

    first = module.execute_pilot(
        output_dir=output_dir,
        backend_factory=lambda: _FakeBackend(counter),
        policy_factory=lambda _condition, _config: _FakePolicy(),
        task_ids={0},
        episodes_per_task=1,
        inference_seed_setter=_mock_inference_seed_setter,
    )
    second = module.execute_pilot(
        output_dir=output_dir,
        backend_factory=lambda: _FakeBackend(counter),
        policy_factory=lambda _condition, _config: _FakePolicy(),
        task_ids={0},
        episodes_per_task=1,
        inference_seed_setter=_mock_inference_seed_setter,
    )

    assert first == {"executed_episodes": 6, "skipped_episodes": 0}
    assert second == {"executed_episodes": 0, "skipped_episodes": 6}
    assert counter == {"opens": 6, "steps": 6, "closes": 6}


def test_executor_persists_every_required_episode_metric(tmp_path: Path) -> None:
    module = _pilot_module()
    output_dir = tmp_path / "pilot"
    base_snapshot = _snapshot(tmp_path, SMOLVLA_REVISION)
    vlm_snapshot = _snapshot(tmp_path, SMOLVLM2_REVISION)
    module.materialize_dry_run(
        config_path=PROJECT_ROOT / "configs" / "evaluation" / "libero_spatial_paired_pilot.yaml",
        output_dir=output_dir,
        base_snapshot_path=str(base_snapshot),
        vlm_snapshot_path=str(vlm_snapshot),
    )
    counter = {"opens": 0, "steps": 0, "closes": 0}
    module.execute_pilot(
        output_dir=output_dir,
        backend_factory=lambda: _FakeBackend(counter),
        policy_factory=lambda _condition, _config: _FakePolicy(),
        task_ids={0},
        episodes_per_task=1,
        inference_seed_setter=_mock_inference_seed_setter,
    )

    result = json.loads(
        (output_dir / "episodes" / "static-h1" / "task_00_seed_1000_state_0.json").read_text()
    )
    assert result["status"] == "completed"
    assert {field for field in SUMMARY_FIELDS} <= result.keys()
    assert result["success_at_280"] is True
    assert result["success_step"] == 1
    assert result["executed_env_steps"] == 1
    assert result["model_invocations"] == 1
    assert result["range_violations"] == 1
    assert result["range_clips"] == 1
    assert result["buffer_discards"] == 1
    assert result["mean_actual_horizon"] == 20.0
    assert result["termination_reason"] == "success"
    assert result["environment_seed"] == 1000
    assert isinstance(result["inference_seed"], int)
    assert len(result["action_trace_sha256"]) == 64


def test_second_episode_uses_only_its_telemetry_and_inference_deltas(tmp_path: Path) -> None:
    module = _pilot_module()
    output_dir = tmp_path / "pilot"
    base_snapshot = _snapshot(tmp_path, SMOLVLA_REVISION)
    vlm_snapshot = _snapshot(tmp_path, SMOLVLM2_REVISION)
    module.materialize_dry_run(
        config_path=PROJECT_ROOT / "configs" / "evaluation" / "libero_spatial_paired_pilot.yaml",
        output_dir=output_dir,
        base_snapshot_path=str(base_snapshot),
        vlm_snapshot_path=str(vlm_snapshot),
    )
    counter = {"opens": 0, "steps": 0, "closes": 0}
    shared_policy = _FakePolicy()

    module.execute_pilot(
        output_dir=output_dir,
        backend_factory=lambda: _FakeBackend(counter),
        policy_factory=lambda _condition, _config: shared_policy,
        task_ids={0},
        episodes_per_task=2,
        inference_seed_setter=_mock_inference_seed_setter,
    )

    first = json.loads(
        (output_dir / "episodes" / "static-h20" / "task_00_seed_1000_state_0.json").read_text()
    )
    second = json.loads(
        (output_dir / "episodes" / "static-h20" / "task_00_seed_1001_state_1.json").read_text()
    )
    assert first["model_invocations"] == second["model_invocations"] == 1
    assert first["range_violations"] == second["range_violations"] == 1
    assert first["range_clips"] == second["range_clips"] == 1
    assert first["buffer_discards"] == second["buffer_discards"] == 1
    assert first["mean_actual_horizon"] == second["mean_actual_horizon"] == 20.0
    assert first["model_inference_time_s"] == second["model_inference_time_s"] == 0.25
    with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = [
            row
            for row in csv.DictReader(handle)
            if row["condition"] == "Static-H20" and row["task_id"] == "0"
        ]
    assert [
        (row["initial_state_id"], row["model_invocations"], row["model_inference_time_s"])
        for row in summary[:2]
    ] == [("0", "1", "0.25"), ("1", "1", "0.25")]


def test_same_pairing_key_repeats_deterministic_action_trace_and_metrics(tmp_path: Path) -> None:
    module = _pilot_module()
    output_dir = tmp_path / "pilot"
    base_snapshot = _snapshot(tmp_path, SMOLVLA_REVISION)
    vlm_snapshot = _snapshot(tmp_path, SMOLVLM2_REVISION)
    module.materialize_dry_run(
        config_path=PROJECT_ROOT / "configs" / "evaluation" / "libero_spatial_paired_pilot.yaml",
        output_dir=output_dir,
        base_snapshot_path=str(base_snapshot),
        vlm_snapshot_path=str(vlm_snapshot),
    )
    config, trials = module._read_execution_inputs(output_dir)
    counter = {"opens": 0, "steps": 0, "closes": 0}
    policy = _FakePolicy()
    condition = config["conditions"][3]
    trial = trials[0]

    first = module._run_episode(
        output_dir=output_dir,
        config=config,
        condition=condition,
        trial=trial,
        backend=_FakeBackend(counter),
        policy=policy,
        inference_seed_setter=_mock_inference_seed_setter,
    )
    second = module._run_episode(
        output_dir=output_dir,
        config=config,
        condition=condition,
        trial=trial,
        backend=_FakeBackend(counter),
        policy=policy,
        inference_seed_setter=_mock_inference_seed_setter,
    )

    fields = (
        "inference_seed",
        "action_trace_sha256",
        "success_at_280",
        "success_step",
        "executed_env_steps",
        "model_invocations",
        "range_violations",
        "range_clips",
        "buffer_discards",
        "mean_actual_horizon",
        "model_inference_time_s",
    )
    assert {field: first[field] for field in fields} == {
        field: second[field] for field in fields
    }
    assert first["environment_seed"] == second["environment_seed"] == trial["seed"]
