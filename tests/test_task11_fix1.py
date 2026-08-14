from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
import yaml

from libero_platform import cli
from libero_platform.backends.fake_backend import FakeBackend
from libero_platform.policies.base import EpisodeContext
from libero_platform.policies.demo_replay_policy import DemoReplayPolicyAdapter
from libero_platform.preflight import validate_demo_hdf5
from libero_platform.recorder import RunRecorder
from libero_platform.runner import RunnerDependencies, run_experiment
from libero_platform.spec import ResolvedExperimentSpec
from libero_platform.validator import validate_config
from tests.test_policy_contract import request


def _write_demo(
    path: Path,
    *,
    demo_id: int = 0,
    states: np.ndarray | None = None,
    actions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as dataset:
        demo = dataset.create_group(f"data/demo_{demo_id}")
        demo.create_dataset(
            "states",
            data=(
                np.zeros((1, 3), dtype=np.float32)
                if states is None
                else states
            ),
        )
        demo.create_dataset("actions", data=actions)


def _demo_spec(
    tmp_path: Path, dataset_directory: Path, *, max_steps: int
) -> ResolvedExperimentSpec:
    source_path = tmp_path / "source.yaml"
    source_path.write_text("schema_version: 1\n", encoding="utf-8")
    return ResolvedExperimentSpec.model_validate(
        {
            "schema_version": 1,
            "name": "demo_replay_test",
            "benchmark": {
                "backend": "fake",
                "suite": "libero_spatial",
                "task_ids": [0],
                "initial_state_ids": [0],
                "max_steps": max_steps,
            },
            "policy": {
                "key": "oracle_or_scripted",
                "checkpoint": "catalog:default",
                "precision": "none",
                "quantization": "none",
            },
            "deployment": {"mode": "pc_local", "profile": "pc_default"},
            "execution": {
                "episodes_per_initial_state": 1,
                "warmup_episodes": 0,
                "seed": 42,
                "on_episode_failure": "continue",
            },
            "viewer": {"enabled": False},
            "recording": {
                "save_frames": False,
                "save_video": False,
                "frame_stride": 20,
                "save_steps": True,
            },
            "source_path": str(source_path),
            "dataset_directory": str(dataset_directory),
            "resolved_checkpoint": "libero-demonstrations-v1",
            "policy_adapter": "demo_replay",
        }
    )


def _run_demo(
    tmp_path: Path,
    *,
    backend: FakeBackend,
    actions: np.ndarray,
    max_steps: int,
):
    dataset_directory = tmp_path / "datasets" / "libero_spatial"
    _write_demo(
        dataset_directory / "fake_task_0_demo.hdf5",
        actions=actions,
    )
    spec = _demo_spec(tmp_path, dataset_directory, max_steps=max_steps)
    recorder = RunRecorder(tmp_path / "outputs")
    outcome = run_experiment(
        spec,
        RunnerDependencies(
            backend=backend,
            policy=DemoReplayPolicyAdapter(
                "oracle_or_scripted", dataset_directory
            ),
            recorder=recorder,
            source_path=Path(spec.source_path),
        ),
    )
    return outcome


def _persisted_trial(outcome) -> dict[str, object]:
    line = (outcome.run_dir / "trials.jsonl").read_text(encoding="utf-8")
    return json.loads(line.splitlines()[0])


def test_exhaustion_persists_typed_policy_failure_evidence(tmp_path: Path) -> None:
    outcome = _run_demo(
        tmp_path,
        backend=FakeBackend(success_step=99),
        actions=np.zeros((1, 7), dtype=np.float32),
        max_steps=3,
    )

    trial = outcome.trials[0]
    assert trial.failure_type == "reference_actions_exhausted"
    assert trial.termination_reason == "policy_failure"
    persisted = _persisted_trial(outcome)
    assert persisted["failure_type"] == "reference_actions_exhausted"
    assert persisted["termination_reason"] == "policy_failure"
    events = [
        json.loads(line)
        for line in (outcome.run_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        event["event"] == "policy_failure"
        and event["failure_type"] == "reference_actions_exhausted"
        for event in events
    )


def test_unsuccessful_done_demo_is_reference_replay_failed(tmp_path: Path) -> None:
    outcome = _run_demo(
        tmp_path,
        backend=FakeBackend(success_step=99, fail_episode=0),
        actions=np.zeros((2, 7), dtype=np.float32),
        max_steps=2,
    )

    trial = outcome.trials[0]
    assert trial.termination_reason == "done"
    assert trial.failure_type == "reference_replay_failed"
    assert _persisted_trial(outcome)["failure_type"] == "reference_replay_failed"


def test_unsuccessful_max_steps_demo_is_reference_replay_failed(
    tmp_path: Path,
) -> None:
    class NeverDoneBackend(FakeBackend):
        def open_episode(self, *args, **kwargs):
            episode = super().open_episode(*args, **kwargs)
            original_step = episode.step

            def step(action):
                result = original_step(action)
                return replace(result, done=False, success=False, reward=0.0)

            episode.step = step
            return episode

    outcome = _run_demo(
        tmp_path,
        backend=NeverDoneBackend(success_step=99),
        actions=np.zeros((2, 7), dtype=np.float32),
        max_steps=2,
    )

    trial = outcome.trials[0]
    assert trial.termination_reason == "max_steps"
    assert trial.failure_type == "reference_replay_failed"
    assert _persisted_trial(outcome)["failure_type"] == "reference_replay_failed"


def test_adapter_normalizes_official_float64_actions_to_float32(
    tmp_path: Path,
) -> None:
    actions = np.array([[0.25, 0, 0, 0, 0, 0, -1]], dtype=np.float64)
    path = tmp_path / "pick_the_block_demo.hdf5"
    _write_demo(path, actions=actions)
    adapter = DemoReplayPolicyAdapter("oracle_or_scripted", tmp_path)

    adapter.begin_episode(EpisodeContext("libero_spatial", 0, "pick_the_block", 0, 42))

    response = adapter.predict(request())
    assert response.action.dtype == np.float32
    np.testing.assert_allclose(response.action, actions[0])


def test_preflight_accepts_official_float64_actions(tmp_path: Path) -> None:
    path = tmp_path / "task_demo.hdf5"
    state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    _write_demo(
        path,
        states=state[None, :],
        actions=np.zeros((1, 7), dtype=np.float64),
    )

    validate_demo_hdf5(
        path,
        initial_state_id=0,
        backend_initial_state=state,
    )


def _copy_catalog(catalog_root: Path, target: Path) -> Path:
    target.mkdir()
    for source in catalog_root.glob("*.yaml"):
        shutil.copy2(source, target / source.name)
    return target


def test_cli_main_routes_formal_oracle_through_stubbed_libero_dependencies(
    tmp_path: Path, catalog_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = _copy_catalog(catalog_root, tmp_path / "configs")
    payload = yaml.safe_load(
        (catalog_root / "experiments" / "libero_spatial_oracle.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["benchmark"].update(task_ids=[0], initial_state_ids=[0], max_steps=2)
    yaml_path = tmp_path / "oracle.yaml"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    dataset_directory = tmp_path / "datasets" / "libero_spatial"
    state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    actions = np.array([[0.25, 0, 0, 0, 0, 0, -1]], dtype=np.float32)
    _write_demo(
        dataset_directory / "selected_task_demo.hdf5",
        states=state[None, :],
        actions=actions,
    )

    class BackendDouble:
        def __init__(
            self,
            *,
            dataset_directory: Path,
            settle_steps: int,
            initial_state_source: str,
        ) -> None:
            assert dataset_directory == tmp_path / "datasets" / "libero_spatial"
            assert settle_steps == 0
            assert initial_state_source == "demonstration"

        def list_tasks(self, suite: str) -> list[dict[str, object]]:
            assert suite == "libero_spatial"
            return [{"task_id": 0, "task_name": "selected_task"}]

        def _read_initial_state(
            self, task_name: str, initial_state_id: int
        ) -> np.ndarray:
            assert (task_name, initial_state_id) == ("selected_task", 0)
            return state.copy()

    class CompletedOutcome:
        status = "completed"

    def run_callable(spec, dependencies):
        assert spec.policy_adapter == "demo_replay"
        assert isinstance(dependencies.policy, DemoReplayPolicyAdapter)
        dependencies.policy.begin_episode(
            EpisodeContext("libero_spatial", 0, "selected_task", 0, 42)
        )
        np.testing.assert_allclose(
            dependencies.policy.predict(request()).action, actions[0]
        )
        return CompletedOutcome()

    from libero_platform.backends import libero_backend
    from libero_platform import validator

    monkeypatch.setattr(validator.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(libero_backend, "LiberoBackend", BackendDouble)

    assert cli.main(
        ["run", str(yaml_path)],
        config_root=config_root,
        output_root=tmp_path / "outputs",
        run_callable=run_callable,
    ) == 0


def test_validator_checks_selected_task_and_demo_cross_product(
    tmp_path: Path, catalog_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = _copy_catalog(catalog_root, tmp_path / "configs")
    payload = yaml.safe_load(
        (catalog_root / "experiments" / "libero_spatial_oracle.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["benchmark"].update(task_ids=[0, 1], initial_state_ids=[0, 1])
    yaml_path = tmp_path / "oracle-cross-product.yaml"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    dataset_directory = tmp_path / "datasets" / "libero_spatial"
    expected_states: dict[tuple[int, int], np.ndarray] = {}
    for task_id in (0, 1):
        path = dataset_directory / f"task_{task_id}_demo.hdf5"
        for initial_state_id in (0, 1):
            state = np.array(
                [task_id, initial_state_id, 1.0], dtype=np.float32
            )
            expected_states[(task_id, initial_state_id)] = state
            _write_demo(
                path,
                demo_id=initial_state_id,
                states=state[None, :],
                actions=np.zeros((1, 7), dtype=np.float32),
            )

    task_calls: list[int] = []
    state_calls: list[tuple[int, int]] = []

    def task_name_resolver(_suite: str, task_id: int) -> str:
        task_calls.append(task_id)
        return f"task_{task_id}"

    def initial_state_resolver(
        _suite: str, task_id: int, initial_state_id: int
    ) -> np.ndarray:
        key = (task_id, initial_state_id)
        state_calls.append(key)
        return expected_states[key].copy()

    from libero_platform import validator

    monkeypatch.setattr(validator.importlib.util, "find_spec", lambda _: object())

    report = validate_config(
        yaml_path,
        config_root,
        task_name_resolver=task_name_resolver,
        initial_state_resolver=initial_state_resolver,
    )

    assert report.ok is True
    assert task_calls == [0, 1]
    assert state_calls == [(0, 0), (0, 1), (1, 0), (1, 1)]
