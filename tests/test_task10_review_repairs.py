from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from libero_platform.backends import libero_backend
from libero_platform.backends.libero_backend import LiberoEpisode
from libero_platform.preflight import ValidationReport
from libero_platform.spec import ResolvedExperimentSpec


def test_cli_uses_resolved_libero_dataset_and_task_resolver(
    tmp_path: Path, catalog_root: Path, monkeypatch
) -> None:
    from libero_platform import cli

    source_path = catalog_root / "experiments" / "smoke_fake.yaml"
    spec = _resolved_libero_spec(source_path, tmp_path / "libero_spatial")
    validation_calls: list[dict[str, object]] = []

    class BackendDouble:
        def __init__(
            self,
            *,
            dataset_directory: Path,
            settle_steps: int,
            initial_state_source: str,
        ) -> None:
            self.dataset_directory = dataset_directory
            assert settle_steps == 0
            assert initial_state_source == "demonstration"

        def list_tasks(self, suite: str) -> list[dict[str, object]]:
            assert suite == "libero_spatial"
            return [{"task_id": 0, "task_name": "resolved_task"}]

    class CompletedOutcome:
        status = "completed"
        result_integrity = "complete"

    def validate_config(*_args, **kwargs) -> ValidationReport:
        validation_calls.append(kwargs)
        return ValidationReport(ok=True, resolved_spec=spec)

    captured: dict[str, object] = {}

    def run_callable(_spec, dependencies) -> CompletedOutcome:
        captured["backend"] = dependencies.backend
        return CompletedOutcome()

    monkeypatch.setattr(cli, "validate_config", validate_config)
    monkeypatch.setattr("libero_platform.backends.libero_backend.LiberoBackend", BackendDouble)

    assert cli.main(
        ["run", str(source_path)],
        config_root=catalog_root,
        output_root=tmp_path / "outputs",
        run_callable=run_callable,
    ) == 0

    backend = captured["backend"]
    assert isinstance(backend, BackendDouble)
    assert backend.dataset_directory == Path(spec.dataset_directory)
    resolver = validation_calls[1]["task_name_resolver"]
    assert callable(resolver)
    assert resolver("libero_spatial", 0) == "resolved_task"


def test_episode_proprioception_is_float32_for_non_identity_orientation() -> None:
    raw_observation = {
        "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
    }

    class Environment:
        sim = SimpleNamespace(
            model=SimpleNamespace(_model=object()), data=SimpleNamespace(_data=object())
        )

        def seed(self, _seed: int) -> None:
            pass

        def reset(self) -> dict[str, np.ndarray]:
            return raw_observation

        def set_init_state(self, _state: np.ndarray) -> dict[str, np.ndarray]:
            return raw_observation

    episode = LiberoEpisode(
        environment=Environment(),
        initial_state=np.zeros(1),
        instruction="task",
        initial_state_source="demonstration",
    )

    observation = episode.reset()

    assert observation.proprioception.dtype == np.float32
    assert np.linalg.norm(observation.proprioception[3:6]) > 0.0


def test_episode_close_retries_after_environment_cleanup_failure() -> None:
    class Environment:
        sim = SimpleNamespace(
            model=SimpleNamespace(_model=object()), data=SimpleNamespace(_data=object())
        )

        def __init__(self) -> None:
            self.close_attempts = 0

        def close(self) -> None:
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise RuntimeError("transient close failure")

    environment = Environment()
    episode = LiberoEpisode(
        environment=environment,
        initial_state=np.zeros(1),
        instruction="task",
        initial_state_source="demonstration",
    )

    try:
        episode.close()
    except RuntimeError as error:
        assert str(error) == "transient close failure"
    else:
        raise AssertionError("first close must propagate the environment failure")
    episode.close()

    assert environment.close_attempts == 2


def _resolved_libero_spec(source_path: Path, dataset_directory: Path) -> ResolvedExperimentSpec:
    return ResolvedExperimentSpec.model_validate(
        {
            "schema_version": 1,
            "name": "libero_smoke",
            "benchmark": {
                "backend": "libero",
                "suite": "libero_spatial",
                "task_ids": [0],
                "initial_state_ids": [0],
                "max_steps": 5,
            },
            "policy": {
                "key": "zero_policy",
                "checkpoint": "none",
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
            "resolved_checkpoint": "none",
            "policy_adapter": "zero",
        }
    )


def test_backend_gives_runner_ownership_of_episode_termination(
    tmp_path: Path, monkeypatch
) -> None:
    import h5py

    with h5py.File(tmp_path / "task_demo.hdf5", "w") as dataset:
        dataset.create_dataset("data/demo_0/states", data=np.zeros((1, 3)))
    task = SimpleNamespace(
        name="task", language="task", problem_folder="libero_spatial", bddl_file="task.bddl"
    )
    constructor_arguments: dict[str, object] = {}

    class Suite:
        def get_num_tasks(self) -> int:
            return 1

        def get_task(self, _task_id: int) -> object:
            return task

    class Environment:
        sim = SimpleNamespace(
            model=SimpleNamespace(_model=object()), data=SimpleNamespace(_data=object())
        )

        def __init__(self, **kwargs: object) -> None:
            constructor_arguments.update(kwargs)

    runtime = SimpleNamespace(
        benchmark=SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": Suite}),
        get_libero_path=lambda _key: str(tmp_path / "bddl_files"),
        offscreen_render_env=Environment,
    )
    monkeypatch.setattr(libero_backend, "_load_libero_runtime", lambda: runtime)

    libero_backend.LiberoBackend(dataset_directory=tmp_path).open_episode(
        "libero_spatial", 0, 0, 17, seed=0
    )

    assert constructor_arguments["horizon"] == 17
    assert constructor_arguments["ignore_done"] is True
