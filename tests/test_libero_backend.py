from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from libero_platform.backends.base import Observation


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_supported_platform_with_missing_libero_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    from libero_platform.backends import libero_backend

    monkeypatch.setattr(libero_backend.sys, "platform", platform)
    monkeypatch.setattr(libero_backend.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="LIBERO dependency"):
        libero_backend._load_libero_runtime()


def test_unsupported_platform_is_rejected_without_importing_libero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from libero_platform.backends import libero_backend

    monkeypatch.setattr(libero_backend.sys, "platform", "darwin")
    monkeypatch.setattr(libero_backend.importlib.util, "find_spec", lambda name: pytest.fail(name))
    with pytest.raises(RuntimeError, match="Linux/WSL or native Windows"):
        libero_backend._load_libero_runtime()


def test_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    from libero_platform.backends.libero_backend import LiberoBackend

    with pytest.raises(RuntimeError, match="LIBERO dependency"):
        LiberoBackend()


def test_open_episode_restores_demo_state_and_normalizes_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from libero_platform.backends import libero_backend

    state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with h5py.File(tmp_path / "task_zero_demo.hdf5", "w") as dataset:
        dataset.create_dataset("data/demo_0/states", data=np.array([state]))

    task = SimpleNamespace(
        name="task_zero",
        language="put the object away",
        problem_folder="libero_spatial",
        bddl_file="task_zero.bddl",
    )
    calls: list[object] = []

    class FakeSuite:
        def get_num_tasks(self) -> int:
            return 1

        def get_task(self, task_id: int) -> object:
            assert task_id == 0
            return task

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)
            self.sim = SimpleNamespace(
                model=SimpleNamespace(_model="model"),
                data=SimpleNamespace(_data="data"),
            )
            self.seed_value: int | None = None
            self.initial_state: np.ndarray | None = None

        def seed(self, seed: int) -> None:
            self.seed_value = seed

        def reset(self) -> dict[str, np.ndarray]:
            assert self.seed_value == 0
            return _raw_observation()

        def set_init_state(self, initial_state: np.ndarray) -> dict[str, np.ndarray]:
            self.initial_state = initial_state
            return _raw_observation()

        def step(
            self, action: np.ndarray
        ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
            assert action.shape == (7,)
            return _raw_observation(), 0.25, False, {"source": "fake"}

        def check_success(self) -> bool:
            return True

        def close(self) -> None:
            calls.append("closed")

    runtime = SimpleNamespace(
        benchmark=SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": FakeSuite}),
        get_libero_path=lambda key: str(tmp_path / key),
        offscreen_render_env=FakeEnvironment,
    )
    monkeypatch.setattr(libero_backend, "_load_libero_runtime", lambda: runtime)

    backend = libero_backend.LiberoBackend(dataset_directory=tmp_path, camera_size=4)
    episode = backend.open_episode("libero_spatial", 0, 0, 5, seed=0)
    observation = episode.reset()
    result = episode.step(np.zeros(7, dtype=np.float32))
    episode.close()

    assert calls[0] == {
        "bddl_file_name": str(tmp_path / "bddl_files" / "libero_spatial" / "task_zero.bddl"),
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_heights": 4,
        "camera_widths": 4,
        "horizon": 5,
        "ignore_done": True,
    }
    assert isinstance(observation, Observation)
    assert np.array_equal(episode._env.initial_state, state)
    assert np.array_equal(observation.images["agentview"], _raw_observation()["agentview_image"][::-1])
    assert np.array_equal(observation.images["wrist"], _raw_observation()["robot0_eye_in_hand_image"][::-1])
    assert np.allclose(observation.proprioception, [1, 2, 3, 0, 0, 0, 0.1, 0.2])
    assert observation.instruction == "put the object away"
    assert (episode.model, episode.data) == ("model", "data")
    assert (result.reward, result.done, result.success, result.info) == (
        0.25,
        True,
        True,
        {"source": "fake"},
    )
    assert calls[-1] == "closed"


def test_open_episode_uses_requested_reset_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from libero_platform.backends import libero_backend

    state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with h5py.File(tmp_path / "task_zero_demo.hdf5", "w") as dataset:
        dataset.create_dataset("data/demo_0/states", data=np.array([state]))

    task = SimpleNamespace(
        name="task_zero",
        language="put the object away",
        problem_folder="libero_spatial",
        bddl_file="task_zero.bddl",
    )

    class FakeSuite:
        def get_num_tasks(self) -> int:
            return 1

        def get_task(self, task_id: int) -> object:
            assert task_id == 0
            return task

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.sim = SimpleNamespace(
                model=SimpleNamespace(_model="model"),
                data=SimpleNamespace(_data="data"),
            )
            self.seed_value: int | None = None

        def seed(self, seed: int) -> None:
            self.seed_value = seed

        def reset(self) -> dict[str, np.ndarray]:
            return _raw_observation()

        def set_init_state(self, initial_state: np.ndarray) -> dict[str, np.ndarray]:
            np.testing.assert_array_equal(initial_state, state)
            return _raw_observation()

        def step(
            self, action: np.ndarray
        ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
            del action
            return _raw_observation(), 0.0, False, {}

        def check_success(self) -> bool:
            return False

        def close(self) -> None:
            pass

    runtime = SimpleNamespace(
        benchmark=SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": FakeSuite}),
        get_libero_path=lambda key: str(tmp_path / key),
        offscreen_render_env=FakeEnvironment,
    )
    monkeypatch.setattr(libero_backend, "_load_libero_runtime", lambda: runtime)

    episode = libero_backend.LiberoBackend(dataset_directory=tmp_path).open_episode(
        "libero_spatial", 0, 0, 5, seed=47
    )

    episode.reset()

    assert episode._env.seed_value == 47


@pytest.mark.parametrize("seed", [-1, True])
def test_open_episode_rejects_invalid_reset_seed(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    from libero_platform.backends import libero_backend

    monkeypatch.setattr(
        libero_backend, "_load_libero_runtime", lambda: SimpleNamespace()
    )
    backend = libero_backend.LiberoBackend()

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        backend.open_episode("libero_spatial", 0, 0, 5, seed=seed)


def test_open_episode_settles_after_restoring_the_initial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from libero_platform.backends import libero_backend

    state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with h5py.File(tmp_path / "task_zero_demo.hdf5", "w") as dataset:
        dataset.create_dataset("data/demo_0/states", data=np.array([state]))

    task = SimpleNamespace(
        name="task_zero",
        language="put the object away",
        problem_folder="libero_spatial",
        bddl_file="task_zero.bddl",
    )

    class FakeSuite:
        def get_num_tasks(self) -> int:
            return 1

        def get_task(self, task_id: int) -> object:
            assert task_id == 0
            return task

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.sim = SimpleNamespace(
                model=SimpleNamespace(_model="model"),
                data=SimpleNamespace(_data="data"),
            )
            self.step_actions: list[np.ndarray] = []

        def seed(self, seed: int) -> None:
            assert seed == 0

        def reset(self) -> dict[str, np.ndarray]:
            return _raw_observation()

        def set_init_state(self, initial_state: np.ndarray) -> dict[str, np.ndarray]:
            np.testing.assert_array_equal(initial_state, state)
            return _raw_observation()

        def step(
            self, action: np.ndarray
        ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
            self.step_actions.append(np.asarray(action))
            return _raw_observation(), 0.0, False, {}

        def check_success(self) -> bool:
            return False

        def close(self) -> None:
            pass

    runtime = SimpleNamespace(
        benchmark=SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": FakeSuite}),
        get_libero_path=lambda key: str(tmp_path / key),
        offscreen_render_env=FakeEnvironment,
    )
    monkeypatch.setattr(libero_backend, "_load_libero_runtime", lambda: runtime)

    episode = libero_backend.LiberoBackend(
        dataset_directory=tmp_path, settle_steps=2
    ).open_episode("libero_spatial", 0, 0, 5, seed=0)

    episode.reset()

    assert len(episode._env.step_actions) == 2
    for action in episode._env.step_actions:
        np.testing.assert_array_equal(
            action, np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        )


def test_open_episode_records_reset_evidence_after_settle_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from libero_platform.backends import libero_backend

    state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with h5py.File(tmp_path / "task_zero_demo.hdf5", "w") as dataset:
        dataset.create_dataset("data/demo_0/states", data=np.array([state]))

    task = SimpleNamespace(
        name="task_zero",
        language="put the object away",
        problem_folder="libero_spatial",
        bddl_file="task_zero.bddl",
    )

    class FakeSuite:
        def get_num_tasks(self) -> int:
            return 1

        def get_task(self, task_id: int) -> object:
            assert task_id == 0
            return task

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.sim = SimpleNamespace(
                model=SimpleNamespace(_model="model"),
                data=SimpleNamespace(_data="data"),
            )
            self._step = 0

        def seed(self, seed: int) -> None:
            assert seed == 42

        def reset(self) -> dict[str, np.ndarray]:
            self._step = 0
            return _raw_observation()

        def set_init_state(self, initial_state: np.ndarray) -> dict[str, np.ndarray]:
            np.testing.assert_array_equal(initial_state, state)
            return _raw_observation()

        def step(
            self, action: np.ndarray
        ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
            del action
            self._step += 1
            observation = _raw_observation()
            observation["robot0_eef_pos"] = np.array(
                [10.0 + self._step, 20.0 + self._step, 30.0 + self._step],
                dtype=np.float32,
            )
            observation["robot0_eef_quat"] = np.array(
                [0.0, 0.0, 0.1 * self._step, 1.0],
                dtype=np.float32,
            )
            return observation, 0.0, False, {}

        def check_success(self) -> bool:
            return False

        def close(self) -> None:
            pass

    runtime = SimpleNamespace(
        benchmark=SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": FakeSuite}),
        get_libero_path=lambda key: str(tmp_path / key),
        offscreen_render_env=FakeEnvironment,
    )
    monkeypatch.setattr(libero_backend, "_load_libero_runtime", lambda: runtime)

    episode = libero_backend.LiberoBackend(
        dataset_directory=tmp_path,
        settle_steps=2,
        initial_state_source="demonstration",
    ).open_episode("libero_spatial", 0, 0, 5, seed=42)

    episode.reset()

    expected = hashlib.sha256()
    expected.update(np.ascontiguousarray(state).tobytes())
    expected.update(
        np.ascontiguousarray(np.array([12.0, 22.0, 32.0], dtype=np.float32)).tobytes()
    )
    expected.update(
        np.ascontiguousarray(np.array([0.0, 0.0, 0.2, 1.0], dtype=np.float32)).tobytes()
    )
    assert episode.reset_evidence.seed == 42
    assert episode.reset_evidence.initial_state_source == "demonstration"
    assert episode.reset_evidence.settle_steps == 2
    assert episode.reset_evidence.fingerprint == expected.hexdigest()[:16]


def test_open_episode_can_restore_official_benchmark_initial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from libero_platform.backends import libero_backend

    import torch

    official_state = np.array([7.0, 8.0, 9.0], dtype=np.float64)
    init_path = tmp_path / "init_states" / "libero_spatial"
    init_path.mkdir(parents=True)
    torch.save(np.array([official_state]), init_path / "task_zero.pruned_init")

    task = SimpleNamespace(
        name="task_zero",
        language="put the object away",
        problem_folder="libero_spatial",
        bddl_file="task_zero.bddl",
        init_states_file="task_zero.pruned_init",
    )

    class FakeSuite:
        def get_num_tasks(self) -> int:
            return 1

        def get_task(self, task_id: int) -> object:
            assert task_id == 0
            return task

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.sim = SimpleNamespace(
                model=SimpleNamespace(_model="model"),
                data=SimpleNamespace(_data="data"),
            )
            self.initial_state: np.ndarray | None = None

        def seed(self, seed: int) -> None:
            assert seed == 0

        def reset(self) -> dict[str, np.ndarray]:
            return _raw_observation()

        def set_init_state(self, initial_state: np.ndarray) -> dict[str, np.ndarray]:
            self.initial_state = initial_state
            return _raw_observation()

        def close(self) -> None:
            pass

    runtime = SimpleNamespace(
        benchmark=SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": FakeSuite}),
        get_libero_path=lambda key: str(tmp_path / key),
        offscreen_render_env=FakeEnvironment,
    )
    monkeypatch.setattr(libero_backend, "_load_libero_runtime", lambda: runtime)

    episode = libero_backend.LiberoBackend(
        dataset_directory=tmp_path / "missing-demo-data",
        initial_state_source="benchmark",
    ).open_episode("libero_spatial", 0, 0, 5, seed=0)

    episode.reset()

    assert episode._env.initial_state is not None
    np.testing.assert_array_equal(episode._env.initial_state, official_state)


@pytest.fixture
def libero_dataset_directory() -> Path:
    root = os.environ.get("LIBERO_DATASET_ROOT")
    if not root:
        pytest.skip("LIBERO_DATASET_ROOT must point to the LIBERO datasets directory")

    directory = Path(root) / "libero_spatial"
    if not directory.is_dir():
        pytest.skip("LIBERO dataset suite directory is missing: libero_spatial")

    from libero.libero import benchmark

    task_name = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0).name
    if not (directory / f"{task_name}_demo.hdf5").is_file():
        pytest.skip(f"LIBERO dataset is missing required task HDF5: {task_name}_demo.hdf5")
    return directory


@pytest.mark.libero
def test_real_libero_reset_and_one_step(libero_dataset_directory: Path) -> None:
    from libero_platform.backends.libero_backend import LiberoBackend

    backend = LiberoBackend(dataset_directory=libero_dataset_directory, camera_size=128)
    episode = backend.open_episode("libero_spatial", 0, 0, 5, seed=0)
    try:
        observation = episode.reset()
        assert observation.images["agentview"].shape == (128, 128, 3)
        result = episode.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
        assert isinstance(result.reward, float)
    finally:
        episode.close()


def _raw_observation() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
        "robot0_eye_in_hand_image": np.arange(48, 96, dtype=np.uint8).reshape(4, 4, 3),
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
    }
