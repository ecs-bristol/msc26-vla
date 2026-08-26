from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from libero_platform.backends import libero_backend


class _Suite:
    def get_num_tasks(self) -> int:
        return 10

    def get_task(self, task_id: int) -> SimpleNamespace:
        return SimpleNamespace(name=f"task-{task_id}", language=f"instruction-{task_id}")


class _Benchmark:
    @staticmethod
    def get_benchmark_dict():
        return {"libero_spatial": _Suite}


def test_official_backend_constructs_exact_lerobot_h1_environment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def env_factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    runtime = SimpleNamespace(
        benchmark=_Benchmark(),
        libero_env=env_factory,
        preprocess_observation=lambda value: value,
        processor_factory=lambda: object(),
    )
    monkeypatch.setattr(libero_backend, "_load_official_lerobot_runtime", lambda: runtime)

    backend = libero_backend.OfficialLeRobotLiberoBackend()
    backend.open_episode("libero_spatial", 0, 0, max_steps=280, seed=1000)

    assert captured == {
        "task_suite": captured["task_suite"],
        "task_id": 0,
        "task_suite_name": "libero_spatial",
        "episode_length": 280,
        "obs_type": "pixels_agent_pos",
        "observation_width": 360,
        "observation_height": 360,
        "init_states": True,
        "episode_index": 0,
        "n_envs": 1,
        "num_steps_wait": 10,
        "control_freq": 20,
        "control_mode": "relative",
        "hard_reset": True,
    }
    backend.close()
    with pytest.raises(RuntimeError, match="backend is closed"):
        backend.open_episode("libero_spatial", 0, 0, max_steps=280, seed=1000)


class _CountingProcessor:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.calls = 0

    def observation(self, value):
        self.calls += 1
        return self.delegate.observation(value)


class _FakeOfficialEnv:
    task_description = "pick up the black bowl"

    def __init__(self, raw_observation: dict[str, object]) -> None:
        self._raw_observation = raw_observation
        self._init_states = np.arange(12, dtype=np.float32).reshape(1, 12)
        self._env = SimpleNamespace(
            sim=SimpleNamespace(
                model=SimpleNamespace(_model=object()),
                data=SimpleNamespace(_data=object()),
            )
        )
        self.reset_seeds: list[int] = []
        self.actions: list[np.ndarray] = []
        self.closed = False

    def reset(self, *, seed: int):
        self.reset_seeds.append(seed)
        return self._raw_observation, {"is_success": False}

    def step(self, action: np.ndarray):
        self.actions.append(action.copy())
        return self._raw_observation, 0.0, False, False, {"is_success": False}

    def close(self) -> None:
        self.closed = True


def _raw_official_observation() -> dict[str, object]:
    agent = np.zeros((360, 360, 3), dtype=np.uint8)
    wrist = np.zeros((360, 360, 3), dtype=np.uint8)
    agent[0, 0] = [1, 2, 3]
    agent[-1, -1] = [4, 5, 6]
    wrist[0, -1] = [7, 8, 9]
    wrist[-1, 0] = [10, 11, 12]
    return {
        "pixels": {"image": agent, "image2": wrist},
        "robot_state": {
            "eef": {
                "pos": np.array([0.1, -0.2, 1.3], dtype=np.float64),
                "quat": np.array([-0.02, 0.001, 0.03, -0.999], dtype=np.float64),
                "mat": np.eye(3, dtype=np.float64),
            },
            "gripper": {
                "qpos": np.array([0.04, -0.04], dtype=np.float64),
                "qvel": np.zeros(2, dtype=np.float64),
            },
            "joints": {
                "pos": np.zeros(7, dtype=np.float64),
                "vel": np.zeros(7, dtype=np.float64),
            },
        },
    }


def test_official_processor_is_applied_once_with_hw_rotation_and_state_parity() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.envs import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep

    raw = _raw_official_observation()
    environment = _FakeOfficialEnv(raw)
    processor = _CountingProcessor(LiberoProcessorStep())
    preprocess_calls = 0

    def counting_preprocess(value):
        nonlocal preprocess_calls
        preprocess_calls += 1
        return preprocess_observation(value)

    episode = libero_backend.OfficialLeRobotLiberoEpisode(
        environment=environment,
        seed=1000,
        initial_state_id=0,
        preprocess_observation=counting_preprocess,
        processor=processor,
    )
    observation = episode.reset()

    np.testing.assert_array_equal(
        observation.images["agentview"], raw["pixels"]["image"][::-1, ::-1]
    )
    np.testing.assert_array_equal(
        observation.images["wrist"], raw["pixels"]["image2"][::-1, ::-1]
    )
    assert observation.images["agentview"].shape == (360, 360, 3)
    expected_batch = preprocess_observation(libero_backend._add_single_environment_batch(raw))
    expected_batch["task"] = [environment.task_description]
    expected = LiberoProcessorStep().observation(expected_batch)
    np.testing.assert_array_equal(
        observation.proprioception,
        expected["observation.state"].detach().cpu().numpy()[0],
    )
    assert observation.proprioception.shape == (8,)
    assert observation.instruction == environment.task_description
    assert preprocess_calls == processor.calls == episode.processor_calls == 1
    assert episode.reset_evidence is not None
    assert episode.reset_evidence.seed == 1000
    assert episode.reset_evidence.settle_steps == 10

    episode.step(np.zeros(7, dtype=np.float32))
    assert preprocess_calls == processor.calls == episode.processor_calls == 2
    assert environment.actions[0].shape == (7,)
    episode.close()
    assert environment.closed is True
