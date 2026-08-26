from __future__ import annotations

import importlib.util
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from .base import Observation, ResetEvidence, StepResult


_LIBERO_DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def _load_libero_runtime() -> SimpleNamespace:
    if sys.platform not in {"linux", "win32"}:
        raise RuntimeError("LIBERO backend requires Linux/WSL or native Windows")
    if importlib.util.find_spec("libero") is None:
        raise RuntimeError("LIBERO backend requires the LIBERO dependency installed")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return SimpleNamespace(
        benchmark=benchmark,
        get_libero_path=get_libero_path,
        offscreen_render_env=OffScreenRenderEnv,
    )


def _load_official_lerobot_runtime() -> SimpleNamespace:
    """Load the LeRobot 0.6.1 LIBERO environment and processor lazily."""

    if importlib.util.find_spec("lerobot") is None:
        raise RuntimeError("official LIBERO backend requires LeRobot installed")
    if importlib.util.find_spec("libero") is None:
        raise RuntimeError("official LIBERO backend requires LIBERO installed")

    from lerobot.envs import preprocess_observation
    from lerobot.envs.libero import LiberoEnv
    from lerobot.processor.env_processor import LiberoProcessorStep
    from libero.libero import benchmark

    return SimpleNamespace(
        benchmark=benchmark,
        libero_env=LiberoEnv,
        preprocess_observation=preprocess_observation,
        processor_factory=LiberoProcessorStep,
    )


class LiberoBackend:
    """Open deterministic LIBERO episodes from demonstration or benchmark states."""

    def __init__(
        self,
        *,
        dataset_directory: Path = Path("."),
        camera_size: int = 256,
        settle_steps: int = 0,
        initial_state_source: str = "demonstration",
    ) -> None:
        if not isinstance(camera_size, int) or isinstance(camera_size, bool) or camera_size < 1:
            raise ValueError("camera_size must be a positive integer")
        if not isinstance(settle_steps, int) or isinstance(settle_steps, bool) or settle_steps < 0:
            raise ValueError("settle_steps must be a non-negative integer")
        if initial_state_source not in {"demonstration", "benchmark"}:
            raise ValueError("initial_state_source must be demonstration or benchmark")
        self._dataset_directory = Path(dataset_directory)
        self._camera_size = camera_size
        self._settle_steps = settle_steps
        self._initial_state_source = initial_state_source
        self._runtime = _load_libero_runtime()

    def list_tasks(self, suite: str) -> list[dict[str, object]]:
        benchmark_suite = self._resolve_suite(suite)
        return [
            {"task_id": task_id, "task_name": benchmark_suite.get_task(task_id).name}
            for task_id in range(benchmark_suite.get_num_tasks())
        ]

    def open_episode(
        self, suite: str, task_id: int, initial_state_id: int, max_steps: int, seed: int
    ) -> LiberoEpisode:
        _validate_index("task_id", task_id)
        _validate_index("initial_state_id", initial_state_id)
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        benchmark_suite = self._resolve_suite(suite)
        if task_id >= benchmark_suite.get_num_tasks():
            raise ValueError(f"task_id {task_id} is outside suite {suite}")
        task = benchmark_suite.get_task(task_id)
        initial_state = self._read_initial_state(task, initial_state_id)
        bddl_file = (
            Path(self._runtime.get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        environment = self._runtime.offscreen_render_env(
            bddl_file_name=str(bddl_file),
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=self._camera_size,
            camera_widths=self._camera_size,
            horizon=max_steps,
            ignore_done=True,
        )
        return LiberoEpisode(
            environment=environment,
            initial_state=initial_state,
            instruction=task.language,
            initial_state_source=self._initial_state_source,
            settle_steps=self._settle_steps,
            seed=seed,
        )

    def _resolve_suite(self, suite: str) -> Any:
        try:
            suite_type = self._runtime.benchmark.get_benchmark_dict()[suite]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unknown LIBERO suite: {suite}") from exc
        return suite_type()

    def resolve_initial_state(
        self, suite: str, task_id: int, initial_state_id: int
    ) -> np.ndarray:
        _validate_index("task_id", task_id)
        _validate_index("initial_state_id", initial_state_id)
        benchmark_suite = self._resolve_suite(suite)
        if task_id >= benchmark_suite.get_num_tasks():
            raise ValueError(f"task_id {task_id} is outside suite {suite}")
        return self._read_initial_state(benchmark_suite.get_task(task_id), initial_state_id)

    def _read_initial_state(self, task: Any, initial_state_id: int) -> np.ndarray:
        if self._initial_state_source == "benchmark":
            return self._read_benchmark_initial_state(task, initial_state_id)

        task_name = task if isinstance(task, str) else task.name
        dataset_path = self._dataset_directory / f"{task_name}_demo.hdf5"
        if not dataset_path.is_file():
            raise RuntimeError(f"LIBERO demonstration file is missing: {dataset_path.name}")
        state_path = f"data/demo_{initial_state_id}/states"
        try:
            with h5py.File(dataset_path, "r") as dataset:
                states = dataset[state_path]
                if len(states) == 0:
                    raise ValueError("states dataset is empty")
                return np.asarray(states[0]).copy()
        except KeyError as exc:
            raise ValueError(
                f"initial_state_id {initial_state_id} is unavailable for task {task_name}"
            ) from exc

    def _read_benchmark_initial_state(self, task: Any, initial_state_id: int) -> np.ndarray:
        import torch

        init_state_path = (
            Path(self._runtime.get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file
        )
        if not init_state_path.is_file():
            raise RuntimeError(
                f"LIBERO benchmark initial-state file is missing: {init_state_path.name}"
            )
        # LIBERO's packaged .pruned_init files are trusted benchmark assets.
        states = torch.load(init_state_path, weights_only=False)
        try:
            return np.asarray(states[initial_state_id]).copy()
        except IndexError as exc:
            raise ValueError(
                f"initial_state_id {initial_state_id} is unavailable for task {task.name}"
            ) from exc


class OfficialLeRobotLiberoBackend:
    """Paired-pilot backend that delegates environment semantics to LeRobot 0.6.1.

    Unlike :class:`LiberoBackend`, this path intentionally does not reproduce
    LeRobot's observation transformations.  It instantiates the official
    ``LiberoEnv`` and calls ``preprocess_observation`` and
    ``LiberoProcessorStep`` exactly once per observation.
    """

    def __init__(self) -> None:
        self._runtime = _load_official_lerobot_runtime()
        self._closed = False

    def _resolve_suite(self, suite: str) -> Any:
        try:
            suite_type = self._runtime.benchmark.get_benchmark_dict()[suite]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unknown LIBERO suite: {suite}") from exc
        return suite_type()

    def list_tasks(self, suite: str) -> list[dict[str, object]]:
        benchmark_suite = self._resolve_suite(suite)
        return [
            {"task_id": task_id, "task_name": benchmark_suite.get_task(task_id).name}
            for task_id in range(benchmark_suite.get_num_tasks())
        ]

    def open_episode(
        self, suite: str, task_id: int, initial_state_id: int, max_steps: int, seed: int
    ) -> OfficialLeRobotLiberoEpisode:
        if self._closed:
            raise RuntimeError("official LIBERO backend is closed")
        _validate_index("task_id", task_id)
        _validate_index("initial_state_id", initial_state_id)
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        benchmark_suite = self._resolve_suite(suite)
        if task_id >= benchmark_suite.get_num_tasks():
            raise ValueError(f"task_id {task_id} is outside suite {suite}")
        environment = self._runtime.libero_env(
            task_suite=benchmark_suite,
            task_id=task_id,
            task_suite_name=suite,
            episode_length=max_steps,
            obs_type="pixels_agent_pos",
            observation_width=360,
            observation_height=360,
            init_states=True,
            episode_index=initial_state_id,
            n_envs=1,
            num_steps_wait=10,
            control_freq=20,
            control_mode="relative",
            hard_reset=True,
        )
        return OfficialLeRobotLiberoEpisode(
            environment=environment,
            seed=seed,
            initial_state_id=initial_state_id,
            preprocess_observation=self._runtime.preprocess_observation,
            processor=self._runtime.processor_factory(),
        )

    def close(self) -> None:
        """Release runtime references before interpreter teardown."""

        if self._closed:
            return
        self._closed = True
        self._runtime = None
        gc.collect()


class OfficialLeRobotLiberoEpisode:
    """Adapt one official LeRobot LIBERO environment to the platform protocol."""

    def __init__(
        self,
        *,
        environment: Any,
        seed: int,
        initial_state_id: int,
        preprocess_observation: Any,
        processor: Any,
    ) -> None:
        self._env = environment
        self._seed = seed
        self._initial_state_id = initial_state_id
        self._preprocess_observation = preprocess_observation
        self._processor = processor
        self._closed = False
        self._last_observation: Observation | None = None
        self.processor_calls = 0
        self.reset_evidence: ResetEvidence | None = None
        self.model: object | None = None
        self.data: object | None = None

    def reset(self) -> Observation:
        self._ensure_open()
        initial_state = np.asarray(self._env._init_states[self._initial_state_id]).copy()
        raw_observation, _ = self._env.reset(seed=self._seed)
        inner = self._env._env
        if inner is not None:
            self.model = inner.sim.model._model
            self.data = inner.sim.data._data
        robot_state = raw_observation["robot_state"]
        self.reset_evidence = ResetEvidence.from_components(
            seed=self._seed,
            initial_state_source="benchmark",
            settle_steps=10,
            initial_state=initial_state,
            robot0_eef_pos=robot_state["eef"]["pos"],
            robot0_eef_quat=robot_state["eef"]["quat"],
        )
        self._last_observation = self._observation(raw_observation)
        return self._last_observation

    def step(self, action: np.ndarray) -> StepResult:
        self._ensure_open()
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (7,):
            raise ValueError(f"LIBERO action must have shape (7,), got {value.shape}")
        raw_observation, reward, terminated, truncated, info = self._env.step(value)
        self._last_observation = self._observation(raw_observation)
        return StepResult(
            observation=self._last_observation,
            reward=float(reward),
            done=bool(terminated) or bool(truncated),
            success=bool(info.get("is_success", False)),
            info=dict(info),
        )

    def render_frame(self) -> np.ndarray | None:
        if self._last_observation is None:
            return None
        return self._last_observation.images["agentview"].copy()

    def close(self) -> None:
        if self._closed:
            return
        environment = self._env
        try:
            environment.close()
        finally:
            self.model = None
            self.data = None
            self._last_observation = None
            self._env = None
            self._processor = None
            self._preprocess_observation = None
            self._closed = True
            del environment
            gc.collect()

    def _observation(self, raw_observation: dict[str, Any]) -> Observation:
        processed = self._preprocess_observation(
            _add_single_environment_batch(raw_observation)
        )
        processed["task"] = [self._env.task_description]
        processed = self._processor.observation(processed)
        self.processor_calls += 1
        return Observation(
            images={
                "agentview": _policy_image_to_hwc_uint8(processed["observation.images.image"]),
                "wrist": _policy_image_to_hwc_uint8(processed["observation.images.image2"]),
            },
            proprioception=np.asarray(
                processed["observation.state"].detach().cpu().numpy()[0], dtype=np.float32
            ),
            instruction=processed["task"][0],
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LIBERO episode is closed")


def _policy_image_to_hwc_uint8(value: Any) -> np.ndarray:
    array = np.asarray(value.detach().cpu().numpy())
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 3:
        raise ValueError(
            f"official processed image must have shape (1, 3, H, W), got {array.shape}"
        )
    return np.rint(array[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)


def _add_single_environment_batch(value: Any) -> Any:
    """Match the leading environment dimension supplied by Gym vector envs."""

    if isinstance(value, dict):
        return {key: _add_single_environment_batch(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None, ...]
    return value


@dataclass
class LiberoEpisode:
    _env: Any
    _initial_state: np.ndarray
    _instruction: str
    _initial_state_source: str
    _settle_steps: int
    _seed: int

    def __init__(
        self,
        *,
        environment: Any,
        initial_state: np.ndarray,
        instruction: str,
        initial_state_source: str,
        settle_steps: int = 0,
        seed: int = 0,
    ) -> None:
        self._env = environment
        self._initial_state = initial_state
        self._instruction = instruction
        self._initial_state_source = initial_state_source
        self._settle_steps = settle_steps
        self._seed = seed
        self._closed = False
        self._last_raw_observation: dict[str, np.ndarray] | None = None
        self.reset_evidence: ResetEvidence | None = None
        self.model = self._env.sim.model._model
        self.data = self._env.sim.data._data

    def reset(self) -> Observation:
        self._ensure_open()
        self._env.seed(self._seed)
        self._env.reset()
        self._last_raw_observation = self._env.set_init_state(self._initial_state)
        for _ in range(self._settle_steps):
            self._last_raw_observation, _, _, _ = self._env.step(_LIBERO_DUMMY_ACTION)
        self.reset_evidence = ResetEvidence.from_components(
            seed=self._seed,
            initial_state_source=self._initial_state_source,
            settle_steps=self._settle_steps,
            initial_state=self._initial_state,
            robot0_eef_pos=self._last_raw_observation["robot0_eef_pos"],
            robot0_eef_quat=self._last_raw_observation["robot0_eef_quat"],
        )
        return self._observation(self._last_raw_observation)

    def step(self, action: np.ndarray) -> StepResult:
        self._ensure_open()
        raw_observation, reward, done, info = self._env.step(np.asarray(action))
        self._last_raw_observation = raw_observation
        success = bool(self._env.check_success())
        return StepResult(
            observation=self._observation(raw_observation),
            reward=float(reward),
            done=bool(done) or success,
            success=success,
            info=dict(info),
        )

    def render_frame(self) -> np.ndarray | None:
        if self._last_raw_observation is None:
            return None
        return self._image(self._last_raw_observation, "agentview_image")

    def close(self) -> None:
        if self._closed:
            return
        self._env.close()
        self._closed = True

    def _observation(self, raw_observation: dict[str, np.ndarray]) -> Observation:
        return Observation(
            images={
                "agentview": self._image(raw_observation, "agentview_image"),
                "wrist": self._image(raw_observation, "robot0_eye_in_hand_image"),
            },
            proprioception=np.concatenate(
                (
                    np.asarray(raw_observation["robot0_eef_pos"], dtype=np.float32),
                    _quaternion_to_axis_angle(raw_observation["robot0_eef_quat"]),
                    np.asarray(raw_observation["robot0_gripper_qpos"], dtype=np.float32),
                )
            ).astype(np.float32, copy=False),
            instruction=self._instruction,
        )

    @staticmethod
    def _image(raw_observation: dict[str, np.ndarray], key: str) -> np.ndarray:
        return np.asarray(raw_observation[key])[::-1].copy()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LIBERO episode is closed")


def _validate_index(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).copy()
    if quaternion.shape != (4,):
        raise ValueError("robot0_eef_quat must contain four values")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        return np.zeros(3, dtype=np.float32)
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    sine = float(np.linalg.norm(quaternion[:3]))
    if sine < 1e-7:
        return np.zeros(3, dtype=np.float32)
    return (quaternion[:3] * (2.0 * np.arctan2(sine, float(quaternion[3])) / sine)).astype(np.float32)
