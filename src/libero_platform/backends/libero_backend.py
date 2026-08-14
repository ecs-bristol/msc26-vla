from __future__ import annotations

import importlib.util
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
