from __future__ import annotations

import numpy as np

from .base import Observation, ResetEvidence, StepResult


class FakeBackend:
    """A deterministic, dependency-free backend for runner integration tests."""

    def __init__(self, *, success_step: int = 1, fail_episode: int | None = None) -> None:
        if success_step < 1:
            raise ValueError("success_step must be at least one")
        if fail_episode is not None and fail_episode < 0:
            raise ValueError("fail_episode must be non-negative or null")
        self._success_step = success_step
        self._fail_episode = fail_episode
        self._opened_episodes = 0

    def list_tasks(self, suite: str) -> list[dict[str, object]]:
        return [
            {
                "task_id": task_id,
                "task_name": f"fake_task_{task_id}",
            }
            for task_id in range(100)
        ]

    def open_episode(
        self, suite: str, task_id: int, initial_state_id: int, max_steps: int, seed: int
    ) -> FakeEpisode:
        del suite
        episode_index = self._opened_episodes
        self._opened_episodes += 1
        return FakeEpisode(
            task_id=task_id,
            initial_state_id=initial_state_id,
            max_steps=max_steps,
            seed=seed,
            success_step=self._success_step,
            force_failure=episode_index == self._fail_episode,
        )


class FakeEpisode:
    model = None
    data = None

    def __init__(
        self,
        *,
        task_id: int,
        initial_state_id: int,
        max_steps: int,
        seed: int,
        success_step: int,
        force_failure: bool,
    ) -> None:
        self._task_id = task_id
        self._initial_state_id = initial_state_id
        self._max_steps = max_steps
        self._seed = seed
        self._success_step = success_step
        self._force_failure = force_failure
        self._step_count = 0
        self._closed = False
        self.reset_evidence: ResetEvidence | None = None

    def reset(self) -> Observation:
        self._step_count = 0
        self.reset_evidence = ResetEvidence.from_components(
            seed=self._seed,
            initial_state_source="fake",
            settle_steps=0,
            initial_state=self._initial_state(),
            robot0_eef_pos=self._eef_position(),
            robot0_eef_quat=self._eef_quaternion(),
        )
        return self._observation()

    def step(self, action: np.ndarray) -> StepResult:
        if self._closed:
            raise RuntimeError("fake episode is closed")
        self._step_count += 1
        success = not self._force_failure and self._step_count >= self._success_step
        done = success or self._step_count >= self._max_steps
        return StepResult(
            observation=self._observation(),
            reward=1.0 if success else 0.0,
            done=done,
            success=success,
            info={"action_norm": float(np.linalg.norm(action))},
        )

    def render_frame(self) -> np.ndarray:
        return self._image()

    def close(self) -> None:
        self._closed = True

    def _observation(self) -> Observation:
        return Observation(
            images={"agentview": self._image()},
            proprioception=np.array(
                [self._task_id, self._initial_state_id, self._step_count],
                dtype=np.float32,
            ),
            instruction=f"complete fake task {self._task_id}",
        )

    def _image(self) -> np.ndarray:
        value = (
            self._task_id * 31 + self._initial_state_id * 17 + self._step_count * 7
        ) % 256
        return np.full((8, 8, 3), value, dtype=np.uint8)

    def _initial_state(self) -> np.ndarray:
        return np.array(
            [self._task_id, self._initial_state_id, self._seed],
            dtype=np.float32,
        )

    def _eef_position(self) -> np.ndarray:
        return np.array(
            [self._task_id, self._initial_state_id, self._seed],
            dtype=np.float32,
        )

    @staticmethod
    def _eef_quaternion() -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
