from __future__ import annotations

from pathlib import Path

import numpy as np

from libero_platform.policies.base import (
    EpisodeContext,
    PolicyAdapter,
    PolicyRequest,
    PolicyResponse,
)
from libero_platform.preflight import validate_demo_actions


class ReferenceActionsExhaustedError(RuntimeError):
    failure_type = "reference_actions_exhausted"

    def __init__(self) -> None:
        super().__init__(self.failure_type)


class DemoReplayPolicyAdapter(PolicyAdapter):
    """Replay the official actions paired with a selected LIBERO demo state."""

    def __init__(self, model_key: str, dataset_directory: Path) -> None:
        self._model_key = model_key
        self._dataset_directory = Path(dataset_directory)
        self._actions: np.ndarray | None = None
        self._cursor = 0

    def begin_episode(self, context: EpisodeContext) -> None:
        if Path(context.task_name).name != context.task_name:
            raise RuntimeError("invalid LIBERO task name for demonstration replay")
        demo_file = self._dataset_directory / f"{context.task_name}_demo.hdf5"
        if not demo_file.is_file():
            raise RuntimeError(
                f"LIBERO demonstration file is missing: {demo_file.name}"
            )

        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py dependency is not installed") from exc

        action_path = f"data/demo_{context.initial_state_id}/actions"
        try:
            with h5py.File(demo_file, "r") as dataset:
                if action_path not in dataset:
                    raise RuntimeError(
                        f"LIBERO demonstration dataset is missing: {action_path}"
                    )
                actions = np.asarray(dataset[action_path][...]).copy()
        except OSError as exc:
            raise RuntimeError(
                f"could not open LIBERO demonstration file: {demo_file.name}"
            ) from exc

        self._actions = validate_demo_actions(actions)
        self._cursor = 0

    def predict(self, request: PolicyRequest) -> PolicyResponse:
        del request
        if self._actions is None:
            raise RuntimeError("reference_episode_not_started")
        if self._cursor >= len(self._actions):
            raise ReferenceActionsExhaustedError()

        action = self._actions[self._cursor].copy()
        self._cursor += 1
        return PolicyResponse(
            action=action,
            inference_ms=0.0,
            model_key=self._model_key,
            device="cpu",
        )
