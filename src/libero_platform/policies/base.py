from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
ImageArray = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class EpisodeContext:
    suite: str
    task_id: int
    task_name: str
    initial_state_id: int
    seed: int


@dataclass(frozen=True)
class PolicyRequest:
    run_id: str
    episode_id: int
    step_id: int
    instruction: str
    images: Mapping[str, ImageArray]
    proprioception: FloatArray
    previous_action: FloatArray | None


@dataclass(frozen=True)
class PolicyResponse:
    action: FloatArray
    inference_ms: float
    model_key: str
    device: str
    raw_action: FloatArray | None = None
    action_chunk: FloatArray | None = None
    action_transform: str = ""
    action_clipped: bool = False
    failure_type: str = ""
    error: str = ""
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)


class PolicyAdapter(ABC):
    @property
    def model_key(self) -> str:
        return self._model_key

    def identity(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "checkpoint": None,
            "revision": None,
            "precision": None,
            "device": "unavailable",
            "ready": False,
        }

    def load(self) -> None:
        return None

    def begin_episode(self, context: EpisodeContext) -> None:
        del context

    @abstractmethod
    def predict(self, request: PolicyRequest) -> PolicyResponse:
        raise NotImplementedError

    def close(self) -> None:
        return None


def validate_action(action: npt.ArrayLike) -> FloatArray:
    try:
        value = np.asarray(action, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("action must contain numeric float32 values") from exc
    if value.shape != (7,):
        raise ValueError("action must contain exactly 7 values")
    if not np.isfinite(value).all():
        raise ValueError("action must contain finite values")
    if (value < -1.0).any() or (value > 1.0).any():
        raise ValueError("action values must stay in [-1, 1]")
    return value


def validate_action_chunk(action_chunk: npt.ArrayLike) -> FloatArray:
    try:
        value = np.asarray(action_chunk, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("action_chunk must contain numeric float32 values") from exc
    if value.ndim != 2 or value.shape[1] != 7 or value.shape[0] < 1:
        raise ValueError(
            f"action_chunk must have shape (n_action_steps, 7), got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError("action_chunk must contain finite values")
    if (value < -1.0).any() or (value > 1.0).any():
        raise ValueError("action_chunk values must stay in [-1, 1]")
    return value
