from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Observation:
    images: dict[str, np.ndarray]
    proprioception: np.ndarray
    instruction: str


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    done: bool
    success: bool
    info: dict[str, object]


@dataclass(frozen=True)
class ResetEvidence:
    seed: int
    initial_state_source: str
    settle_steps: int
    fingerprint: str

    @classmethod
    def from_components(
        cls,
        *,
        seed: int,
        initial_state_source: str,
        settle_steps: int,
        initial_state: np.ndarray,
        robot0_eef_pos: np.ndarray,
        robot0_eef_quat: np.ndarray,
    ) -> ResetEvidence:
        digest = hashlib.sha256()
        for value in (initial_state, robot0_eef_pos, robot0_eef_quat):
            digest.update(np.ascontiguousarray(np.asarray(value)).tobytes())
        return cls(
            seed=seed,
            initial_state_source=initial_state_source,
            settle_steps=settle_steps,
            fingerprint=digest.hexdigest()[:16],
        )


class Episode(Protocol):
    model: object | None
    data: object | None
    reset_evidence: ResetEvidence | None

    def reset(self) -> Observation:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> StepResult:
        raise NotImplementedError

    def render_frame(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class BenchmarkBackend(Protocol):
    def list_tasks(self, suite: str) -> list[dict[str, object]]:
        raise NotImplementedError

    def open_episode(
        self, suite: str, task_id: int, initial_state_id: int, max_steps: int, seed: int
    ) -> Episode:
        raise NotImplementedError
