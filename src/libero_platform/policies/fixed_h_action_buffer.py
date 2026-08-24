"""Project-owned fixed-horizon action buffering for the SmolVLA parity gate.

This module deliberately does not know about environments or LeRobot's native
``select_action`` queue.  Its dependency receives one observation and must
return a complete postprocessed action chunk.  Keeping this boundary narrow is
what makes the Fixed-H wrapper independently testable before rollout parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt


Action: TypeAlias = npt.NDArray[np.float32]
TelemetryValue: TypeAlias = str | int | float | bool | None
TelemetryRecord: TypeAlias = dict[str, TelemetryValue]

_ACTION_DIM = 7
_CHUNK_SIZE = 50
_DEFAULT_HORIZON = 20
_ALLOWED_HORIZONS = frozenset({1, 5, 10, 20, 50})
_GRIPPER_INDEX = 6


class InvalidPolicyActionError(RuntimeError):
    """A policy chunk or action violates the frozen LIBERO raw-action contract."""


class ActionChunkPredictor(Protocol):
    """Minimal dependency boundary used by the project-owned buffer."""

    def predict_action_chunk(self, observation: object) -> object:
        """Return one postprocessed chunk with shape ``[1, 50, 7]``."""


@dataclass(frozen=True)
class ActionRelease:
    """One safe-to-send action and its JSON-serializable provenance record."""

    action: Action
    telemetry: Mapping[str, TelemetryValue]


class FixedHActionBuffer:
    """Own and release postprocessed SmolVLA actions at a permitted horizon.

    The wrapper intentionally calls only ``predict_action_chunk``.  It does
    not call or rely on a base policy's ``select_action`` implementation, and
    it never changes a native LeRobot ``n_action_steps`` configuration.
    """

    def __init__(
        self,
        predictor: ActionChunkPredictor,
        *,
        horizon: int = _DEFAULT_HORIZON,
        safety_enabled: bool = True,
        replan_after_safety_violation: bool = False,
    ) -> None:
        if type(horizon) is not int or horizon not in _ALLOWED_HORIZONS:
            allowed = ", ".join(str(value) for value in sorted(_ALLOWED_HORIZONS))
            raise ValueError(f"horizon must be one of {{{allowed}}}")
        if replan_after_safety_violation and not safety_enabled:
            raise ValueError("replan_after_safety_violation requires safety_enabled=True")
        self._predictor = predictor
        self._horizon = horizon
        self._buffer: list[Action] = []
        self._buffer_origin: int | None = None
        self._buffer_action_index = 0
        self._model_invocations = 0
        self._safety_enabled = bool(safety_enabled)
        self._replan_after_safety_violation = bool(replan_after_safety_violation)
        self._force_next_horizon_one = False
        self._active_horizon = horizon
        self._telemetry: list[TelemetryRecord] = []

    @property
    def model_invocations(self) -> int:
        return self._model_invocations

    @property
    def buffered_actions(self) -> int:
        return len(self._buffer)

    @property
    def telemetry(self) -> tuple[TelemetryRecord, ...]:
        """Immutable snapshot of JSON-safe refill and release records."""

        return tuple(dict(record) for record in self._telemetry)

    def reset(self) -> None:
        """Clear all episode-local actions; never carry them across episodes."""

        self._clear_buffer()
        self._force_next_horizon_one = False

    def next_action(self, observation: object) -> ActionRelease:
        """Return the next action, refilling from the supplied observation if needed.

        All exceptions clear the project-owned buffer before they propagate.
        In particular, invalid dimensions or non-finite values are rejected,
        never replaced with a zero action.
        """

        try:
            if not self._buffer:
                self._refill(observation)
            return self._release_action()
        except Exception:
            self._clear_buffer()
            raise

    def _refill(self, observation: object) -> None:
        planned_horizon = 1 if self._force_next_horizon_one else self._horizon
        self._force_next_horizon_one = False
        chunk = _validated_chunk(self._predictor.predict_action_chunk(observation))

        self._model_invocations += 1
        self._buffer = [row.copy() for row in chunk[0]]
        self._buffer_origin = self._model_invocations
        self._buffer_action_index = 0
        self._telemetry.append(
            {
                "event": "refill",
                "chunk_origin": f"model_invocation:{self._buffer_origin}",
                "model_invocation": self._buffer_origin,
                "planned_horizon": planned_horizon,
                "chunk_size": _CHUNK_SIZE,
                "action_dim": _ACTION_DIM,
            }
        )
        self._active_horizon = planned_horizon

    def _release_action(self) -> ActionRelease:
        if not self._buffer or self._buffer_origin is None:
            raise RuntimeError("action buffer unexpectedly empty")

        buffer_size_before = len(self._buffer)
        action = _validated_action(self._buffer.pop(0))
        chunk_origin = self._buffer_origin
        active_horizon = self._active_horizon
        chunk_action_index = self._buffer_action_index
        self._buffer_action_index += 1
        range_violation = bool((action < -1.0).any() or (action > 1.0).any())
        range_clipped = self._safety_enabled and range_violation

        buffer_discarded = range_clipped and self._replan_after_safety_violation
        if range_clipped:
            released_action = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)
        else:
            released_action = action

        if buffer_discarded:
            self._clear_buffer()
            self._force_next_horizon_one = True
        else:
            if self._buffer_action_index >= active_horizon:
                # The next action must be planned from its new observation;
                # no unused tail of a 50-step model chunk may leak through.
                self._clear_buffer()

        record: TelemetryRecord = {
            "event": "action_release",
            "chunk_origin": f"model_invocation:{chunk_origin}",
            "chunk_action_index": chunk_action_index,
            "planned_horizon": active_horizon,
            "actual_horizon": active_horizon,
            "buffer_size_before": buffer_size_before,
            "buffer_size_after": len(self._buffer),
            "model_invocation": chunk_origin,
            "model_invoked": chunk_action_index == 0,
            "safety_enabled": self._safety_enabled,
            "range_violation": range_violation,
            "range_clipped": range_clipped,
            "buffer_discarded": buffer_discarded,
            "forced_horizon_next": 1 if buffer_discarded else None,
            "gripper_index": _GRIPPER_INDEX,
            "gripper_negative_is_open": True,
            "gripper_positive_is_closed": True,
        }
        self._telemetry.append(record)
        return ActionRelease(action=released_action.copy(), telemetry=dict(record))

    def _clear_buffer(self) -> None:
        self._buffer.clear()
        self._buffer_origin = None
        self._buffer_action_index = 0


def _validated_chunk(value: object) -> npt.NDArray[np.float32]:
    """Convert and validate the single-batch, complete postprocessed chunk."""

    try:
        array = _to_numpy(value).astype(np.float32, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidPolicyActionError("policy action chunk must be numeric") from exc
    if array.shape != (1, _CHUNK_SIZE, _ACTION_DIM):
        raise InvalidPolicyActionError(
            "policy action chunk must have shape [1, 50, 7] for batch-size-one parity"
        )
    return array


def _validated_action(value: object) -> Action:
    try:
        action = _to_numpy(value).astype(np.float32, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidPolicyActionError("policy action must be numeric") from exc
    if action.shape != (_ACTION_DIM,):
        raise InvalidPolicyActionError("policy action must have shape (7,)")
    if not np.isfinite(action).all():
        raise InvalidPolicyActionError("policy action must contain only finite values")
    return action


def _to_numpy(value: object) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    on_cpu = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.asarray(on_cpu.numpy() if hasattr(on_cpu, "numpy") else on_cpu)
