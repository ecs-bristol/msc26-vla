"""Pre-registered Adaptive-v2a trigger and H20/H1 recovery state machine.

The trigger consumes only the raw action that is about to be executed.  It has
no observation, reward, terminal, or outcome-label input.  Thresholds are
fixed fractions of the declared LIBERO action span [-1, 1], not estimates from
the v1 outcome table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from libero_platform.policies.fixed_h_action_buffer import (
    ActionRelease,
    FixedHActionBuffer,
    _validated_action,
)


_ACTION_DIM = 7
_GRIPPER_INDEX = 6
_LOWER_BOUND = -1.0
_UPPER_BOUND = 1.0
_ACTION_SPAN = _UPPER_BOUND - _LOWER_BOUND
_DEFAULT_HORIZON = 20
_FALLBACK_HORIZON = 1

# Calibration-free, action-space-derived preregistration constants.  Severity
# is excess / action_span, so these correspond to 1% and 5% of [-1, 1].
PERSISTENCE_SEVERITY = 0.01
IMMEDIATE_SEVERITY = 0.05
PERSISTENCE_STEPS = 2
COOLDOWN_H20_ACTIONS = 20


class AdaptiveV2State(StrEnum):
    MONITORING_H20 = "monitoring_h20"
    FALLBACK_H1_PENDING = "fallback_h1_pending"
    FALLBACK_H1 = "fallback_h1"
    COOLDOWN_H20_PENDING = "cooldown_h20_pending"
    COOLDOWN_H20 = "cooldown_h20"


@dataclass(frozen=True)
class TriggerDecision:
    triggered: bool
    state_before: str
    state_after: str
    violation_dimensions: list[int]
    violation_raw_values: list[float]
    violation_bounds: list[float]
    violation_excess: list[float]
    violation_severity: list[float]
    non_gripper_dimensions: list[int]
    trigger_dimensions: list[int]
    trigger_raw_values: list[float]
    trigger_bounds: list[float]
    trigger_excess: list[float]
    trigger_severity: list[float]
    trigger_persistence_counts: list[int]
    immediate_dimensions: list[int]
    persistent_dimensions: list[int]
    persistence_counts: list[int]
    cooldown_actions_remaining_before: int
    cooldown_actions_remaining_after: int


class AdaptiveV2Trigger:
    """Deterministic, outcome-blind severity/persistence trigger.

    A non-gripper dimension triggers immediately at 5% of action span excess,
    or after two consecutive pre-action checks at 1% of span excess.  A single
    lower-severity crossing therefore cannot trigger.  Gripper dimension 6 is
    recorded but never enters either trigger path.
    """

    def __init__(self) -> None:
        self.reset()

    @property
    def state(self) -> AdaptiveV2State:
        return self._state

    def reset(self) -> None:
        self._state = AdaptiveV2State.MONITORING_H20
        self._persistence = [0] * _ACTION_DIM
        self._cooldown_actions_remaining = 0

    def on_refill(self, planned_horizon: int) -> None:
        if self._state is AdaptiveV2State.FALLBACK_H1_PENDING:
            if planned_horizon != _FALLBACK_HORIZON:
                raise RuntimeError("Adaptive-v2a fallback must refill at H1")
            self._state = AdaptiveV2State.FALLBACK_H1
        elif self._state is AdaptiveV2State.COOLDOWN_H20_PENDING:
            if planned_horizon != _DEFAULT_HORIZON:
                raise RuntimeError("Adaptive-v2a recovery must return to H20")
            self._state = AdaptiveV2State.COOLDOWN_H20
            self._cooldown_actions_remaining = COOLDOWN_H20_ACTIONS
        elif self._state is AdaptiveV2State.MONITORING_H20:
            if planned_horizon != _DEFAULT_HORIZON:
                raise RuntimeError("Adaptive-v2a monitoring horizon must be H20")
        else:
            raise RuntimeError(f"unexpected refill while Adaptive-v2a state={self._state}")

    def on_call_finalized(self, *, planned_horizon: int, reason: str) -> None:
        if (
            self._state is AdaptiveV2State.FALLBACK_H1
            and planned_horizon == _FALLBACK_HORIZON
            and reason == "horizon"
        ):
            self._state = AdaptiveV2State.COOLDOWN_H20_PENDING
            return
        if (
            self._state is AdaptiveV2State.COOLDOWN_H20
            and planned_horizon == _DEFAULT_HORIZON
            and reason == "horizon"
        ):
            if self._cooldown_actions_remaining != 0:
                raise RuntimeError("Adaptive-v2a cooldown ended before twenty H20 actions")
            self._state = AdaptiveV2State.MONITORING_H20
            self._persistence = [0] * _ACTION_DIM

    def assess(self, action: npt.NDArray[np.float32]) -> TriggerDecision:
        """Evaluate the raw action before the caller can send it to the environment."""

        raw = np.asarray(action, dtype=np.float32)
        state_before = self._state
        cooldown_before = self._cooldown_actions_remaining
        excess_by_dimension = np.maximum(np.abs(raw) - _UPPER_BOUND, 0.0)
        violation_dimensions = np.flatnonzero(excess_by_dimension > 0.0).astype(int).tolist()
        non_gripper = [
            dimension
            for dimension in violation_dimensions
            if dimension != _GRIPPER_INDEX
        ]
        severities = excess_by_dimension / _ACTION_SPAN

        immediate: list[int] = []
        persistent: list[int] = []
        triggered = False
        if self._state is AdaptiveV2State.MONITORING_H20:
            for dimension in range(_GRIPPER_INDEX):
                if severities[dimension] >= PERSISTENCE_SEVERITY:
                    self._persistence[dimension] += 1
                else:
                    self._persistence[dimension] = 0
                if severities[dimension] >= IMMEDIATE_SEVERITY:
                    immediate.append(dimension)
                if self._persistence[dimension] >= PERSISTENCE_STEPS:
                    persistent.append(dimension)
            self._persistence[_GRIPPER_INDEX] = 0
            triggered = bool(immediate or persistent)
            if triggered:
                self._state = AdaptiveV2State.FALLBACK_H1_PENDING
        else:
            self._persistence = [0] * _ACTION_DIM
            if self._state is AdaptiveV2State.COOLDOWN_H20:
                if self._cooldown_actions_remaining <= 0:
                    raise RuntimeError("Adaptive-v2a cooldown action counter underflow")
                self._cooldown_actions_remaining -= 1

        trigger_dimensions = sorted(set(immediate) | set(persistent))

        def values(
            dimensions: list[int],
        ) -> tuple[list[float], list[float], list[float], list[float]]:
            raw_values = [float(raw[index]) for index in dimensions]
            bounds = [
                float(_UPPER_BOUND if raw[index] > 0 else _LOWER_BOUND)
                for index in dimensions
            ]
            excesses = [float(excess_by_dimension[index]) for index in dimensions]
            dimension_severities = [float(severities[index]) for index in dimensions]
            return raw_values, bounds, excesses, dimension_severities

        violation_raw, violation_bounds, violation_excess, violation_severity = values(
            violation_dimensions
        )
        trigger_raw, trigger_bounds, trigger_excess, trigger_severity = values(
            trigger_dimensions
        )
        return TriggerDecision(
            triggered=triggered,
            state_before=str(state_before),
            state_after=str(self._state),
            violation_dimensions=violation_dimensions,
            violation_raw_values=violation_raw,
            violation_bounds=violation_bounds,
            violation_excess=violation_excess,
            violation_severity=violation_severity,
            non_gripper_dimensions=non_gripper,
            trigger_dimensions=trigger_dimensions,
            trigger_raw_values=trigger_raw,
            trigger_bounds=trigger_bounds,
            trigger_excess=trigger_excess,
            trigger_severity=trigger_severity,
            trigger_persistence_counts=[self._persistence[index] for index in trigger_dimensions],
            immediate_dimensions=immediate,
            persistent_dimensions=persistent,
            persistence_counts=list(self._persistence),
            cooldown_actions_remaining_before=cooldown_before,
            cooldown_actions_remaining_after=self._cooldown_actions_remaining,
        )


class AdaptiveV2ActionBuffer(FixedHActionBuffer):
    """H20 buffer with the preregistered v2a trigger and native actions."""

    def __init__(self, predictor: object) -> None:
        super().__init__(
            predictor,
            horizon=_DEFAULT_HORIZON,
            safety_enabled=True,
            replan_after_safety_violation=False,
            clip_actions=False,
        )
        self._v2_trigger = AdaptiveV2Trigger()

    @property
    def v2_state(self) -> AdaptiveV2State:
        return self._v2_trigger.state

    def reset(self) -> None:
        super().reset()
        self._v2_trigger.reset()

    def _refill(self, observation: object) -> None:
        super()._refill(observation)
        self._v2_trigger.on_refill(self._active_horizon)
        self._telemetry[-1]["adaptive_v2_state"] = str(self._v2_trigger.state)

    def _record_call_finalized(self, *, reason: str) -> None:
        planned_horizon = self._active_horizon
        super()._record_call_finalized(reason=reason)
        self._v2_trigger.on_call_finalized(
            planned_horizon=planned_horizon,
            reason=reason,
        )
        self._telemetry[-1]["adaptive_v2_state_after_finalization"] = str(
            self._v2_trigger.state
        )

    def _release_action(self) -> ActionRelease:
        if not self._buffer or self._buffer_origin is None:
            raise RuntimeError("action buffer unexpectedly empty")

        buffer_size_before = len(self._buffer)
        action = _validated_action(self._buffer.pop(0))
        chunk_origin = self._buffer_origin
        active_horizon = self._active_horizon
        chunk_action_index = self._buffer_action_index
        self._buffer_action_index += 1

        # This assessment occurs before ActionRelease is returned, hence before
        # the evaluator can call env.step with the unchanged native action.
        decision = self._v2_trigger.assess(action)
        buffer_discarded = decision.triggered
        horizon_complete = self._buffer_action_index >= active_horizon
        planned_remainder = active_horizon - self._buffer_action_index
        record: dict[str, object] = {
            "event": "action_release",
            "chunk_origin": f"model_invocation:{chunk_origin}",
            "chunk_action_index": chunk_action_index,
            "planned_horizon": active_horizon,
            "buffer_size_before": buffer_size_before,
            "buffer_size_after": 0 if buffer_discarded or horizon_complete else len(self._buffer),
            "model_invocation": chunk_origin,
            "model_invoked": chunk_action_index == 0,
            "safety_enabled": True,
            "clip_actions": False,
            "range_violation": bool(decision.violation_dimensions),
            "range_violation_dimensions": decision.violation_dimensions,
            "range_violation_raw_values": decision.violation_raw_values,
            "range_violation_bounds": decision.violation_bounds,
            "range_violation_excess": decision.violation_excess,
            "range_violation_severity": decision.violation_severity,
            "range_violation_max_excess": max(decision.violation_excess, default=0.0),
            "qualifying_non_gripper_violation": bool(decision.non_gripper_dimensions),
            "trigger_range_violation": decision.triggered,
            "trigger_violation_dimensions": decision.trigger_dimensions,
            "gripper_only_range_violation": (
                bool(decision.violation_dimensions) and not decision.non_gripper_dimensions
            ),
            "range_clipped": False,
            "buffer_discarded": buffer_discarded,
            "forced_horizon_next": _FALLBACK_HORIZON if buffer_discarded else None,
            "gripper_index": _GRIPPER_INDEX,
            "gripper_negative_is_open": True,
            "gripper_positive_is_closed": True,
            "adaptive_v2_enabled": True,
            "adaptive_v2_evaluated_before_action_execution": True,
            "adaptive_v2_triggered": decision.triggered,
            "adaptive_v2_state_before": decision.state_before,
            "adaptive_v2_state_after": decision.state_after,
            "adaptive_v2_immediate_dimensions": decision.immediate_dimensions,
            "adaptive_v2_persistent_dimensions": decision.persistent_dimensions,
            "adaptive_v2_persistence_counts": decision.persistence_counts,
            "adaptive_v2_trigger_raw_values": decision.trigger_raw_values,
            "adaptive_v2_trigger_bounds": decision.trigger_bounds,
            "adaptive_v2_trigger_excess": decision.trigger_excess,
            "adaptive_v2_trigger_severity": decision.trigger_severity,
            "adaptive_v2_trigger_persistence_counts": decision.trigger_persistence_counts,
            "adaptive_v2_trigger_tail_discarded_actions": (
                planned_remainder if decision.triggered else 0
            ),
            "adaptive_v2_buffer_entries_cleared": len(self._buffer) if decision.triggered else 0,
            "adaptive_v2_horizon_before": active_horizon,
            "adaptive_v2_horizon_after": _FALLBACK_HORIZON if decision.triggered else None,
            "adaptive_v2_recovery_horizon": _DEFAULT_HORIZON if decision.triggered else None,
            "adaptive_v2_cooldown_actions_remaining_before": (
                decision.cooldown_actions_remaining_before
            ),
            "adaptive_v2_cooldown_actions_remaining_after": (
                decision.cooldown_actions_remaining_after
            ),
            "adaptive_v2_persistence_severity": PERSISTENCE_SEVERITY,
            "adaptive_v2_immediate_severity": IMMEDIATE_SEVERITY,
            "adaptive_v2_persistence_steps": PERSISTENCE_STEPS,
            "adaptive_v2_cooldown_h20_actions": COOLDOWN_H20_ACTIONS,
        }
        self._telemetry.append(record)
        if buffer_discarded:
            self._record_call_finalized(reason="trigger")
            self._clear_buffer()
            self._force_next_horizon_one = True
        elif horizon_complete:
            self._record_call_finalized(reason="horizon")
            self._clear_buffer()
        return ActionRelease(action=action.copy(), telemetry=dict(record))
