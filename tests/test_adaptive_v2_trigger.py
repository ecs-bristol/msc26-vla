from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from libero_platform.policies.adaptive_v2_trigger import (
    AdaptiveV2ActionBuffer,
    AdaptiveV2State,
    IMMEDIATE_SEVERITY,
    PERSISTENCE_SEVERITY,
    PERSISTENCE_STEPS,
)


ROOT = Path(__file__).resolve().parents[1]
TRIGGER_SOURCE = ROOT / "src" / "libero_platform" / "policies" / "adaptive_v2_trigger.py"


def _chunk(actions: list[np.ndarray] | None = None) -> np.ndarray:
    value = np.zeros((1, 50, 7), dtype=np.float32)
    for index, action in enumerate(actions or []):
        value[0, index] = action
    return value


class Predictor:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self.chunks = list(chunks)
        self.calls = 0

    def predict_action_chunk(self, observation: object) -> np.ndarray:
        del observation
        self.calls += 1
        return self.chunks.pop(0)


def test_single_mild_violation_does_not_trigger_but_persistence_does() -> None:
    mild = np.array([1.03, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    buffer = AdaptiveV2ActionBuffer(Predictor([_chunk([mild, mild])]))

    first = buffer.next_action(None)
    second = buffer.next_action(None)

    np.testing.assert_array_equal(first.action, mild)
    assert first.telemetry["adaptive_v2_triggered"] is False
    assert first.telemetry["buffer_discarded"] is False
    assert first.telemetry["adaptive_v2_persistence_counts"][0] == 1
    assert second.telemetry["adaptive_v2_triggered"] is True
    assert second.telemetry["adaptive_v2_immediate_dimensions"] == []
    assert second.telemetry["adaptive_v2_persistent_dimensions"] == [0]
    assert second.telemetry["adaptive_v2_trigger_persistence_counts"] == [2]
    assert second.telemetry["adaptive_v2_trigger_tail_discarded_actions"] == 18
    assert second.telemetry["adaptive_v2_evaluated_before_action_execution"] is True


def test_severe_non_gripper_violation_triggers_immediately_without_clipping() -> None:
    severe = np.array([-1.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    buffer = AdaptiveV2ActionBuffer(Predictor([_chunk([severe])]))

    release = buffer.next_action(None)

    np.testing.assert_array_equal(release.action, severe)
    assert release.telemetry["adaptive_v2_triggered"] is True
    assert release.telemetry["adaptive_v2_immediate_dimensions"] == [0]
    assert release.telemetry["adaptive_v2_trigger_raw_values"] == [float(severe[0])]
    assert release.telemetry["adaptive_v2_trigger_bounds"] == [-1.0]
    assert release.telemetry["adaptive_v2_trigger_excess"][0] > 0.19
    assert release.telemetry["adaptive_v2_trigger_severity"][0] > 0.09
    assert release.telemetry["range_clipped"] is False


def test_gripper_only_violation_is_recorded_and_never_triggers() -> None:
    gripper = np.array([0, 0, 0, 0, 0, 0, -1.5], dtype=np.float32)
    buffer = AdaptiveV2ActionBuffer(Predictor([_chunk([gripper, gripper])]))

    first = buffer.next_action(None)
    second = buffer.next_action(None)

    for release in (first, second):
        np.testing.assert_array_equal(release.action, gripper)
        assert release.telemetry["range_violation_dimensions"] == [6]
        assert release.telemetry["gripper_only_range_violation"] is True
        assert release.telemetry["adaptive_v2_triggered"] is False
        assert release.telemetry["buffer_discarded"] is False


def test_fallback_cooldown_and_recovery_state_machine_is_explicit() -> None:
    severe = np.array([1.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    predictor = Predictor(
        [
            _chunk([severe]),
            _chunk(),
            _chunk(),
            _chunk([severe]),
        ]
    )
    buffer = AdaptiveV2ActionBuffer(predictor)

    triggered = buffer.next_action(None)
    assert triggered.telemetry["planned_horizon"] == 20
    assert buffer.v2_state is AdaptiveV2State.FALLBACK_H1_PENDING

    fallback = buffer.next_action(None)
    assert fallback.telemetry["planned_horizon"] == 1
    assert buffer.v2_state is AdaptiveV2State.COOLDOWN_H20_PENDING

    cooldown = [buffer.next_action(None) for _ in range(20)]
    assert all(release.telemetry["planned_horizon"] == 20 for release in cooldown)
    assert all(release.telemetry["adaptive_v2_triggered"] is False for release in cooldown)
    assert cooldown[0].telemetry["adaptive_v2_cooldown_actions_remaining_before"] == 20
    assert cooldown[-1].telemetry["adaptive_v2_cooldown_actions_remaining_after"] == 0
    assert buffer.v2_state is AdaptiveV2State.MONITORING_H20

    retriggered = buffer.next_action(None)
    assert retriggered.telemetry["planned_horizon"] == 20
    assert retriggered.telemetry["adaptive_v2_triggered"] is True
    assert predictor.calls == 4


def test_preregistered_thresholds_are_fixed_action_span_fractions() -> None:
    assert PERSISTENCE_SEVERITY == 0.01
    assert IMMEDIATE_SEVERITY == 0.05
    assert PERSISTENCE_STEPS == 2
    assert 0 < PERSISTENCE_SEVERITY < IMMEDIATE_SEVERITY


def test_trigger_implementation_cannot_read_an_outcome_field() -> None:
    tree = ast.parse(TRIGGER_SOURCE.read_text(encoding="utf-8"))
    forbidden: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and "success" in node.id.lower():
            forbidden.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and "success" in node.attr.lower():
            forbidden.append((node.lineno, node.attr))
        elif isinstance(node, ast.Subscript):
            key = node.slice
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and "success" in key.value.lower()
            ):
                forbidden.append((node.lineno, key.value))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and node.args:
                key = node.args[0]
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and "success" in key.value.lower()
                ):
                    forbidden.append((node.lineno, key.value))
    assert forbidden == []
