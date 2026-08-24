import numpy as np

from libero_platform.policies.fixed_h_action_buffer import FixedHActionBuffer


class ConstantOutOfRangePredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict_action_chunk(self, observation: object) -> np.ndarray:
        self.calls += 1
        return np.full((1, 50, 7), 2.0, dtype=np.float32)


def test_static_h20_safety_clips_without_discarding_or_changing_refill_boundary() -> None:
    predictor = ConstantOutOfRangePredictor()
    buffer = FixedHActionBuffer(predictor, safety_enabled=True)

    first = buffer.next_action(None)

    assert np.array_equal(first.action, np.ones(7, dtype=np.float32))
    assert first.telemetry["safety_enabled"] is True
    assert first.telemetry["range_violation"] is True
    assert first.telemetry["range_clipped"] is True
    assert first.telemetry["buffer_discarded"] is False
    assert first.telemetry["actual_horizon"] == 20
    assert first.telemetry["forced_horizon_next"] is None
    assert buffer.buffered_actions == 49

    for _ in range(19):
        buffer.next_action(None)
    refill = buffer.next_action(None)

    assert predictor.calls == 2
    assert refill.telemetry["planned_horizon"] == 20


def test_adaptive_h20_safety_clips_discards_and_forces_h1() -> None:
    predictor = ConstantOutOfRangePredictor()
    buffer = FixedHActionBuffer(
        predictor, safety_enabled=True, replan_after_safety_violation=True
    )

    first = buffer.next_action(None)
    second = buffer.next_action(None)

    assert np.array_equal(first.action, np.ones(7, dtype=np.float32))
    assert first.telemetry["range_clipped"] is True
    assert first.telemetry["buffer_discarded"] is True
    assert first.telemetry["forced_horizon_next"] == 1
    assert second.telemetry["planned_horizon"] == 1
    assert second.telemetry["actual_horizon"] == 1
    assert predictor.calls == 2


def test_safety_disabled_preserves_native_postprocessed_action() -> None:
    predictor = ConstantOutOfRangePredictor()
    buffer = FixedHActionBuffer(predictor, safety_enabled=False)

    release = buffer.next_action(None)

    assert np.array_equal(release.action, np.full(7, 2.0, dtype=np.float32))
    assert release.telemetry["safety_enabled"] is False
    assert release.telemetry["range_violation"] is True
    assert release.telemetry["range_clipped"] is False
    assert release.telemetry["buffer_discarded"] is False
    assert release.telemetry["actual_horizon"] == 20
    assert release.telemetry["forced_horizon_next"] is None
    assert predictor.calls == 1
    assert buffer.buffered_actions == 49
