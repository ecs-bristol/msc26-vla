import numpy as np

from libero_platform.policies.fixed_h_action_buffer import FixedHActionBuffer


class ConstantOutOfRangePredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict_action_chunk(self, observation: object) -> np.ndarray:
        self.calls += 1
        return np.full((1, 50, 7), 2.0, dtype=np.float32)


def test_safety_enabled_clips_discards_and_forces_h1() -> None:
    predictor = ConstantOutOfRangePredictor()
    buffer = FixedHActionBuffer(predictor, safety_enabled=True)

    first = buffer.next_action(None)

    assert np.array_equal(first.action, np.ones(7, dtype=np.float32))
    assert first.telemetry["safety_enabled"] is True
    assert first.telemetry["range_violation"] is True
    assert first.telemetry["range_clipped"] is True
    assert first.telemetry["actual_horizon"] == 1
    assert first.telemetry["forced_horizon_next"] == 1
    assert buffer.buffered_actions == 0

    second = buffer.next_action(None)

    assert predictor.calls == 2
    assert second.telemetry["planned_horizon"] == 1


def test_safety_disabled_preserves_native_postprocessed_action() -> None:
    predictor = ConstantOutOfRangePredictor()
    buffer = FixedHActionBuffer(predictor, safety_enabled=False)

    release = buffer.next_action(None)

    assert np.array_equal(release.action, np.full(7, 2.0, dtype=np.float32))
    assert release.telemetry["safety_enabled"] is False
    assert release.telemetry["range_violation"] is True
    assert release.telemetry["range_clipped"] is False
    assert release.telemetry["actual_horizon"] == 20
    assert release.telemetry["forced_horizon_next"] is None
    assert predictor.calls == 1
    assert buffer.buffered_actions == 49
