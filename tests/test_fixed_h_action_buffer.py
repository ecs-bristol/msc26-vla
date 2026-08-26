from __future__ import annotations

import json

import numpy as np
import pytest

from libero_platform.policies.fixed_h_action_buffer import (
    FixedHActionBuffer,
    InvalidPolicyActionError,
)


def chunk(*, first: np.ndarray | None = None, fill: float = 0.0) -> np.ndarray:
    value = np.full((1, 50, 7), fill, dtype=np.float32)
    if first is not None:
        value[0, 0] = first
    return value


class FakeChunkPredictor:
    def __init__(self, chunks: list[object]) -> None:
        self.chunks = list(chunks)
        self.observations: list[object] = []
        self.select_action_calls = 0

    def predict_action_chunk(self, observation: object) -> object:
        self.observations.append(observation)
        return self.chunks.pop(0)

    def select_action(self, observation: object) -> object:
        del observation
        self.select_action_calls += 1
        raise AssertionError("the project-owned buffer must never call select_action")


def test_fixed_h_twenty_refills_once_and_releases_twenty_actions() -> None:
    predictor = FakeChunkPredictor([chunk(fill=0.25), chunk(fill=-0.25)])
    buffer = FixedHActionBuffer(predictor)

    first_twenty = [buffer.next_action({"step": index}) for index in range(20)]
    twenty_first = buffer.next_action({"step": 20})

    assert len(predictor.observations) == 2
    assert buffer.model_invocations == 2
    assert predictor.select_action_calls == 0
    assert buffer.buffered_actions == 49
    assert all(release.action.tolist() == pytest.approx([0.25] * 7) for release in first_twenty)
    assert twenty_first.action.tolist() == pytest.approx([-0.25] * 7)
    assert first_twenty[0].telemetry["planned_horizon"] == 20
    assert first_twenty[0].telemetry["actual_horizon"] == 20
    assert first_twenty[0].telemetry["buffer_size_before"] == 50
    assert first_twenty[19].telemetry["buffer_size_after"] == 0
    assert twenty_first.telemetry["model_invocation"] == 2


@pytest.mark.parametrize("horizon", [1, 5, 10, 20, 25, 30, 50])
def test_permitted_static_horizons_refill_at_the_configured_boundary(horizon: int) -> None:
    predictor = FakeChunkPredictor([chunk(fill=0.25), chunk(fill=-0.25)])
    buffer = FixedHActionBuffer(predictor, horizon=horizon)

    releases = [buffer.next_action({"step": index}) for index in range(horizon + 1)]

    assert len(predictor.observations) == 2
    assert releases[0].telemetry["planned_horizon"] == horizon
    assert releases[horizon - 1].telemetry["buffer_size_after"] == 0
    assert releases[-1].telemetry["model_invocation"] == 2


@pytest.mark.parametrize("horizon", [0, 2, 6, 19, 21, 51, True])
def test_invalid_horizons_are_rejected(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon must be one of"):
        FixedHActionBuffer(FakeChunkPredictor([chunk()]), horizon=horizon)


def test_reset_discards_episode_local_actions() -> None:
    predictor = FakeChunkPredictor([chunk(fill=0.1), chunk(fill=0.2)])
    buffer = FixedHActionBuffer(predictor)

    buffer.next_action("episode-one")
    assert buffer.buffered_actions == 49
    buffer.reset()
    assert buffer.buffered_actions == 0

    release = buffer.next_action("episode-two")
    assert len(predictor.observations) == 2
    assert release.action.tolist() == pytest.approx([0.2] * 7)


@pytest.mark.parametrize(
    ("bad_action", "message"),
    [
        (np.zeros(6, dtype=np.float32), "shape"),
        (np.array([np.nan, 0, 0, 0, 0, 0, 0], dtype=np.float32), "finite"),
        (np.array([np.inf, 0, 0, 0, 0, 0, 0], dtype=np.float32), "finite"),
    ],
)
def test_invalid_released_action_clears_buffer_and_never_returns_a_fallback(
    bad_action: np.ndarray, message: str
) -> None:
    predictor = FakeChunkPredictor([chunk(fill=0.3)])
    buffer = FixedHActionBuffer(predictor)

    buffer.next_action("first-valid-action")
    buffer._buffer[0] = bad_action  # Assert the release boundary itself is guarded.
    with pytest.raises(InvalidPolicyActionError, match=message):
        buffer.next_action("invalid-buffered-action")

    assert buffer.buffered_actions == 0
    assert buffer.model_invocations == 1


def test_invalid_chunk_shape_clears_buffer_and_raises_explicit_error() -> None:
    predictor = FakeChunkPredictor([np.zeros((1, 20, 7), dtype=np.float32)])
    buffer = FixedHActionBuffer(predictor)

    with pytest.raises(InvalidPolicyActionError, match=r"\[1, 50, 7\]"):
        buffer.next_action("bad-chunk")

    assert buffer.buffered_actions == 0


def test_adaptive_h20_detection_only_discards_and_forces_one_step_replan() -> None:
    too_large = np.array([2.0, 0, 0, 0, 0, 0, -2.0], dtype=np.float32)
    predictor = FakeChunkPredictor(
        [chunk(first=too_large, fill=0.4), chunk(fill=-0.4), chunk(fill=0.6)]
    )
    buffer = FixedHActionBuffer(predictor, replan_after_safety_violation=True)

    triggered = buffer.next_action("observation-a")
    recovery = buffer.next_action("observation-b")
    normal = buffer.next_action("observation-c")

    assert triggered.action.tolist() == pytest.approx(too_large.tolist())
    assert triggered.telemetry["range_clipped"] is False
    assert triggered.telemetry["buffer_discarded"] is True
    assert triggered.telemetry["buffer_size_after"] == 0
    assert triggered.telemetry["forced_horizon_next"] == 1
    assert triggered.telemetry["actual_horizon"] == 20
    assert recovery.telemetry["planned_horizon"] == 1
    assert recovery.telemetry["actual_horizon"] == 1
    assert buffer.buffered_actions == 49
    assert normal.telemetry["planned_horizon"] == 20
    assert len(predictor.observations) == 3
    assert predictor.select_action_calls == 0


def test_static_h1_original_does_not_modify_out_of_range_native_action() -> None:
    native = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -1.061], dtype=np.float32)
    predictor = FakeChunkPredictor([chunk(first=native, fill=0.0)])
    buffer = FixedHActionBuffer(
        predictor,
        horizon=1,
        safety_enabled=False,
        replan_after_safety_violation=False,
    )

    release = buffer.next_action("official-observation")

    np.testing.assert_array_equal(release.action, native)
    assert release.telemetry["range_violation"] is True
    assert release.telemetry["range_clipped"] is False
    assert release.telemetry["buffer_discarded"] is False
    assert release.telemetry["planned_horizon"] == 1
    assert release.telemetry["actual_horizon"] == 1


def test_release_telemetry_is_complete_and_json_serializable() -> None:
    predictor = FakeChunkPredictor([chunk(fill=0.5)])
    buffer = FixedHActionBuffer(predictor)

    release = buffer.next_action({"observation": 1})

    assert {
        "chunk_origin",
        "planned_horizon",
        "actual_horizon",
        "buffer_size_before",
        "buffer_size_after",
        "model_invocation",
        "model_invoked",
        "range_violation",
        "clip_actions",
        "range_clipped",
        "buffer_discarded",
        "forced_horizon_next",
        "gripper_index",
        "gripper_negative_is_open",
        "gripper_positive_is_closed",
    } <= release.telemetry.keys()
    assert release.telemetry["gripper_index"] == 6
    assert json.loads(json.dumps(buffer.telemetry))[-1]["event"] == "action_release"
