from __future__ import annotations

import numpy as np
import pytest

from libero_platform.policies.base import (
    EpisodeContext,
    PolicyAdapter,
    PolicyRequest,
    validate_action,
)
from libero_platform.policies.zero_policy import ZeroPolicyAdapter


def request() -> PolicyRequest:
    return PolicyRequest(
        run_id="run_1",
        episode_id=0,
        step_id=1,
        instruction="move the object",
        images={"agentview": np.zeros((8, 8, 3), dtype=np.uint8)},
        proprioception=np.zeros(8, dtype=np.float32),
        previous_action=None,
    )


def test_zero_policy_returns_exact_valid_libero_gripper_close_action() -> None:
    response = ZeroPolicyAdapter("zero_policy").predict(request())

    assert response.action.dtype == np.float32
    assert response.action.shape == (7,)
    assert response.action.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert response.failure_type == ""


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(6),
        np.array([0, 0, 0, 0, 0, float("nan"), -1]),
        np.array([0, 0, 0, 0, 0, 0, 1.01]),
    ],
)
def test_validate_action_rejects_values_outside_the_libero_contract(
    value: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="action"):
        validate_action(value)


def test_validate_action_returns_a_float32_seven_dimension_vector() -> None:
    action = validate_action([0, 0, 0, 0, 0, 0, -1])

    assert action.dtype == np.float32
    assert action.shape == (7,)


def test_policy_adapter_exposes_episode_context_lifecycle_seam() -> None:
    class RecordingAdapter(PolicyAdapter):
        def __init__(self) -> None:
            self.context: EpisodeContext | None = None

        def begin_episode(self, context: EpisodeContext) -> None:
            self.context = context

        def predict(self, request: PolicyRequest):
            return ZeroPolicyAdapter("zero_policy").predict(request)

    context = EpisodeContext(
        suite="libero_spatial",
        task_id=1,
        task_name="put_the_bowl_on_the_plate",
        initial_state_id=2,
        seed=42,
    )
    adapter = RecordingAdapter()

    adapter.load()
    adapter.begin_episode(context)
    adapter.close()

    assert adapter.context == context
