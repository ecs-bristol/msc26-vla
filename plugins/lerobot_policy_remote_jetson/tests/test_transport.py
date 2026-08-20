import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot_policy_remote_jetson.transport import (
    decode_action_response,
    observation_to_request,
)


def test_observation_to_request_preserves_official_lerobot_inputs():
    observation = {
        "observation.images.image": torch.zeros((1, 3, 8, 6), dtype=torch.float32),
        "observation.images.image2": torch.ones((1, 3, 8, 6), dtype=torch.float32),
        "observation.state": torch.arange(8, dtype=torch.float32).reshape(1, 8),
        "task": ["pick up the black bowl"],
    }

    request = observation_to_request(
        observation,
        run_id="official-eval",
        episode_id=3,
        step_id=7,
        previous_action=np.zeros(7, dtype=np.float32),
    )

    assert request["instruction"] == "pick up the black bowl"
    assert request["proprioception"] == pytest.approx(list(range(8)))
    assert request["previous_action"] == pytest.approx([0.0] * 7)
    assert set(request["images"]) == {"agentview", "wrist"}

    decoded = Image.open(io.BytesIO(base64.b64decode(request["images"]["wrist"])))
    assert decoded.size == (6, 8)
    assert np.asarray(decoded).mean() == 255


def test_decode_action_response_requires_one_finite_normalized_action():
    action = decode_action_response({"action": [0.0, -1.0, 1.0, 0.25, 0.0, 0.0, 0.5]})
    assert action.shape == (7,)
    assert action.dtype == np.float32

    with pytest.raises(ValueError, match="7 values"):
        decode_action_response({"action": [0.0] * 6})
    with pytest.raises(ValueError, match="finite"):
        decode_action_response({"action": [0.0] * 6 + [float("nan")]})
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        decode_action_response({"action": [0.0] * 6 + [1.01]})
