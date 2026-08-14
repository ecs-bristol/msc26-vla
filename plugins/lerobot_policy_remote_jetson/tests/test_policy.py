import numpy as np
import torch

from lerobot_policy_remote_jetson.configuration_remote_jetson import RemoteJetsonConfig
from lerobot_policy_remote_jetson.modeling_remote_jetson import RemoteJetsonPolicy
from lerobot_policy_remote_jetson.processor_remote_jetson import (
    make_remote_jetson_pre_post_processors,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append((url, json, timeout))
        if url.endswith("/reset"):
            return FakeResponse({"status": "ok"})
        return FakeResponse({"action": [0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0]})


def _observation():
    return {
        "observation.images.image": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        "observation.images.image2": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        "observation.state": torch.zeros((1, 8), dtype=torch.float32),
        "task": ["pick up the black bowl"],
    }


def test_policy_reset_and_select_action_use_remote_service():
    session = FakeSession()
    config = RemoteJetsonConfig(endpoint="http://10.42.0.2:8081", device="cpu")
    policy = RemoteJetsonPolicy(config, session=session)

    policy.reset()
    action = policy.select_action(_observation())

    assert session.calls[0][0] == "http://10.42.0.2:8081/reset"
    assert session.calls[1][0] == "http://10.42.0.2:8081/predict"
    assert action.shape == (1, 7)
    assert action.device.type == "cpu"
    assert np.allclose(action.numpy(), [[0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0]])


def test_policy_processors_are_identity_pipelines():
    config = RemoteJetsonConfig(device="cpu")
    preprocessor, postprocessor = make_remote_jetson_pre_post_processors(config)
    observation = _observation()
    action = torch.zeros((1, 7), dtype=torch.float32)

    processed = preprocessor(observation)
    restored = postprocessor(action)

    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(processed[key], value)
        else:
            assert processed[key] == value
    assert restored is action
