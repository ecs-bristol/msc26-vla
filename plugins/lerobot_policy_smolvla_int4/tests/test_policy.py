from types import SimpleNamespace

import pytest
import torch

from lerobot_policy_smolvla_int4.configuration_smolvla_int4 import SmolVLAInt4Config
from lerobot_policy_smolvla_int4.modeling_smolvla_int4 import SmolVLAInt4Policy


class FakeInner:
    config = SimpleNamespace(use_amp=False, num_steps=10)

    def reset(self):
        return None

    def select_action(self, batch, noise=None, **kwargs):
        return torch.zeros(1, 7)

    def predict_action_chunk(self, batch, noise=None, **kwargs):
        return torch.zeros(1, 1, 7)


def _make_policy(quant_method="none", inner=None):
    config = SmolVLAInt4Config(quant_method=quant_method)
    return SmolVLAInt4Policy(config, inner_policy=inner or FakeInner())


def test_policy_delegates_select_action_to_inner():
    policy = _make_policy()

    action = policy.select_action({})

    assert action.shape == (1, 7)


def test_policy_delegates_reset_to_inner():
    policy = _make_policy()

    policy.reset()  # must not raise


def test_policy_mirrors_inner_amp_setting():
    policy = _make_policy()

    assert policy.config.use_amp is False


def test_policy_forward_and_optim_params_are_inference_only():
    policy = _make_policy()

    with pytest.raises(RuntimeError, match="inference-only"):
        policy.get_optim_params()
    with pytest.raises(RuntimeError, match="inference-only"):
        policy.forward({})


def test_find_vlm_module_locates_vlm_with_expert_vlm():
    class VLM:
        pass

    class VlmWithExpert:
        def __init__(self):
            self.vlm = VLM()

    class Model:
        def __init__(self):
            self.vlm_with_expert = VlmWithExpert()

    class Inner:
        def __init__(self):
            self.model = Model()

    policy = _make_policy(inner=Inner())

    assert policy._find_vlm_module().__class__ is VLM
