from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_policy_config
from libero_platform.policies.fixed_h_action_buffer import InvalidPolicyActionError

import lerobot_policy_smolvla_adaptive  # noqa: F401
from lerobot_policy_smolvla_adaptive.configuration_smolvla_adaptive import (
    SmolVLAAdaptiveConfig,
)
from lerobot_policy_smolvla_adaptive.modeling_smolvla_adaptive import (
    SmolVLAAdaptivePolicy,
    _load_local_tokenizer,
    _runtime_preprocessor_overrides,
)
from lerobot_policy_smolvla_adaptive.processor_smolvla_adaptive import (
    make_smolvla_adaptive_pre_post_processors,
)


def _chunk(*, first: np.ndarray | None = None, fill: float = 0.0) -> torch.Tensor:
    value = torch.full((1, 50, 7), fill, dtype=torch.float32)
    if first is not None:
        value[0, 0] = torch.from_numpy(first)
    return value


class FakeBasePolicy(torch.nn.Module):
    def __init__(self, chunks: list[object]) -> None:
        super().__init__()
        self._chunks = list(chunks)
        self.predict_calls: list[dict[str, object]] = []
        self.select_calls = 0
        self.reset_calls = 0

    def predict_action_chunk(self, batch: dict[str, object]) -> object:
        self.predict_calls.append(batch)
        return self._chunks.pop(0)

    def select_action(self, batch: dict[str, object]) -> object:
        del batch
        self.select_calls += 1
        raise AssertionError("adaptive wrapper must not call base.select_action")

    def reset(self) -> None:
        self.reset_calls += 1


class RecordingProcessor:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def __call__(self, value: object) -> object:
        self.calls.append(value)
        return value


def _policy(chunks: list[object]):
    base = FakeBasePolicy(chunks)
    preprocessor = RecordingProcessor()
    postprocessor = RecordingProcessor()
    policy = SmolVLAAdaptivePolicy(
        SmolVLAAdaptiveConfig(),
        base_policy=base,
        base_preprocessor=preprocessor,
        base_postprocessor=postprocessor,
    )
    return policy, base, preprocessor, postprocessor


def test_plugin_registers_config_and_policy_with_lerobot_factory() -> None:
    config = make_policy_config("smolvla_adaptive")

    assert PreTrainedConfig.get_choice_class("smolvla_adaptive") is SmolVLAAdaptiveConfig
    assert get_policy_class("smolvla_adaptive") is SmolVLAAdaptivePolicy
    assert config.base_checkpoint == "HuggingFaceVLA/smolvla_libero"
    assert config.base_revision == "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    assert config.fixed_h == 20
    assert config.num_steps == 2
    assert config.local_files_only is True
    assert config.vlm_checkpoint == "HuggingFaceTB/SmolVLM2-500M-Instruct"
    assert config.vlm_revision == "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def test_factory_parses_frozen_native_and_adaptive_policy_arguments() -> None:
    native = make_policy_config("smolvla", n_action_steps=20, num_steps=2, chunk_size=50)
    adaptive = make_policy_config("smolvla_adaptive", fixed_h=20, num_steps=2)

    assert native.n_action_steps == 20
    assert native.num_steps == 2
    assert native.chunk_size == 50
    assert adaptive.fixed_h == 20
    assert adaptive.num_steps == 2


def test_plugin_fixes_remote_code_to_false_without_a_cli_config_field() -> None:
    from lerobot_policy_smolvla_adaptive import modeling_smolvla_adaptive

    assert modeling_smolvla_adaptive._TRUST_REMOTE_CODE is False
    assert "trust_remote_code" not in SmolVLAAdaptiveConfig.__dataclass_fields__


def test_frozen_tokenizer_loader_uses_only_absolute_local_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import transformers

    snapshot = tmp_path / "frozen-vlm"
    snapshot.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []
    sentinel = object()

    def fake_from_pretrained(source: str, **kwargs: object) -> object:
        calls.append((source, kwargs))
        return sentinel

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    tokenizer = _load_local_tokenizer(str(snapshot))
    overrides = _runtime_preprocessor_overrides(tokenizer=tokenizer, device="cpu")

    assert tokenizer is sentinel
    assert calls == [
        (
            str(snapshot.resolve()),
            {"local_files_only": True, "trust_remote_code": False},
        )
    ]
    assert overrides == {
        "tokenizer_processor": {"tokenizer": sentinel, "tokenizer_name": None},
        "device_processor": {"device": "cpu"},
    }


def test_cpu_processor_override_uses_cpu_without_touching_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lerobot.processor.device_processor import DeviceProcessorStep

    sentinel = object()
    overrides = _runtime_preprocessor_overrides(tokenizer=sentinel, device="cpu")

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CPU construction must not query CUDA")),
    )
    processor = DeviceProcessorStep(**overrides["device_processor"])

    assert processor.tensor_device.type == "cpu"


def test_cuda_processor_override_preserves_formal_cuda_path() -> None:
    overrides = _runtime_preprocessor_overrides(tokenizer=object(), device="cuda")

    assert overrides["device_processor"] == {"device": "cuda"}


def test_processor_runtime_overrides_do_not_mutate_checkpoint_json_shape() -> None:
    frozen_checkpoint_config = {
        "steps": [
            {"registry_name": "tokenizer_processor", "config": {"tokenizer_name": "Hub/ID"}},
            {"registry_name": "device_processor", "config": {"device": "cuda"}},
        ]
    }
    original = deepcopy(frozen_checkpoint_config)

    _runtime_preprocessor_overrides(tokenizer=object(), device="cpu")

    assert frozen_checkpoint_config == original


def test_plugin_processor_factory_is_external_identity_pipeline() -> None:
    preprocessor, postprocessor = make_smolvla_adaptive_pre_post_processors(
        SmolVLAAdaptiveConfig()
    )
    observation = {"observation.state": torch.zeros((1, 8)), "task": ["task"]}
    action = torch.zeros((1, 7))

    processed = preprocessor(observation)
    restored = postprocessor(action)

    assert torch.equal(processed["observation.state"], observation["observation.state"])
    assert processed["task"] == observation["task"]
    assert restored is action


def test_plugin_uses_base_chunk_path_and_fixed_h_twenty() -> None:
    policy, base, preprocessor, postprocessor = _policy([_chunk(fill=0.2), _chunk(fill=-0.2)])

    releases = [policy.select_action({"step": index}) for index in range(21)]

    assert len(base.predict_calls) == 2
    assert base.select_calls == 0
    assert len(preprocessor.calls) == 2
    assert len(postprocessor.calls) == 2
    assert all(tuple(action.shape) == (1, 7) for action in releases)
    assert torch.allclose(releases[0], torch.full((1, 7), 0.2))
    assert torch.allclose(releases[-1], torch.full((1, 7), -0.2))
    assert policy.action_telemetry[-1]["model_invocation"] == 2


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (_chunk(first=np.array([np.nan, 0, 0, 0, 0, 0, 0], dtype=np.float32)), "finite"),
        (torch.zeros((1, 50, 6)), "shape"),
    ],
)
def test_plugin_rejects_invalid_actions_and_clears_buffer(invalid: object, message: str) -> None:
    policy, base, _, _ = _policy([invalid, _chunk(fill=0.3)])

    with pytest.raises(InvalidPolicyActionError, match=message):
        policy.select_action({"step": 0})

    assert policy._buffer.buffered_actions == 0
    recovered = policy.select_action({"step": 1})
    assert torch.allclose(recovered, torch.full((1, 7), 0.3))
    assert base.select_calls == 0


def test_plugin_clips_then_forces_one_step_refill_and_reset_clears_both_layers() -> None:
    too_large = np.array([2.0, 0, 0, 0, 0, 0, -2.0], dtype=np.float32)
    policy, base, _, _ = _policy(
        [_chunk(first=too_large, fill=0.4), _chunk(fill=-0.4), _chunk(fill=0.6)]
    )

    clipped = policy.select_action({"step": 0})
    recovery = policy.select_action({"step": 1})
    policy.reset()
    after_reset = policy.select_action({"step": 2})

    assert torch.allclose(clipped, torch.tensor([[1, 0, 0, 0, 0, 0, -1]], dtype=torch.float32))
    assert policy.action_telemetry[1]["range_clipped"] is True
    assert recovery.shape == (1, 7)
    assert base.predict_calls[1] == {"step": 1}
    assert base.reset_calls == 1
    assert torch.allclose(after_reset, torch.full((1, 7), 0.6))
    assert json.loads(json.dumps(policy.action_telemetry))[-1]["event"] == "action_release"
