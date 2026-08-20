from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from libero_platform.policies.base import EpisodeContext, PolicyRequest
from libero_platform.spec import SmolVLAInferenceSpec


def request(image_size: int = 8) -> PolicyRequest:
    return PolicyRequest(
        run_id="smolvla-run",
        episode_id=2,
        step_id=3,
        instruction="put the bowl on the plate",
        images={
            "agentview": np.zeros((image_size, image_size, 3), dtype=np.uint8),
            "wrist": np.ones((image_size, image_size, 3), dtype=np.uint8),
        },
        proprioception=np.arange(8, dtype=np.float32),
        previous_action=None,
    )


class FakeRuntime:
    device = "cuda:0"

    def __init__(
        self,
        action: object = None,
        load_error: Exception | None = None,
        predict_error: Exception | None = None,
    ) -> None:
        self.action = (
            np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 99.0],
                dtype=np.float32,
            )
            if action is None
            else action
        )
        self.load_error = load_error
        self.predict_error = predict_error
        self.loaded = False
        self.last_batch: dict[str, object] | None = None
        self.reset_calls = 0
        self.reset_seeds: list[int] = []

    def load(self) -> None:
        if self.load_error is not None:
            raise self.load_error
        self.loaded = True

    def predict(self, batch: dict[str, object]) -> object:
        self.last_batch = batch
        if self.predict_error is not None:
            raise self.predict_error
        return self.action

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)


def test_smolvla_runtime_applies_paper_simulation_inference_settings() -> None:
    module = importlib.import_module("libero_platform.policies.smolvla_policy")
    config = SimpleNamespace(chunk_size=50, n_action_steps=50, num_steps=30)

    module._apply_smolvla_inference_settings(
        config, SmolVLAInferenceSpec(n_action_steps=1, num_steps=10)
    )

    assert config.n_action_steps == 1
    assert config.num_steps == 10


def test_smolvla_runtime_rejects_action_steps_larger_than_checkpoint_chunk() -> None:
    module = importlib.import_module("libero_platform.policies.smolvla_policy")
    config = SimpleNamespace(chunk_size=4, n_action_steps=1, num_steps=10)

    with pytest.raises(ValueError, match="chunk_size"):
        module._apply_smolvla_inference_settings(
            config, SmolVLAInferenceSpec(n_action_steps=5, num_steps=10)
        )


def _adapter(runtime: FakeRuntime, *, action_control: object | None = None):
    module = importlib.import_module("libero_platform.policies.smolvla_policy")
    kwargs = {
        "model_key": "smolvla_libero",
        "checkpoint": "lerobot/smolvla_libero",
        "precision": "fp16",
        "runtime": runtime,
    }
    if action_control is not None:
        kwargs["action_control"] = action_control
    return module.SmolVLAPolicyAdapter(
        **kwargs,
    )


def test_smolvla_adapter_identity_tracks_configured_model_and_loaded_runtime() -> None:
    runtime = FakeRuntime()
    adapter = importlib.import_module(
        "libero_platform.policies.smolvla_policy"
    ).SmolVLAPolicyAdapter(
        model_key="smolvla_libero",
        checkpoint="HuggingFaceVLA/smolvla_libero",
        revision="0123456789abcdef",
        precision="fp16",
        runtime=runtime,
    )

    assert adapter.identity() == {
        "model_key": "smolvla_libero",
        "checkpoint": "HuggingFaceVLA/smolvla_libero",
        "revision": "0123456789abcdef",
        "precision": "fp16",
        "device": "cuda:0",
        "ready": False,
    }

    adapter.load()

    assert adapter.identity()["ready"] is True


def test_smolvla_adapter_maps_libero_request_and_takes_first_seven_values() -> None:
    runtime = FakeRuntime()
    adapter = _adapter(runtime)
    policy_request = request()

    adapter.load()
    response = adapter.predict(policy_request)

    assert response.failure_type == ""
    assert response.action.tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7]
    )
    assert response.inference_ms >= 0
    assert runtime.loaded is True
    assert runtime.last_batch is not None
    assert runtime.last_batch["task"] == policy_request.instruction
    mapped_images = runtime.last_batch["images"]
    assert isinstance(mapped_images, dict)
    assert set(mapped_images) == {"agentview", "wrist"}
    np.testing.assert_array_equal(mapped_images["agentview"], policy_request.images["agentview"])
    np.testing.assert_array_equal(mapped_images["wrist"], policy_request.images["wrist"])
    np.testing.assert_array_equal(
        runtime.last_batch["proprioception"], policy_request.proprioception
    )


def test_smolvla_adapter_preserves_the_official_eight_dimensional_libero_state() -> None:
    module = importlib.import_module('libero_platform.policies.smolvla_policy')
    preprocessor_input: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def permute(self, *dimensions: int) -> FakeTensor:
            del dimensions
            return self

        def contiguous(self) -> FakeTensor:
            return self

        def to(self, *args, **kwargs) -> FakeTensor:
            del args, kwargs
            return self

        def div(self, value: float) -> FakeTensor:
            del value
            return self

    class FakeContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args) -> None:
            del args

    class FakeTorch:
        float32 = object()

        class cuda:
            OutOfMemoryError = RuntimeError

        @staticmethod
        def from_numpy(value: np.ndarray) -> FakeTensor:
            return FakeTensor(value)

        @staticmethod
        def inference_mode() -> FakeContext:
            return FakeContext()

        @staticmethod
        def autocast(**kwargs) -> FakeContext:
            del kwargs
            return FakeContext()

    class FakePolicy:
        def select_action(self, *, batch: dict[str, object]) -> np.ndarray:
            del batch
            return np.zeros(7, dtype=np.float32)

    runtime = module.LeRobotSmolVLARuntime('checkpoint', 'fp32')
    runtime._torch = FakeTorch()
    runtime._policy = FakePolicy()
    runtime._preprocessor = lambda observation: preprocessor_input.setdefault(
        'observation', observation
    )
    runtime._postprocessor = lambda action: action
    runtime.device = 'cpu'

    policy_request = request()
    runtime.predict(
        {
            'task': policy_request.instruction,
            'images': dict(policy_request.images),
            'proprioception': policy_request.proprioception,
        }
    )

    state = preprocessor_input['observation']['observation.state']
    assert isinstance(state, FakeTensor)
    np.testing.assert_array_equal(
        state.value,
        np.arange(8, dtype=np.float32),
    )


def test_smolvla_adapter_resets_runtime_for_each_episode_start() -> None:
    runtime = FakeRuntime()
    adapter = _adapter(runtime)
    context = EpisodeContext('libero_spatial', 0, 'task', 0, 1)

    adapter.begin_episode(context)
    adapter.begin_episode(context)

    assert runtime.reset_calls == 2
    assert runtime.reset_seeds == [1, 1]


def test_smolvla_runtime_uses_episode_seed_for_explicit_sampling_noise() -> None:
    torch = pytest.importorskip("torch")
    module = importlib.import_module("libero_platform.policies.smolvla_policy")

    class FakePolicy:
        config = SimpleNamespace(chunk_size=4, max_action_dim=7)

        def __init__(self) -> None:
            self.noises = []

        def reset(self) -> None:
            pass

        def select_action(self, *, batch: dict[str, object], noise) -> object:
            del batch
            self.noises.append(noise.detach().clone())
            return noise[:, 0, :7]

    policy = FakePolicy()
    runtime = module.LeRobotSmolVLARuntime("checkpoint", "fp32")
    runtime._torch = torch
    runtime._policy = policy
    runtime._preprocessor = lambda observation: {
        key: value.unsqueeze(0) if hasattr(value, "unsqueeze") else value
        for key, value in observation.items()
    }
    runtime._postprocessor = lambda action: action
    runtime.device = "cpu"
    batch = {
        "task": request().instruction,
        "images": dict(request().images),
        "proprioception": request().proprioception,
    }

    runtime.reset(47)
    first = runtime.predict(batch)
    runtime.reset(47)
    repeated = runtime.predict(batch)
    runtime.reset(48)
    different = runtime.predict(batch)

    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, different)
    assert tuple(policy.noises[0].shape) == (1, 4, 7)


def test_smolvla_runtime_preserves_official_eight_dimensional_normalizer_stats() -> None:
    module = importlib.import_module('libero_platform.policies.smolvla_policy')
    normalizer = SimpleNamespace(
        stats={
            'observation.state': {
                'count': 1.0,
                'mean': np.arange(8, dtype=np.float32),
                'std': np.arange(10, 18, dtype=np.float32),
            }
        }
    )
    preprocessor = SimpleNamespace(steps=[normalizer])

    module._validate_smolvla_preprocessor_state(preprocessor)

    np.testing.assert_array_equal(
        normalizer.stats['observation.state']['mean'],
        np.arange(8, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        normalizer.stats['observation.state']['std'],
        np.arange(10, 18, dtype=np.float32),
    )


def test_smolvla_runtime_keeps_loaded_lerobot_normalizer_tensor_stats_at_eight_dims(
    tmp_path,
) -> None:
    pytest.importorskip('lerobot', minversion='0.6.1')
    torch = pytest.importorskip('torch')
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import NormalizerProcessorStep
    from lerobot.processor.pipeline import (
        DataProcessorPipeline,
        batch_to_transition,
        transition_to_batch,
    )

    normalizer = NormalizerProcessorStep(
        features={
            'observation.state': PolicyFeature(
                type=FeatureType.STATE,
                shape=(8,),
            )
        },
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={
            'observation.state': {
                'mean': np.arange(8, dtype=np.float32),
                'std': np.ones(8, dtype=np.float32),
            }
        },
    )
    config_filename = 'policy_preprocessor.json'
    DataProcessorPipeline(
        steps=[normalizer],
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    ).save_pretrained(tmp_path, config_filename=config_filename)
    preprocessor = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=config_filename,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    loaded_normalizer = preprocessor.steps[0]
    assert tuple(
        loaded_normalizer._tensor_stats['observation.state']['mean'].shape
    ) == (8,)

    module = importlib.import_module('libero_platform.policies.smolvla_policy')
    module._validate_smolvla_preprocessor_state(preprocessor)
    normalized = preprocessor(
        {'observation.state': torch.arange(8, dtype=torch.float32)}
    )

    assert tuple(
        loaded_normalizer._tensor_stats['observation.state']['mean'].shape
    ) == (8,)
    torch.testing.assert_close(
        normalized['observation.state'],
        torch.zeros(8, dtype=torch.float32),
    )


def test_smolvla_module_import_keeps_lerobot_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "lerobot" or name.startswith("lerobot."):
            raise AssertionError("LeRobot must not import before runtime.load()")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("libero_platform.policies.smolvla_policy", None)

    module = importlib.import_module("libero_platform.policies.smolvla_policy")

    assert module.SmolVLAPolicyAdapter.__name__ == "SmolVLAPolicyAdapter"


def test_smolvla_load_status_flushes_a_clear_runtime_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("libero_platform.policies.smolvla_policy")
    messages = []

    monkeypatch.setattr(
        builtins,
        "print",
        lambda *args, **kwargs: messages.append((args, kwargs)),
    )

    module._load_status("loading checkpoint configuration")

    assert messages == [
        (("SmolVLA load: loading checkpoint configuration",), {"flush": True})
    ]


def test_smolvla_runtime_loads_checkpoint_weights_on_cpu_before_gpu_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("libero_platform.policies.smolvla_policy")
    config = SimpleNamespace(
        type="smolvla", device="cuda", chunk_size=50, n_action_steps=50, num_steps=30
    )
    seen: dict[str, object] = {}

    class FakePolicy:
        def to(self, device: str) -> FakePolicy:
            seen.setdefault("moves", []).append(device)
            return self

    class FakePolicyClass:
        @classmethod
        def from_pretrained(
            cls, checkpoint: str, *, config: object, revision: str | None = None
        ) -> FakePolicy:
            del cls, checkpoint, revision
            seen["load_device"] = config.device
            seen["n_action_steps"] = config.n_action_steps
            return FakePolicy()

    class FakeTorch:
        float16 = object()
        bfloat16 = object()

        class cuda:
            OutOfMemoryError = RuntimeError

            @staticmethod
            def is_available() -> bool:
                return True

    factory = SimpleNamespace(
        get_policy_class=lambda policy_type: FakePolicyClass,
        make_pre_post_processors=lambda **kwargs: (SimpleNamespace(steps=[]), object()),
    )
    policy_config = SimpleNamespace(
        PreTrainedConfig=SimpleNamespace(
            from_pretrained=lambda checkpoint, *, revision=None: config
        )
    )
    smolvla_config = SimpleNamespace(SmolVLAConfig=object)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", policy_config)
    monkeypatch.setitem(
        sys.modules,
        "lerobot.policies.smolvla.configuration_smolvla",
        smolvla_config,
    )

    runtime = module.LeRobotSmolVLARuntime("checkpoint", "fp16")
    runtime.load()

    assert seen["load_device"] == "cpu"
    assert seen["n_action_steps"] == 1
    assert config.num_steps == 10
    assert seen["moves"] == ["cuda"]


def test_smolvla_runtime_loads_all_hub_artifacts_at_requested_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("libero_platform.policies.smolvla_policy")
    config = SimpleNamespace(
        type="smolvla", device="cuda", chunk_size=50, n_action_steps=50, num_steps=30
    )
    revisions: dict[str, str | None] = {}

    class FakePolicy:
        def to(self, device: str) -> FakePolicy:
            del device
            return self

    class FakePolicyClass:
        @classmethod
        def from_pretrained(
            cls, checkpoint: str, *, config: object, revision: str | None = None
        ) -> FakePolicy:
            del cls, checkpoint, config
            revisions["policy"] = revision
            return FakePolicy()

    class FakeTorch:
        float16 = object()
        bfloat16 = object()

        class cuda:
            OutOfMemoryError = RuntimeError

            @staticmethod
            def is_available() -> bool:
                return True

    factory = SimpleNamespace(
        get_policy_class=lambda policy_type: FakePolicyClass,
        make_pre_post_processors=lambda **kwargs: (
            revisions.__setitem__("preprocessor", kwargs.get("revision"))
            or (SimpleNamespace(steps=[]), object())
        ),
    )
    policy_config = SimpleNamespace(
        PreTrainedConfig=SimpleNamespace(
            from_pretrained=lambda checkpoint, *, revision=None: (
                revisions.__setitem__("config", revision) or config
            )
        )
    )
    smolvla_config = SimpleNamespace(SmolVLAConfig=object)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", policy_config)
    monkeypatch.setitem(
        sys.modules,
        "lerobot.policies.smolvla.configuration_smolvla",
        smolvla_config,
    )

    runtime = module.LeRobotSmolVLARuntime("checkpoint", "fp16", revision="0123456789abcdef")
    runtime.load()

    assert revisions == {
        "config": "0123456789abcdef",
        "policy": "0123456789abcdef",
        "preprocessor": "0123456789abcdef",
    }


def test_smolvla_adapter_classifies_model_load_failures() -> None:
    runtime = FakeRuntime(load_error=OSError("checkpoint unavailable"))
    adapter = _adapter(runtime)

    with pytest.raises(Exception) as error:
        adapter.load()

    assert error.value.failure_type == "model_load_error"


def test_smolvla_adapter_classifies_cuda_oom_during_prediction() -> None:
    module = importlib.import_module("libero_platform.policies.smolvla_policy")
    runtime = FakeRuntime(
        predict_error=module.SmolVLAPolicyRuntimeError(
            "oom", "CUDA out of memory"
        )
    )
    adapter = _adapter(runtime)
    adapter.load()

    response = adapter.predict(request())

    assert response.failure_type == "oom"
    assert response.action.shape == (7,)


def test_smolvla_adapter_rejects_malformed_postprocessed_action() -> None:
    runtime = FakeRuntime(action=np.zeros(6, dtype=np.float32))
    adapter = _adapter(runtime)
    adapter.load()

    response = adapter.predict(request())

    assert response.failure_type == "invalid_action"
    assert response.action.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_smolvla_adapter_saturates_finite_control_actions() -> None:
    runtime = FakeRuntime(action=np.full(7, 1.5, dtype=np.float32))
    adapter = _adapter(runtime)
    adapter.load()

    response = adapter.predict(request())

    assert response.failure_type == ""
    np.testing.assert_allclose(response.action, np.ones(7, dtype=np.float32))


def test_smolvla_adapter_saturates_small_control_bound_overshoot() -> None:
    runtime = FakeRuntime(
        action=np.array([0.2, -1.00796, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    )
    adapter = _adapter(runtime)
    adapter.load()

    response = adapter.predict(request())

    assert response.failure_type == ""
    np.testing.assert_allclose(response.action, [0.2, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0])


def test_smolvla_adapter_scales_named_control_axes_and_preserves_raw_action() -> None:
    spec_module = importlib.import_module("libero_platform.spec")
    runtime = FakeRuntime(
        action=np.array([0.8, 0.4, -0.8, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    )
    adapter = _adapter(
        runtime,
        action_control=spec_module.ActionControlSpec(
            mode="scaled",
            translation_scale=0.5,
            rotation_scale=0.25,
        ),
    )
    adapter.load()

    response = adapter.predict(request())

    assert response.failure_type == ""
    np.testing.assert_allclose(
        response.raw_action,
        np.array([0.8, 0.4, -0.8, 0.5, -0.5, 1.0, -1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        response.action,
        np.array([0.4, 0.2, -0.4, 0.125, -0.125, 0.25, -1.0], dtype=np.float32),
    )
    assert response.action_clipped is False
    assert "scaled" in response.action_transform
    assert "clip[-1,1]" in response.action_transform


def test_cli_builds_smolvla_for_libero_without_marking_it_unavailable() -> None:
    from libero_platform import cli

    spec = SimpleNamespace(
        policy_adapter="smolvla",
        policy=SimpleNamespace(key="smolvla_libero", precision="fp16"),
        resolved_checkpoint="lerobot/smolvla_libero",
        resolved_revision="pinned-model-revision",
        benchmark=SimpleNamespace(backend="libero"),
    )

    policy = cli._build_policy(spec)

    assert policy.__class__.__name__ == "SmolVLAPolicyAdapter"
    assert policy.identity()["revision"] == "pinned-model-revision"
    assert cli._unavailable_category(spec) is None


@pytest.mark.model
def test_smolvla_model_smoke_returns_a_finite_libero_action() -> None:
    module = importlib.import_module("libero_platform.policies.smolvla_policy")
    adapter = module.SmolVLAPolicyAdapter(
        model_key="smolvla_libero",
        checkpoint="lerobot/smolvla_libero",
        precision="fp16",
    )

    adapter.load()
    response = adapter.predict(request(image_size=256))

    assert response.failure_type == "", response.error
    assert response.action.shape == (7,)
    assert np.isfinite(response.action).all()
