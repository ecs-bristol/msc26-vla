from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Protocol

import numpy as np

from libero_platform.policies.base import (
    EpisodeContext,
    PolicyAdapter,
    PolicyRequest,
    PolicyResponse,
    validate_action,
    validate_action_chunk,
)
from libero_platform.spec import ActionControlSpec, SmolVLAInferenceSpec

_FAILURE_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def _load_status(message: str) -> None:
    print(f"SmolVLA load: {message}", flush=True)


def _apply_smolvla_inference_settings(
    config: object, settings: SmolVLAInferenceSpec
) -> None:
    chunk_size = int(getattr(config, "chunk_size"))
    if settings.chunk_size is not None:
        chunk_size = int(settings.chunk_size)
        config.chunk_size = chunk_size
    if settings.n_action_steps > chunk_size:
        raise ValueError(
            "SmolVLA n_action_steps must not exceed checkpoint chunk_size"
        )
    config.n_action_steps = settings.n_action_steps
    config.num_steps = settings.num_steps


class SmolVLAPolicyRuntimeError(RuntimeError):
    def __init__(self, failure_type: str, error: str) -> None:
        super().__init__(error)
        self.failure_type = failure_type


class SmolVLARuntime(Protocol):
    device: str

    def load(self) -> None: ...

    def predict(self, batch: dict[str, object]) -> object: ...

    def reset(self, seed: int) -> None: ...


class SmolVLAPolicyAdapter(PolicyAdapter):
    def __init__(
        self,
        model_key: str,
        checkpoint: str,
        precision: str,
        revision: str | None = None,
        action_control: ActionControlSpec | None = None,
        smolvla_inference: SmolVLAInferenceSpec | None = None,
        runtime: SmolVLARuntime | None = None,
        quant_method: str = "none",
        quant_scope: str = "language",
        vision_bits: int = 4,
        connector_bits: int = 8,
        text_bits: int = 8,
    ) -> None:
        self._model_key = model_key
        self._checkpoint = checkpoint
        self._revision = revision
        self._precision = precision
        self._action_control = action_control or ActionControlSpec()
        self._smolvla_inference = smolvla_inference or SmolVLAInferenceSpec()
        self._runtime = runtime or LeRobotSmolVLARuntime(
            checkpoint,
            precision,
            smolvla_inference=smolvla_inference,
            revision=revision,
            quant_method=quant_method,
            quant_scope=quant_scope,
            vision_bits=vision_bits,
            connector_bits=connector_bits,
            text_bits=text_bits,
        )
        self._loaded = False

    def identity(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "checkpoint": self._checkpoint,
            "revision": self._revision,
            "precision": self._precision,
            "device": self._runtime.device,
            "num_steps": self._smolvla_inference.num_steps,
            "n_action_steps": self._smolvla_inference.n_action_steps,
            "chunk_size": self._smolvla_inference.chunk_size,
            "ready": self._loaded,
        }

    def load(self) -> None:
        try:
            self._runtime.load()
        except SmolVLAPolicyRuntimeError:
            raise
        except Exception as exc:
            raise SmolVLAPolicyRuntimeError("model_load_error", str(exc)) from exc
        self._loaded = True

    def begin_episode(self, context: EpisodeContext) -> None:
        self._runtime.reset(context.seed)

    def predict(self, request: PolicyRequest) -> PolicyResponse:
        started_at = time.perf_counter()
        if not self._loaded:
            return self._failure("model_load_error", "SmolVLA policy is not loaded", started_at)

        action_values: np.ndarray | None = None
        scaled_chunk: np.ndarray | None = None
        transformed_chunk: np.ndarray | None = None
        transformed_action: np.ndarray | None = None
        try:
            postprocessed = self._runtime.predict(
                {
                    "task": request.instruction,
                    "images": dict(request.images),
                    "proprioception": np.asarray(request.proprioception, dtype=np.float32),
                }
            )
            action_chunk_values = _action_chunk_values(
                postprocessed, self._smolvla_inference.n_action_steps
            )
            action_values = action_chunk_values[0]
            scaled_chunk = _scale_control_action(
                action_chunk_values, self._action_control
            )
            transformed_chunk = _transform_control_action(
                action_chunk_values, self._action_control
            )
            transformed_action = transformed_chunk[0]
            action = validate_action(transformed_action)
            action_chunk = validate_action_chunk(transformed_chunk)
        except SmolVLAPolicyRuntimeError as exc:
            return self._failure(exc.failure_type, str(exc), started_at)
        except (TypeError, ValueError, OverflowError) as exc:
            return self._failure(
                "invalid_action",
                _action_validation_error(action_values, exc),
                started_at,
            )

        return PolicyResponse(
            action=action,
            inference_ms=_elapsed_ms(started_at),
            model_key=self._model_key,
            device=self._runtime.device,
            raw_action=action_values.copy(),
            action_chunk=action_chunk,
            action_transform=_action_transform_metadata(self._action_control),
            action_clipped=_action_clipped(scaled_chunk, transformed_chunk),
        )

    def _failure(
        self, failure_type: str, error: str, started_at: float
    ) -> PolicyResponse:
        return PolicyResponse(
            action=_FAILURE_ACTION.copy(),
            inference_ms=_elapsed_ms(started_at),
            model_key=self._model_key,
            device=self._runtime.device,
            failure_type=failure_type,
            error=error,
        )


class LeRobotSmolVLARuntime:
    def __init__(
        self,
        checkpoint: str,
        precision: str,
        smolvla_inference: SmolVLAInferenceSpec | None = None,
        revision: str | None = None,
        quant_method: str = "none",
        quant_scope: str = "language",
        vision_bits: int = 4,
        connector_bits: int = 8,
        text_bits: int = 8,
    ) -> None:
        self._checkpoint = checkpoint
        self._precision = precision
        self._revision = revision
        self._smolvla_inference = smolvla_inference or SmolVLAInferenceSpec()
        self._quant_method = quant_method
        self._quant_scope = quant_scope
        self._vision_bits = vision_bits
        self._connector_bits = connector_bits
        self._text_bits = text_bits
        self.device = "unavailable"
        self._torch = None
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._inference_dtype = None
        self._noise_generator = None

    def load(self) -> None:
        try:
            import torch
        except Exception as exc:
            raise SmolVLAPolicyRuntimeError("model_load_error", str(exc)) from exc

        try:
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

            del SmolVLAConfig  # Import registers the config before deserialization.
            _load_status("loading checkpoint configuration")
            if self._quant_method == "none":
                config = PreTrainedConfig.from_pretrained(
                    self._checkpoint, revision=self._revision
                )
                _apply_smolvla_inference_settings(config, self._smolvla_inference)
            else:
                from lerobot_policy_smolvla_int4.configuration_smolvla_int4 import (
                    SmolVLAInt4Config,
                )

                config = SmolVLAInt4Config(
                    checkpoint=self._checkpoint,
                    revision=self._revision,
                    quant_method=self._quant_method,
                    quant_scope=self._quant_scope,
                    vision_bits=self._vision_bits,
                    connector_bits=self._connector_bits,
                    text_bits=self._text_bits,
                    num_steps=self._smolvla_inference.num_steps,
                    n_action_steps=self._smolvla_inference.n_action_steps,
                    device="cpu",
                )
            _load_status(
                "simulation inference: "
                f"action_steps={config.n_action_steps}, flow_steps={config.num_steps}"
            )
            if self._quant_method == "none":
                # SmolVLA's nested VLM loader must materialize cached safetensors on CPU first.
                config.device = "cpu"
                policy_class = get_policy_class(config.type)
                _load_status("loading policy weights")
                policy = policy_class.from_pretrained(
                    self._checkpoint, config=config, revision=self._revision
                )
            else:
                from lerobot_policy_smolvla_int4.modeling_smolvla_int4 import (
                    SmolVLAInt4Policy,
                )

                config.device = "cpu"
                _load_status("loading quantized policy weights")
                policy = SmolVLAInt4Policy(config)
            _load_status("loading preprocessors")
            if self._quant_method == "none":
                preprocessor, postprocessor = make_pre_post_processors(
                    policy_cfg=config,
                    pretrained_path=self._checkpoint,
                    revision=self._revision,
                )
            else:
                from lerobot_policy_smolvla_int4.processor_smolvla_int4 import (
                    make_smolvla_int4_pre_post_processors,
                )

                preprocessor, postprocessor = (
                    make_smolvla_int4_pre_post_processors(config)
                )
            _validate_smolvla_preprocessor_state(preprocessor)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            _load_status(f"moving policy to {self.device}")
            policy.to(self.device)
            inference_dtype = None
            if self.device == "cuda" and self._precision == "fp16":
                inference_dtype = torch.float16
            elif self.device == "cuda" and self._precision == "bf16":
                inference_dtype = torch.bfloat16
            elif self._precision not in {"fp32", "fp16", "bf16"}:
                raise ValueError(f"unsupported SmolVLA precision: {self._precision}")
        except torch.cuda.OutOfMemoryError as exc:
            raise SmolVLAPolicyRuntimeError("oom", str(exc)) from exc
        except SmolVLAPolicyRuntimeError:
            raise
        except Exception as exc:
            raise SmolVLAPolicyRuntimeError("model_load_error", str(exc)) from exc

        self._torch = torch
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._inference_dtype = inference_dtype
        _load_status(f"ready on {self.device} ({self._precision})")

    def reset(self, seed: int) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self._policy is not None:
            self._policy.reset()
        if self._torch is not None:
            self._noise_generator = self._torch.Generator(device=self.device)
            self._noise_generator.manual_seed(seed)

    def predict(self, batch: dict[str, object]) -> np.ndarray:
        if self._torch is None or self._policy is None:
            raise SmolVLAPolicyRuntimeError("model_load_error", "SmolVLA runtime is not loaded")
        if self._preprocessor is None or self._postprocessor is None:
            raise SmolVLAPolicyRuntimeError("model_load_error", "SmolVLA processors are not loaded")

        try:
            observation = _libero_observation(batch, self._torch)
            normalized = self._preprocessor(observation)
            normalized = {
                key: value.to(self.device)
                if hasattr(value, "to")
                else value
                for key, value in normalized.items()
            }
            with self._torch.inference_mode():
                with self._torch.autocast(
                    device_type=self.device,
                    dtype=self._inference_dtype,
                    enabled=self._inference_dtype is not None,
                ):
                    noise = self._sampling_noise(normalized)
                    if noise is None:
                        action = self._policy.predict_action_chunk(batch=normalized)
                    else:
                        action = self._policy.predict_action_chunk(
                            batch=normalized,
                            noise=noise,
                        )
            return self._postprocessor(action)
        except self._torch.cuda.OutOfMemoryError as exc:
            raise SmolVLAPolicyRuntimeError("oom", str(exc)) from exc
        except SmolVLAPolicyRuntimeError:
            raise
        except Exception as exc:
            raise SmolVLAPolicyRuntimeError("invalid_action", str(exc)) from exc

    def _sampling_noise(self, batch: dict[str, object]):
        if self._noise_generator is None or self._torch is None or self._policy is None:
            return None
        state = batch.get("observation.state")
        shape = getattr(state, "shape", ())
        if not shape:
            raise SmolVLAPolicyRuntimeError(
                "invalid_action", "normalized SmolVLA state must have a batch dimension"
            )
        batch_size = 1 if len(shape) == 1 else int(shape[0])
        config = self._policy.config
        inner_policy = getattr(self._policy, "inner", None)
        if inner_policy is not None and getattr(inner_policy, "config", None) is not None:
            config = inner_policy.config
        noise_shape = (
            batch_size,
            int(config.chunk_size),
            int(config.max_action_dim),
        )
        return self._torch.randn(
            noise_shape,
            dtype=self._torch.float32,
            device=self.device,
            generator=self._noise_generator,
        )


def _libero_observation(batch: dict[str, object], torch) -> dict[str, object]:
    images = batch.get("images")
    if not isinstance(images, Mapping) or "agentview" not in images:
        raise SmolVLAPolicyRuntimeError("invalid_action", "LIBERO observation requires agentview")
    task = batch.get("task")
    if not isinstance(task, str) or not task:
        raise SmolVLAPolicyRuntimeError(
            "invalid_action", "LIBERO observation requires an instruction"
        )

    proprioception = np.asarray(batch.get("proprioception"), dtype=np.float32)
    if proprioception.ndim != 1:
        raise SmolVLAPolicyRuntimeError(
            "invalid_action", "LIBERO proprioception must be one-dimensional"
        )
    agentview = _image_tensor(images["agentview"], torch)
    wrist = _image_tensor(images.get("wrist", images["agentview"]), torch)
    if proprioception.shape != (8,):
        raise SmolVLAPolicyRuntimeError(
            "invalid_action",
            "LIBERO proprioception must contain the official 8-dimensional "
            "EEF position, axis-angle, and gripper state",
        )
    return {
        "task": task,
        "observation.state": torch.from_numpy(np.ascontiguousarray(proprioception)),
        "observation.images.image": agentview,
        "observation.images.image2": wrist,
    }


def _validate_smolvla_preprocessor_state(preprocessor: object) -> None:
    for step in getattr(preprocessor, 'steps', ()):
        stats = getattr(step, 'stats', None)
        if not isinstance(stats, dict):
            continue
        state_stats = stats.get('observation.state')
        if not isinstance(state_stats, dict):
            continue
        for value in state_stats.values():
            if isinstance(value, np.ndarray) and value.ndim == 1 and value.shape != (8,):
                raise SmolVLAPolicyRuntimeError(
                    "model_load_error",
                    "SmolVLA observation.state normalizer must be 8-dimensional",
                )


def _image_tensor(image: object, torch):
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise SmolVLAPolicyRuntimeError(
            "invalid_action", "LIBERO images must be HWC uint8 RGB arrays"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(value))
    return tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32).div(255.0)


def _action_chunk_values(action: object, n_action_steps: int) -> np.ndarray:
    value = action.detach().cpu().numpy() if hasattr(action, "detach") else action
    flattened = np.asarray(value, dtype=np.float32).reshape(-1)
    if flattened.size < n_action_steps * 7:
        raise ValueError(
            "postprocessed action must contain at least "
            f"{n_action_steps * 7} values, got {flattened.size}"
        )
    return flattened[: n_action_steps * 7].reshape(n_action_steps, 7)


def _action_validation_error(action: np.ndarray | None, error: Exception) -> str:
    if action is None or action.shape != (7,) or not np.isfinite(action).all():
        return str(error)
    return (
        f"{error}; observed min={float(action.min()):.6g}, "
        f"max={float(action.max()):.6g}"
    )


def _scale_control_action(raw: np.ndarray, control: ActionControlSpec) -> np.ndarray:
    action = raw.copy()
    if control.mode == "scaled":
        if action.ndim == 2:
            action[:, :3] *= control.translation_scale
            action[:, 3:6] *= control.rotation_scale
        else:
            action[:3] *= control.translation_scale
            action[3:6] *= control.rotation_scale
    return action


def _transform_control_action(
    raw: np.ndarray, control: ActionControlSpec
) -> np.ndarray:
    return np.clip(_scale_control_action(raw, control), -1.0, 1.0)


def _action_clipped(
    scaled_action: np.ndarray | None, transformed_action: np.ndarray | None
) -> bool:
    if scaled_action is None or transformed_action is None:
        return False
    if transformed_action.shape != scaled_action.shape:
        return False
    if not np.isfinite(scaled_action).all() or not np.isfinite(transformed_action).all():
        return False
    return not np.array_equal(scaled_action, transformed_action)


def _action_transform_metadata(control: ActionControlSpec) -> str:
    if control.mode == "scaled":
        return (
            "scaled"
            f"[translation={control.translation_scale},rotation={control.rotation_scale}] "
            "-> clip[-1,1]"
        )
    return "clip[-1,1]"


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
