"""LeRobot policy wrapper using the project-owned horizon action buffer.

The external evaluator sees an ordinary ``select_action`` policy. Internally,
the only action-generation call is SmolVLA's ``predict_action_chunk``; the
native SmolVLA ``select_action`` queue is deliberately never used.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy

from libero_platform.policies.fixed_h_action_buffer import FixedHActionBuffer

from .configuration_smolvla_adaptive import SmolVLAAdaptiveConfig

_TRUST_REMOTE_CODE = False


def _load_local_tokenizer(vlm_snapshot_path: str) -> Any:
    """Load the frozen VLM tokenizer without resolving a Hub identifier.

    The checkpoint processor config records the original Hub ID.  That value is
    useful provenance but cannot be the runtime source in offline evaluation:
    ``TokenizerProcessorStep`` otherwise invokes ``AutoTokenizer`` with it.
    Loading the tokenizer here lets us inject the concrete object into that
    processor step while retaining explicit local-only and no-remote-code
    protections.
    """

    from transformers import AutoTokenizer

    snapshot = Path(vlm_snapshot_path).expanduser().resolve(strict=True)
    if not snapshot.is_absolute():  # Defensive: resolve() above should guarantee this.
        raise ValueError("vlm tokenizer source must be an absolute local snapshot path")
    return AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )


def _runtime_preprocessor_overrides(
    *, tokenizer: Any, device: str | None
) -> dict[str, dict[str, Any]]:
    """Return an isolated runtime-only overlay for immutable processor JSON.

    LeRobot loads ``policy_preprocessor.json`` from the frozen checkpoint and
    merges this deep-copied overlay in memory.  The on-disk JSON is never
    rewritten: only the processor instances see the selected runtime device.
    """

    # Copy only JSON-like configuration.  A tokenizer is a live Transformers
    # object and must be passed through by identity rather than duplicated.
    overrides = deepcopy(
        {
            "tokenizer_processor": {"tokenizer_name": None},
            "device_processor": {"device": device or "cpu"},
        }
    )
    overrides["tokenizer_processor"]["tokenizer"] = tokenizer
    return overrides


class _PostprocessedChunkPredictor:
    """Adapt raw evaluator observations to the buffer's narrow chunk protocol."""

    def __init__(
        self,
        base_policy: Any,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
        postprocessor: Callable[[Any], Any],
    ) -> None:
        self._base_policy = base_policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor

    def predict_action_chunk(self, observation: object) -> object:
        if not isinstance(observation, dict):
            raise TypeError("LeRobot policy observations must be dictionaries")
        processed = self._preprocessor(observation)
        chunk = self._base_policy.predict_action_chunk(processed)
        return self._postprocessor(chunk)


class SmolVLAAdaptivePolicy(PreTrainedPolicy):
    """Independent static or safety-triggered horizon wrapper for SmolVLA."""

    config_class = SmolVLAAdaptiveConfig
    name = "smolvla_adaptive"

    def __init__(
        self,
        config: SmolVLAAdaptiveConfig,
        *,
        base_policy: Any | None = None,
        base_preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        base_postprocessor: Callable[[Any], Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__(config)
        self.config = config
        if base_policy is None:
            base_policy, base_preprocessor, base_postprocessor = _load_frozen_base(config)
        if base_preprocessor is None or base_postprocessor is None:
            raise ValueError("base SmolVLA preprocessor and postprocessor are required")

        # Assigning a real SmolVLAPolicy here registers it as a submodule. Tests may
        # inject a mock object instead, so never assume a particular implementation.
        self._base_policy = base_policy
        self._base_preprocessor = base_preprocessor
        self._base_postprocessor = base_postprocessor
        self._chunk_predictor = _PostprocessedChunkPredictor(
            self._base_policy, self._base_preprocessor, self._base_postprocessor
        )
        self._buffer = FixedHActionBuffer(
            self._chunk_predictor,
            horizon=config.fixed_h,
            safety_enabled=config.safety_enabled,
            replan_after_safety_violation=config.replan_after_safety_violation,
            clip_actions=config.clip_actions,
        )

    def get_optim_params(self) -> dict:
        raise RuntimeError("smolvla_adaptive is an inference-only policy")

    @property
    def action_telemetry(self) -> tuple[dict[str, object], ...]:
        """Read-only, JSON-safe action-buffer provenance for the current episode."""

        return tuple(dict(record) for record in self._buffer.telemetry)

    @torch.inference_mode()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Release one postprocessed raw action from the project-owned buffer."""

        try:
            release = self._buffer.next_action(batch)
            return torch.from_numpy(np.asarray(release.action, dtype=np.float32)).unsqueeze(0)
        except Exception:
            self._buffer.reset()
            raise

    def reset(self) -> None:
        """Reset both base-policy state and all episode-local project buffer state."""

        try:
            reset = getattr(self._base_policy, "reset", None)
            if callable(reset):
                reset()
        finally:
            self._buffer.reset()

    def finalize_episode(self) -> None:
        """Record the realized tail of the last model call before episode reset."""

        self._buffer.finalize_episode()

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Prevent evaluator code from accidentally creating a second action queue."""

        del batch
        raise RuntimeError(
            "smolvla_adaptive exposes only select_action; it owns Fixed-H action buffering"
        )

    def forward(self, batch: dict[str, torch.Tensor], **_: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        del batch
        raise RuntimeError("smolvla_adaptive is an inference-only policy")


def _load_frozen_base(
    config: SmolVLAAdaptiveConfig,
) -> tuple[Any, Callable[[dict[str, Any]], dict[str, Any]], Callable[[Any], Any]]:
    """Load only local, fixed components; never mutate n_action_steps.

    Concrete installed SmolVLA/SmolVLM2 classes are used instead of an
    AutoModel remote-code path. The nested Transformers calls retain their
    default ``trust_remote_code=False`` behavior and this is not configurable.
    """

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if _TRUST_REMOTE_CODE:
        raise RuntimeError("smolvla_adaptive never permits remote code")
    load_kwargs = {
        "revision": config.base_revision,
        "cache_dir": config.base_cache_dir,
        "local_files_only": config.local_files_only,
    }
    base_config = SmolVLAConfig.from_pretrained(config.base_load_path, **load_kwargs)
    base_config.device = config.device or "cpu"
    if int(base_config.chunk_size) != config.chunk_size:
        raise ValueError("frozen SmolVLA checkpoint chunk_size must equal 50")
    # The installed SmolVLA VLM loader accepts a model path but no revision
    # argument. Point it at the immutable local snapshot so it cannot follow
    # an unpinned Hub cache ref.
    base_config.vlm_model_name = config.vlm_load_path
    # num_steps is explicitly frozen by the formal protocol. n_action_steps is
    # intentionally untouched and is never used as the wrapper's control knob.
    base_config.num_steps = config.num_steps
    base_policy = SmolVLAPolicy.from_pretrained(
        config.base_load_path,
        config=base_config,
        **load_kwargs,
    )
    local_tokenizer = _load_local_tokenizer(config.vlm_load_path)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=base_config,
        pretrained_path=config.base_load_path,
        pretrained_revision=config.base_revision,
        preprocessor_overrides=_runtime_preprocessor_overrides(
            tokenizer=local_tokenizer,
            device=config.device,
        ),
    )
    return base_policy, preprocessor, postprocessor
