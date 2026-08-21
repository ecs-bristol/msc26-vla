from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION


@PreTrainedConfig.register_subclass("smolvla_int4")
@dataclass
class SmolVLAInt4Config(PreTrainedConfig):
    # Source checkpoint. Defaults match the reportable PC-local reference run.
    checkpoint: str = "HuggingFaceVLA/smolvla_libero"
    revision: str = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"

    # "none" loads the plain SmolVLA checkpoint (fp16 baseline through the same
    # plugin harness). "int4_groupwise" is the self-contained per-group 4-bit
    # weight quantization; "bnb_nf4" is the optional bitsandbytes NF4 variant.
    quant_method: str = "int4_groupwise"
    group_size: int = 128
    # "language" quantizes only the text transformer (vision encoder and
    # connector stay fp16); "backbone" quantizes vision + connector + text.
    quant_scope: str = "language"
    # Per-part bit widths used when quant_method == "mixed". 16 keeps fp16.
    vision_bits: int = 4
    connector_bits: int = 8
    text_bits: int = 8

    # Inference-time overrides. None keeps the checkpoint's own values.
    num_steps: int | None = None
    n_action_steps: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quant_method not in {
            "none",
            "int4_groupwise",
            "int8_groupwise",
            "bnb_nf4",
            "mixed",
        }:
            raise ValueError(f"unsupported quant_method: {self.quant_method}")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        if self.quant_scope not in {"language", "backbone"}:
            raise ValueError(f"unsupported quant_scope: {self.quant_scope}")
        for name, bits in (
            ("vision_bits", self.vision_bits),
            ("connector_bits", self.connector_bits),
            ("text_bits", self.text_bits),
        ):
            if bits not in {4, 8, 16}:
                raise ValueError(f"{name} must be one of 4, 8, 16")
        if self.num_steps is not None and not (1 <= self.num_steps <= 100):
            raise ValueError("num_steps must be in [1, 100] when set")
        if self.n_action_steps is not None and not (1 <= self.n_action_steps <= 50):
            raise ValueError("n_action_steps must be in [1, 50] when set")

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int] | None:
        return None

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        raise RuntimeError("smolvla_int4 is an inference-only policy")

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        raise RuntimeError("smolvla_int4 is an inference-only policy")

    def validate_features(self) -> None:
        action = self.output_features.get(ACTION)
        if action is not None and tuple(action.shape) != (7,):
            raise ValueError(f"LIBERO int4 action must have shape (7,), got {action.shape}")
