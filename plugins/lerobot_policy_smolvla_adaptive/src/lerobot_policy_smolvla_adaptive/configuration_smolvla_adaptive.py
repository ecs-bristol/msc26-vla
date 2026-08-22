from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION


_FROZEN_SMOLVLA_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
_FROZEN_SMOLVLM2_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


@PreTrainedConfig.register_subclass("smolvla_adaptive")
@dataclass
class SmolVLAAdaptiveConfig(PreTrainedConfig):
    """Configuration for the external Fixed-H wrapper, not a new checkpoint."""

    base_checkpoint: str = "HuggingFaceVLA/smolvla_libero"
    base_revision: str = _FROZEN_SMOLVLA_REVISION
    fixed_h: int = 20
    num_steps: int = 2
    precision: str = "fp16"
    base_cache_dir: str | None = None
    base_snapshot_path: str | None = None
    vlm_checkpoint: str = "HuggingFaceTB/SmolVLM2-500M-Instruct"
    vlm_revision: str = _FROZEN_SMOLVLM2_REVISION
    vlm_snapshot_path: str | None = None
    local_files_only: bool = True
    device: str | None = "cpu"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.base_checkpoint:
            raise ValueError("base_checkpoint must be set")
        if self.base_revision != _FROZEN_SMOLVLA_REVISION:
            raise ValueError("base_revision must equal the frozen SmolVLA revision")
        if self.fixed_h != 20:
            raise ValueError("the initial wrapper supports only fixed_h=20")
        if self.num_steps != 2:
            raise ValueError("the frozen evaluation protocol requires num_steps=2")
        if self.precision != "fp16":
            raise ValueError("the frozen evaluation protocol requires fp16")
        if not self.local_files_only:
            raise ValueError("the wrapper must use a local frozen cache or snapshot")
        if self.base_snapshot_path and not Path(self.base_snapshot_path).expanduser().is_absolute():
            raise ValueError("base_snapshot_path must be an absolute local snapshot path")
        if self.vlm_checkpoint != "HuggingFaceTB/SmolVLM2-500M-Instruct":
            raise ValueError("vlm_checkpoint must equal the frozen SmolVLM2 repository")
        if self.vlm_revision != _FROZEN_SMOLVLM2_REVISION:
            raise ValueError("vlm_revision must equal the frozen SmolVLM2 revision")
        if self.vlm_snapshot_path and not Path(self.vlm_snapshot_path).expanduser().is_absolute():
            raise ValueError("vlm_snapshot_path must be an absolute local snapshot path")

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
        raise RuntimeError("smolvla_adaptive is an inference-only policy")

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        raise RuntimeError("smolvla_adaptive is an inference-only policy")

    def validate_features(self) -> None:
        action = self.output_features.get(ACTION)
        if action is not None and tuple(action.shape) != (7,):
            raise ValueError(f"LIBERO action must have shape (7,), got {action.shape}")

    @property
    def base_load_path(self) -> str:
        """Use an explicit snapshot when supplied; otherwise use the frozen repo id."""

        if self.base_snapshot_path:
            path = Path(self.base_snapshot_path).expanduser()
            if path.name != self.base_revision or path.parent.name != "snapshots":
                raise ValueError("base_snapshot_path must name the frozen SmolVLA revision")
            return str(path)
        return self.base_checkpoint

    @property
    def vlm_load_path(self) -> str:
        """The nested VLM must use its exact snapshot, not an unpinned cache ref."""

        if not self.vlm_snapshot_path:
            raise ValueError(
                "vlm_snapshot_path is required for construction so SmolVLM2 revision is explicit"
            )
        path = Path(self.vlm_snapshot_path).expanduser()
        if path.name != self.vlm_revision or path.parent.name != "snapshots":
            raise ValueError("vlm_snapshot_path must name the frozen SmolVLM2 revision")
        return str(path)
