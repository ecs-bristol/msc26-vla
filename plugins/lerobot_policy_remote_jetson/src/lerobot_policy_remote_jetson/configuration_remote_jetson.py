from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION


@PreTrainedConfig.register_subclass("remote_jetson")
@dataclass
class RemoteJetsonConfig(PreTrainedConfig):
    endpoint: str = "http://10.42.0.2:8081"
    checkpoint: str = "HuggingFaceVLA/smolvla_libero"
    revision: str = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    precision: str = "fp16"
    request_timeout_s: float = 180.0
    reset_timeout_s: float = 30.0
    seed: int = 42
    suite: str = "libero_spatial"
    n_action_steps: int = 1
    telemetry_path: str | None = None
    device: str | None = "cpu"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.endpoint = self.endpoint.rstrip("/")
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must start with http:// or https://")
        if self.request_timeout_s <= 0 or self.reset_timeout_s <= 0:
            raise ValueError("HTTP timeouts must be positive")
        if self.precision != "fp16":
            raise ValueError("the reportable Jetson workflow requires fp16 precision")
        if not (1 <= self.n_action_steps <= 50):
            raise ValueError("n_action_steps must be in [1, 50]")

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
        raise RuntimeError("remote_jetson is an inference-only policy")

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        raise RuntimeError("remote_jetson is an inference-only policy")

    def validate_features(self) -> None:
        action = self.output_features.get(ACTION)
        if action is not None and tuple(action.shape) != (7,):
            raise ValueError(f"LIBERO remote action must have shape (7,), got {action.shape}")
