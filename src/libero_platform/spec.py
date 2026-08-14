from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class BenchmarkSpec(StrictModel):
    backend: Literal["fake", "libero"]
    suite: Literal["libero_spatial", "libero_object", "libero_goal"]
    task_ids: list[StrictInt] = Field(min_length=1)
    initial_state_ids: list[StrictInt] = Field(min_length=1)
    max_steps: StrictInt = Field(ge=1, le=5000)
    settle_steps: StrictInt = Field(default=0, ge=0, le=100)
    initial_state_source: Literal["demonstration", "benchmark"] = "demonstration"

    @model_validator(mode="after")
    def unique_non_negative_ids(self) -> "BenchmarkSpec":
        for name, values in (
            ("task_ids", self.task_ids),
            ("initial_state_ids", self.initial_state_ids),
        ):
            if any(value < 0 for value in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique non-negative integers")
        return self


class ActionControlSpec(StrictModel):
    mode: Literal["identity", "scaled"] = "identity"
    translation_scale: float = Field(default=1.0, gt=0.0, le=1.0)
    rotation_scale: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def identity_uses_unit_scales(self) -> "ActionControlSpec":
        if self.mode == "identity" and (
            self.translation_scale != 1.0 or self.rotation_scale != 1.0
        ):
            raise ValueError("identity action control requires unit scales")
        return self


class SmolVLAInferenceSpec(StrictModel):
    n_action_steps: StrictInt = Field(default=1, ge=1, le=50)
    num_steps: StrictInt = Field(default=10, ge=1, le=100)


class PolicySpec(StrictModel):
    key: str = Field(min_length=1)
    model_key: str | None = Field(default=None, min_length=1)
    checkpoint: str = Field(min_length=1)
    revision: str | None = None
    precision: Literal["none", "fp32", "fp16", "bf16", "int8", "int4"]
    quantization: Literal["none", "int8", "int4"]
    action_control: ActionControlSpec = Field(default_factory=ActionControlSpec)
    smolvla_inference: SmolVLAInferenceSpec = Field(
        default_factory=SmolVLAInferenceSpec
    )


class DeploymentSpec(StrictModel):
    mode: Literal[
        "pc_local",
        "jetson_local",
        "jetson_quantized",
        "jetson_remote_client",
        "remote_server",
    ]
    profile: str = Field(min_length=1)
    allow_loopback_endpoint: StrictBool = False


class ExecutionSpec(StrictModel):
    episodes_per_initial_state: StrictInt = Field(ge=1, le=100)
    warmup_episodes: StrictInt = Field(ge=0, le=10)
    seed: StrictInt = Field(ge=0, le=2_147_483_647)
    on_episode_failure: Literal["continue", "stop"]


class ViewerSpec(StrictModel):
    enabled: StrictBool


class RecordingSpec(StrictModel):
    save_frames: StrictBool
    save_video: StrictBool
    frame_stride: StrictInt = Field(ge=1, le=1000)
    save_steps: StrictBool


class ExperimentSpec(StrictModel):
    schema_version: Literal[1]
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
    benchmark: BenchmarkSpec
    policy: PolicySpec
    deployment: DeploymentSpec
    execution: ExecutionSpec
    viewer: ViewerSpec
    recording: RecordingSpec


class ResolvedExperimentSpec(ExperimentSpec):
    source_path: str
    dataset_directory: str
    resolved_checkpoint: str
    resolved_revision: str | None
    policy_adapter: str
    policy_endpoint: str | None = None
    device_metadata: dict[str, str | int | float | None] = Field(default_factory=dict)


def load_experiment_spec(path: Path) -> ExperimentSpec:
    resolved = path.resolve(strict=True)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment YAML root must be a mapping")
    _expand_policy_revision_environment_placeholder(payload)
    return ExperimentSpec.model_validate(payload)


_ENVIRONMENT_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _expand_policy_revision_environment_placeholder(payload: dict[object, object]) -> None:
    """Expand one explicit `${NAME}` placeholder in policy.revision, if present."""
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        return
    revision = policy.get("revision")
    if not isinstance(revision, str) or "${" not in revision:
        return
    match = _ENVIRONMENT_PLACEHOLDER.fullmatch(revision)
    if match is None:
        raise ValueError(
            "policy.revision supports only an exact environment placeholder like "
            "'${MODEL_REVISION}'"
        )
    variable_name = match.group(1)
    value = os.environ.get(variable_name)
    if not value:
        raise ValueError(
            f"policy.revision requires environment variable {variable_name!r}, but it is not set"
        )
    policy["revision"] = value


def write_resolved_spec(path: Path, spec: ResolvedExperimentSpec) -> None:
    path.write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
