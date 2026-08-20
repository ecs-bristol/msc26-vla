from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from libero_platform.spec import (
    ExperimentSpec,
    ResolvedExperimentSpec,
    load_experiment_spec,
    write_resolved_spec,
)


VALID = {
    "schema_version": 1,
    "name": "fake_smoke",
    "benchmark": {
        "backend": "fake",
        "suite": "libero_spatial",
        "task_ids": [0],
        "initial_state_ids": [0],
        "max_steps": 5,
    },
    "policy": {
        "key": "zero_policy",
        "checkpoint": "none",
        "precision": "none",
        "quantization": "none",
    },
    "deployment": {"mode": "pc_local", "profile": "pc_default"},
    "execution": {
        "episodes_per_initial_state": 1,
        "warmup_episodes": 0,
        "seed": 42,
        "on_episode_failure": "continue",
    },
    "viewer": {"enabled": False},
    "recording": {
        "save_frames": False,
        "save_video": False,
        "frame_stride": 20,
        "save_steps": True,
    },
}


def test_valid_spec_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(VALID, sort_keys=False), encoding="utf-8")

    spec = load_experiment_spec(path)

    assert spec.name == "fake_smoke"
    assert spec.execution.seed == 42


def test_unknown_field_is_rejected() -> None:
    payload = {**VALID, "dashboard": {"enabled": True}}

    with pytest.raises(ValidationError, match="dashboard"):
        ExperimentSpec.model_validate(payload)


@pytest.mark.parametrize("value", [True, "42", -1])
def test_seed_requires_non_negative_integer(value: object) -> None:
    payload = {**VALID, "execution": {**VALID["execution"], "seed": value}}

    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


def test_identity_action_control_requires_unit_scales() -> None:
    payload = {
        **VALID,
        "policy": {
            **VALID["policy"],
            "action_control": {
                "mode": "identity",
                "translation_scale": 0.75,
            },
        },
    }

    with pytest.raises(
        ValidationError, match="identity action control requires unit scales"
    ):
        ExperimentSpec.model_validate(payload)


def test_smolvla_inference_settings_round_trip_through_spec() -> None:
    payload = {
        **VALID,
        "policy": {
            **VALID["policy"],
            "smolvla_inference": {"n_action_steps": 1, "num_steps": 10},
        },
    }

    spec = ExperimentSpec.model_validate(payload)

    assert spec.policy.smolvla_inference.n_action_steps == 1
    assert spec.policy.smolvla_inference.num_steps == 10


def test_resolved_spec_is_written_as_yaml(tmp_path: Path) -> None:
    spec = ResolvedExperimentSpec.model_validate(
        {
            **VALID,
            "source_path": "configs/experiments/smoke_fake.yaml",
            "dataset_directory": "datasets/libero",
            "resolved_checkpoint": "none",
            "policy_adapter": "zero",
        }
    )
    path = tmp_path / "resolved.yaml"

    write_resolved_spec(path, spec)

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == spec.model_dump(
        mode="json"
    )
