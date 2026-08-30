from __future__ import annotations

from pathlib import Path

import pytest

from libero_platform.catalog import Catalog
from libero_platform.spec import load_experiment_spec


FORMAL_CONFIGS = (
    ("libero_spatial_oracle.yaml", "libero_spatial", "oracle_or_scripted", "none"),
    ("libero_spatial_smolvla.yaml", "libero_spatial", "smolvla_libero", "fp16"),
    ("libero_object_oracle.yaml", "libero_object", "oracle_or_scripted", "none"),
    ("libero_object_smolvla.yaml", "libero_object", "smolvla_libero", "fp16"),
    ("libero_goal_oracle.yaml", "libero_goal", "oracle_or_scripted", "none"),
    ("libero_goal_smolvla.yaml", "libero_goal", "smolvla_libero", "fp16"),
)


def test_every_experiment_config_loads_without_forbidden_schema_fields(
    catalog_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_REVISION", "test-revision")
    monkeypatch.setenv("JETSON_ENDPOINT", "http://jetson-under-test:8081")
    for path in sorted((catalog_root / "experiments").glob("*.yaml")):
        load_experiment_spec(path)


@pytest.mark.parametrize(
    ("filename", "suite", "policy_key", "precision"), FORMAL_CONFIGS
)
def test_formal_experiment_config_is_fixed_and_catalog_resolvable(
    catalog_root: Path, filename: str, suite: str, policy_key: str, precision: str
) -> None:
    path = catalog_root / "experiments" / filename

    spec = load_experiment_spec(path)
    resolved = Catalog.load(catalog_root).resolve(spec, path)

    assert spec.name == path.stem
    assert spec.benchmark.backend == "libero"
    assert spec.benchmark.suite == suite
    assert spec.benchmark.task_ids == [0, 1]
    assert spec.benchmark.initial_state_ids == [0, 1, 2, 3, 4]
    assert spec.benchmark.max_steps == 600
    assert spec.policy.key == policy_key
    assert spec.policy.checkpoint == "catalog:default"
    assert spec.policy.precision == precision
    assert spec.policy.quantization == "none"
    assert spec.deployment.mode == "pc_local"
    assert spec.deployment.profile == "pc_default"
    assert spec.execution.episodes_per_initial_state == 1
    assert spec.execution.warmup_episodes == 0
    assert spec.execution.seed == 42
    assert spec.execution.on_episode_failure == "continue"
    assert spec.viewer.enabled is False
    assert spec.recording.save_frames is False
    assert spec.recording.save_video is True
    assert spec.recording.frame_stride == 20
    assert spec.recording.save_steps is True
    assert resolved.policy_adapter == (
        "demo_replay" if policy_key == "oracle_or_scripted" else "smolvla"
    )


def test_formal_experiment_configs_define_the_fixed_sixty_episode_matrix(
    catalog_root: Path,
) -> None:
    paths = tuple(catalog_root / "experiments" / filename for filename, *_ in FORMAL_CONFIGS)
    specs = [load_experiment_spec(path) for path in paths]

    episodes = sum(
        len(spec.benchmark.task_ids)
        * len(spec.benchmark.initial_state_ids)
        * spec.execution.episodes_per_initial_state
        for spec in specs
    )

    assert len(FORMAL_CONFIGS) == 6
    assert {path.name for path in paths} == {
        "libero_spatial_oracle.yaml",
        "libero_spatial_smolvla.yaml",
        "libero_object_oracle.yaml",
        "libero_object_smolvla.yaml",
        "libero_goal_oracle.yaml",
        "libero_goal_smolvla.yaml",
    }
    assert episodes == 60


def test_custom_runner_has_no_active_jetson_remote_experiment_configs(catalog_root: Path) -> None:
    assert not list((catalog_root / "experiments").glob("*jetson_remote_smolvla*.yaml"))


def test_pc_local_smolvla_baseline_matches_the_jetson_task_envelope(
    catalog_root: Path,
) -> None:
    path = catalog_root / "experiments" / "libero_spatial_pc_local_smolvla_baseline.yaml"
    spec = load_experiment_spec(path)
    resolved = Catalog.load(catalog_root).resolve(spec, path)

    assert spec.name == "libero_spatial_pc_local_smolvla_baseline"
    assert spec.benchmark.suite == "libero_spatial"
    assert spec.benchmark.task_ids == [0]
    assert spec.benchmark.initial_state_ids == [0]
    assert spec.benchmark.max_steps == 100
    assert spec.policy.key == "smolvla_libero"
    assert spec.policy.precision == "fp16"
    assert spec.deployment.mode == "pc_local"
    assert spec.deployment.profile == "pc_default"
    assert spec.execution.seed == 42
    assert spec.execution.on_episode_failure == "stop"
    assert resolved.policy_adapter == "smolvla"
    assert resolved.viewer.enabled is True
    assert resolved.recording.save_frames is True


def test_pc_local_smolvla_screening_covers_all_spatial_tasks(
    catalog_root: Path,
) -> None:
    path = catalog_root / "experiments" / "libero_spatial_pc_local_smolvla_screening.yaml"
    spec = load_experiment_spec(path)
    resolved = Catalog.load(catalog_root).resolve(spec, path)

    assert spec.name == "libero_spatial_pc_local_smolvla_screening"
    assert spec.benchmark.suite == "libero_spatial"
    assert spec.benchmark.task_ids == list(range(10))
    assert spec.benchmark.initial_state_ids == [0]
    assert spec.benchmark.max_steps == 280
    assert spec.benchmark.settle_steps == 0
    assert spec.benchmark.initial_state_source == "demonstration"
    assert spec.policy.key == "smolvla_libero"
    assert spec.deployment.mode == "pc_local"
    assert spec.execution.episodes_per_initial_state == 1
    assert spec.execution.on_episode_failure == "continue"
    assert resolved.policy_adapter == "smolvla"
    assert resolved.viewer.enabled is False
    assert resolved.recording.save_frames is True
    assert resolved.recording.frame_stride == 10


def test_oracle_all_tasks_smoke_config_uses_demo_replay_without_recordings(
    catalog_root: Path,
) -> None:
    path = catalog_root / "experiments" / "libero_spatial_oracle_all_tasks_smoke.yaml"
    spec = load_experiment_spec(path)
    resolved = Catalog.load(catalog_root).resolve(spec, path)

    assert spec.name == "libero_spatial_oracle_all_tasks_smoke"
    assert spec.benchmark.suite == "libero_spatial"
    assert spec.benchmark.task_ids == list(range(10))
    assert spec.benchmark.initial_state_ids == [0]
    assert spec.benchmark.max_steps == 600
    assert spec.benchmark.settle_steps == 0
    assert spec.benchmark.initial_state_source == "demonstration"
    assert spec.policy.key == "oracle_or_scripted"
    assert spec.deployment.mode == "pc_local"
    assert spec.execution.episodes_per_initial_state == 1
    assert resolved.policy_adapter == "demo_replay"
    assert resolved.viewer.enabled is False
    assert resolved.recording.save_frames is False
    assert resolved.recording.save_video is False


def test_pc_local_smolvla_official_alignment_is_a_one_hundred_trial_protocol(
    catalog_root: Path,
) -> None:
    path = catalog_root / "experiments" / "libero_spatial_pc_local_smolvla_official_alignment.yaml"
    spec = load_experiment_spec(path)
    resolved = Catalog.load(catalog_root).resolve(spec, path)

    assert spec.name == "libero_spatial_pc_local_smolvla_official_alignment"
    assert spec.benchmark.suite == "libero_spatial"
    assert spec.benchmark.task_ids == list(range(10))
    assert spec.benchmark.initial_state_ids == list(range(10))
    assert spec.execution.episodes_per_initial_state == 1
    assert spec.benchmark.max_steps == 600
    assert spec.benchmark.settle_steps == 0
    assert spec.benchmark.initial_state_source == "benchmark"
    assert spec.policy.key == "smolvla_libero"
    assert spec.policy.smolvla_inference.n_action_steps == 1
    assert spec.policy.smolvla_inference.num_steps == 10
    assert spec.deployment.mode == "pc_local"
    assert spec.execution.on_episode_failure == "continue"
    assert resolved.policy_adapter == "smolvla"
    assert resolved.viewer.enabled is False
    assert resolved.recording.save_frames is False
    assert resolved.recording.save_video is False


def test_pc_local_smolvla_paired_configs_vary_only_initial_state_source(
    catalog_root: Path,
) -> None:
    catalog = Catalog.load(catalog_root)
    paths = [
        catalog_root / "experiments" / "libero_spatial_pc_local_smolvla_paired_demo.yaml",
        catalog_root / "experiments" / "libero_spatial_pc_local_smolvla_paired_benchmark.yaml",
    ]
    specs = [catalog.resolve(load_experiment_spec(path), path) for path in paths]
    demo, benchmark = specs

    assert demo.benchmark.initial_state_source == "demonstration"
    assert benchmark.benchmark.initial_state_source == "benchmark"
    assert demo.benchmark.task_ids == benchmark.benchmark.task_ids == [3]
    assert demo.benchmark.initial_state_ids == benchmark.benchmark.initial_state_ids == [0]
    assert demo.benchmark.max_steps == benchmark.benchmark.max_steps == 280
    assert demo.benchmark.settle_steps == benchmark.benchmark.settle_steps == 10
    assert demo.execution.episodes_per_initial_state == benchmark.execution.episodes_per_initial_state == 5
    assert demo.execution.seed == benchmark.execution.seed == 42
    assert demo.policy == benchmark.policy
    assert demo.deployment == benchmark.deployment
    assert demo.viewer == benchmark.viewer
    assert demo.recording == benchmark.recording


def test_pc_local_smolvla_settle_ablation_configs_vary_only_settle_steps(
    catalog_root: Path,
) -> None:
    catalog = Catalog.load(catalog_root)
    paths = [
        catalog_root / "experiments" / "libero_spatial_pc_local_smolvla_settle_0.yaml",
        catalog_root / "experiments" / "libero_spatial_pc_local_smolvla_settle_10.yaml",
    ]
    specs = [catalog.resolve(load_experiment_spec(path), path) for path in paths]
    no_settle, official_settle = specs

    assert no_settle.benchmark.settle_steps == 0
    assert official_settle.benchmark.settle_steps == 10
    assert no_settle.benchmark.initial_state_source == official_settle.benchmark.initial_state_source == "demonstration"
    assert no_settle.benchmark.task_ids == official_settle.benchmark.task_ids == [3]
    assert no_settle.benchmark.initial_state_ids == official_settle.benchmark.initial_state_ids == [0]
    assert no_settle.benchmark.max_steps == official_settle.benchmark.max_steps == 280
    assert no_settle.execution.episodes_per_initial_state == official_settle.execution.episodes_per_initial_state == 5
    assert no_settle.execution.seed == official_settle.execution.seed == 42
    assert no_settle.policy == official_settle.policy
    assert no_settle.deployment == official_settle.deployment
    assert no_settle.viewer == official_settle.viewer
    assert no_settle.recording == official_settle.recording


def test_pc_local_smolvla_action_control_configs_vary_only_action_control(
    catalog_root: Path,
) -> None:
    catalog = Catalog.load(catalog_root)
    paths = [
        catalog_root
        / "experiments"
        / "libero_spatial_pc_local_smolvla_action_identity.yaml",
        catalog_root
        / "experiments"
        / "libero_spatial_pc_local_smolvla_action_scaled_075.yaml",
    ]
    identity, scaled = [
        catalog.resolve(load_experiment_spec(path), path) for path in paths
    ]

    assert identity.policy.action_control.mode == "identity"
    assert identity.policy.action_control.translation_scale == 1.0
    assert identity.policy.action_control.rotation_scale == 1.0
    assert scaled.policy.action_control.mode == "scaled"
    assert scaled.policy.action_control.translation_scale == 0.75
    assert scaled.policy.action_control.rotation_scale == 0.75

    def without_action_control(spec: object) -> dict[str, object]:
        payload = spec.model_dump(
            mode="python",
            include={
                "benchmark",
                "policy",
                "deployment",
                "execution",
                "viewer",
                "recording",
                "policy_adapter",
                "resolved_checkpoint",
                "dataset_directory",
                "device_metadata",
                "policy_endpoint",
            },
        )
        payload["policy"].pop("action_control", None)
        return payload

    assert without_action_control(identity) == without_action_control(scaled)


def test_pc_local_smolvla_selected_five_repeat_config_matches_selection_and_retained_control(
    catalog_root: Path,
) -> None:
    path = (
        catalog_root / "experiments" / "libero_spatial_pc_local_smolvla_selected_5rep.yaml"
    )
    spec = load_experiment_spec(path)
    resolved = Catalog.load(catalog_root).resolve(spec, path)

    assert spec.name == "libero_spatial_pc_local_smolvla_selected_5rep"
    assert spec.benchmark.suite == "libero_spatial"
    assert spec.benchmark.task_ids == [3, 0, 1, 2, 4]
    assert spec.benchmark.initial_state_ids == [0]
    assert spec.benchmark.max_steps == 280
    assert spec.benchmark.settle_steps == 0
    assert spec.benchmark.initial_state_source == "demonstration"
    assert spec.policy.key == "smolvla_libero"
    assert spec.policy.precision == "fp16"
    assert spec.policy.quantization == "none"
    assert spec.policy.action_control.mode == "identity"
    assert spec.policy.action_control.translation_scale == 1.0
    assert spec.policy.action_control.rotation_scale == 1.0
    assert spec.deployment.mode == "pc_local"
    assert spec.deployment.profile == "pc_default"
    assert spec.execution.episodes_per_initial_state == 5
    assert spec.execution.warmup_episodes == 0
    assert spec.execution.seed == 42
    assert spec.execution.on_episode_failure == "continue"
    assert resolved.policy_adapter == "smolvla"
    assert resolved.viewer.enabled is False
    assert resolved.recording.save_frames is True
    assert resolved.recording.save_video is False
    assert resolved.recording.frame_stride == 10
    assert resolved.recording.save_steps is True
