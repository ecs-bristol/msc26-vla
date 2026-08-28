from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "libero_spatial_paired_pilot.py"
CONFIG = ROOT / "configs" / "evaluation" / "libero_spatial_adaptive_v2_prereg.yaml"
BASE_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
V2_NAME = "Adaptive-v2a-H20→H1"


def _module():
    spec = importlib.util.spec_from_file_location("adaptive_v2_pilot_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(tmp_path: Path, revision: str) -> Path:
    path = tmp_path / "hf" / "hub" / "models--test" / "snapshots" / revision
    path.mkdir(parents=True)
    return path


def test_v2_dry_run_has_300_plans_and_held_out_paired_seeds(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "dry-run"
    result = module.materialize_dry_run(
        config_path=CONFIG,
        output_dir=output,
        base_snapshot_path=str(_snapshot(tmp_path, BASE_REVISION)),
        vlm_snapshot_path=str(_snapshot(tmp_path, VLM_REVISION)),
    )

    assert result["planned_episodes"] == 300
    resolved = json.loads((output / "resolved_config.json").read_text())
    static_h20 = next(
        item for item in resolved["conditions"] if item["name"] == "Static-H20"
    )
    v2 = next(item for item in resolved["conditions"] if item["name"] == V2_NAME)
    assert static_h20 == {
        "name": "Static-H20",
        "fixed_h": 20,
        "safety_enabled": False,
        "replan_after_safety_violation": False,
        "adaptive_v2_trigger": False,
        "clip_actions": False,
    }
    assert v2 == {
        "name": V2_NAME,
        "fixed_h": 20,
        "safety_enabled": True,
        "replan_after_safety_violation": False,
        "adaptive_v2_trigger": True,
        "clip_actions": False,
    }
    config, trials = module._read_execution_inputs(output)
    selected = module._selected_trials(trials, set(range(10)), 1)
    assert len(selected) == 10
    assert [(item["task_id"], item["seed"], item["initial_state_id"]) for item in selected] == [
        (task_id, 1000, 0) for task_id in range(10)
    ]
    assert [item["name"] for item in config["conditions"] if item["name"] == V2_NAME] == [
        V2_NAME
    ]
    episodes = [json.loads(path.read_text()) for path in (output / "episodes").rglob("*.json")]
    assert len(episodes) == 300
    assert all(item["status"] == "planned_dry_run" for item in episodes)
    assert {item["condition"] for item in episodes} == {
        "Static-H1-original", "Static-H5", "Static-H10", "Static-H20",
        "Static-H50", V2_NAME,
    }
    manifest = json.loads((output / "paired_manifest.json").read_text())
    assert manifest["inference_seed_namespace"] == "adaptive-v2-heldout-v1|libero_spatial"
    assert len(manifest["trials"]) == len(manifest["inference_seed_records"]) == 50
    held_out = {item["inference_seed"] for item in manifest["inference_seed_records"]}
    legacy = {module._inference_seed(trial) for trial in manifest["trials"]}
    assert len(held_out) == 50
    assert held_out.isdisjoint(legacy)
    for key in {(item["task_id"], item["seed"], item["initial_state_id"]) for item in episodes}:
        paired = [
            item["inference_seed"] for item in episodes
            if (item["task_id"], item["seed"], item["initial_state_id"]) == key
        ]
        assert len(paired) == 6 and len(set(paired)) == 1


class _Policy:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.records: list[dict[str, object]] = []

    @property
    def telemetry(self) -> tuple[dict[str, object], ...]:
        return tuple(self.records)

    @property
    def model_inference_time_s(self) -> float:
        return 0.25

    def reset(self) -> None:
        return None

    def finalize_episode(self) -> None:
        return None

    def close(self) -> None:
        return None

    def select_action(self, observation: object) -> np.ndarray:
        del observation
        self.order.append("policy_select")
        self.records.extend(
            [
                {"event": "refill", "model_invocation": 1, "planned_horizon": 20},
                {
                    "event": "action_release",
                    "model_invocation": 1,
                    "adaptive_v2_enabled": True,
                    "adaptive_v2_triggered": True,
                    "adaptive_v2_trigger_raw_values": [1.2],
                    "adaptive_v2_trigger_bounds": [1.0],
                    "adaptive_v2_trigger_excess": [0.2],
                    "adaptive_v2_trigger_severity": [0.1],
                    "adaptive_v2_trigger_persistence_counts": [1],
                    "adaptive_v2_trigger_tail_discarded_actions": 19,
                    "adaptive_v2_buffer_entries_cleared": 49,
                    "adaptive_v2_horizon_before": 20,
                    "adaptive_v2_horizon_after": 1,
                    "adaptive_v2_recovery_horizon": 20,
                    "adaptive_v2_state_before": "monitoring_h20",
                    "adaptive_v2_state_after": "fallback_h1_pending",
                    "adaptive_v2_immediate_dimensions": [0],
                    "adaptive_v2_persistent_dimensions": [],
                    "adaptive_v2_cooldown_h20_actions": 20,
                    "trigger_violation_dimensions": [0],
                    "range_violation": True,
                    "range_violation_dimensions": [0],
                    "range_violation_excess": [0.2],
                    "trigger_range_violation": True,
                    "gripper_only_range_violation": False,
                    "range_clipped": False,
                    "buffer_discarded": True,
                },
                {
                    "event": "call_finalized",
                    "model_invocation": 1,
                    "planned_horizon": 20,
                    "actual_horizon": 1,
                    "finalization_reason": "trigger",
                },
            ]
        )
        return np.array([1.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)


class _Episode:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def reset(self) -> object:
        return object()

    def step(self, action: np.ndarray) -> object:
        assert action[0] == np.float32(1.2)
        self.order.append("env_step")
        return type("Step", (), {"observation": object(), "success": True, "done": True})()

    def close(self) -> None:
        return None


class _Backend:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def open_episode(self, *args: object) -> _Episode:
        del args
        return _Episode(self.order)


def test_trigger_event_is_complete_and_built_before_env_step(tmp_path: Path) -> None:
    module = _module()
    order: list[str] = []
    original_builder = module._adaptive_v2_trigger_event

    def recording_builder(*args: object, **kwargs: object) -> dict[str, object]:
        order.append("trigger_event_built")
        return original_builder(*args, **kwargs)

    module._adaptive_v2_trigger_event = recording_builder
    condition = {
        "name": V2_NAME,
        "fixed_h": 20,
        "safety_enabled": True,
        "replan_after_safety_violation": False,
        "adaptive_v2_trigger": True,
        "clip_actions": False,
    }
    config = {
        "suite": "libero_spatial",
        "episode_cap": 280,
        "inference_seed_namespace": "adaptive-v2-heldout-v1|libero_spatial",
    }
    result = module._run_episode(
        output_dir=tmp_path,
        config=config,
        condition=condition,
        trial={"task_id": 3, "seed": 1004, "initial_state_id": 4, "episode_index": 4},
        backend=_Backend(order),
        policy=_Policy(order),
        inference_seed_setter=lambda _: None,
    )

    assert order == ["policy_select", "trigger_event_built", "env_step"]
    assert result["range_clips"] == 0
    assert result["adaptive_v2_trigger_events"] == [
        {
            "schema_version": 1,
            "task_id": 3,
            "initial_state_id": 4,
            "seed": 1004,
            "environment_seed": 1004,
            "inference_seed": result["inference_seed"],
            "model_call_index": 1,
            "env_step": 1,
            "evaluation_order": "trigger_evaluated_before_env_step",
            "dimensions": [0],
            "raw_values": [1.2],
            "bounds": [1.0],
            "excess": [0.2],
            "severity": [0.1],
            "persistence_counts": [1],
            "violations": [{
                "dimension": 0, "raw_value": 1.2, "bound": 1.0,
                "excess": 0.2, "severity": 0.1, "persistence_count": 1,
            }],
            "discarded_actions": 19,
            "buffer_entries_cleared": 49,
            "horizon_before": 20,
            "horizon_after": 1,
            "recovery_horizon": 20,
            "state_before": "monitoring_h20",
            "state_after": "fallback_h1_pending",
            "immediate_dimensions": [0],
            "persistent_dimensions": [],
            "cooldown_h20_actions": 20,
        }
    ]
