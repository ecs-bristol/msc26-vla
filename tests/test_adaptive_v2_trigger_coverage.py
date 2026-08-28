from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COVERAGE_SCRIPT = ROOT / "scripts" / "analysis" / "adaptive_v2_trigger_coverage.py"
PILOT_SCRIPT = ROOT / "scripts" / "analysis" / "libero_spatial_paired_pilot.py"
COVERAGE_CONFIG = (
    ROOT / "configs" / "evaluation" / "libero_spatial_adaptive_v2_trigger_coverage.yaml"
)
CANDIDATES = (
    ROOT
    / "configs"
    / "evaluation"
    / "libero_spatial_adaptive_v2_trigger_coverage_candidates.json"
)
CONFIRMATORY = (
    ROOT
    / "configs"
    / "evaluation"
    / "libero_spatial_adaptive_v2_confirmatory_seed_manifest.json"
)
BASE_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
V2_NAME = "Adaptive-v2a-H20→H1"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _coverage():
    return _load(COVERAGE_SCRIPT, "adaptive_v2_coverage_test")


def _pilot():
    return _load(PILOT_SCRIPT, "adaptive_v2_coverage_pilot_test")


def _snapshot(tmp_path: Path, revision: str) -> Path:
    path = tmp_path / "hf" / "models--test" / "snapshots" / revision
    path.mkdir(parents=True)
    return path


def _record(*, task: int, non_gripper: int, triggers: int) -> dict[str, object]:
    return {
        "status": "completed",
        "condition": "Adaptive-H20→H1",
        "task_id": task,
        "seed": 1000,
        "initial_state_id": 0,
        "inference_seed": 123 + task,
        "range_violation_dimension_counts": {
            "0": non_gripper,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 0,
            "6": 4,
        },
        "trigger_range_violations": triggers,
        "success_at_280": object(),
        "success_step": object(),
        "termination_reason": object(),
    }


def test_outcome_firewall_raises_and_selection_source_is_success_blind() -> None:
    module = _coverage()
    wrapped = module.OutcomeBlindEpisode(_record(task=0, non_gripper=1, triggers=1))
    for field in module.FORBIDDEN_OUTCOME_FIELDS:
        with pytest.raises(module.OutcomeFieldAccessError):
            _ = wrapped[field]
        with pytest.raises(module.OutcomeFieldAccessError):
            wrapped.get(field)
        with pytest.raises(module.OutcomeFieldAccessError):
            _ = field in wrapped

    source = inspect.getsource(module.select_candidate_pairing_keys)
    tree = ast.parse(source)
    string_constants = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert module.FORBIDDEN_OUTCOME_FIELDS.isdisjoint(string_constants)
    assert module.select_candidate_pairing_keys(
        [_record(task=0, non_gripper=1, triggers=1)]
    )[0]["task_id"] == 0


def test_selection_uses_union_of_non_gripper_violation_and_trigger() -> None:
    module = _coverage()
    records = [
        _record(task=0, non_gripper=1, triggers=0),
        _record(task=1, non_gripper=0, triggers=1),
        _record(task=2, non_gripper=0, triggers=0),
    ]
    selected = module.select_candidate_pairing_keys(records)
    assert [item["task_id"] for item in selected] == [0, 1]


def test_development_dry_run_and_pairing_filter_are_exact(tmp_path: Path) -> None:
    pilot = _pilot()
    output = tmp_path / "dry-run"
    result = pilot.materialize_dry_run(
        config_path=COVERAGE_CONFIG,
        output_dir=output,
        base_snapshot_path=str(_snapshot(tmp_path, BASE_REVISION)),
        vlm_snapshot_path=str(_snapshot(tmp_path, VLM_REVISION)),
    )
    assert result["planned_episodes"] == 300
    resolved, trials = pilot._read_execution_inputs(output)
    assert resolved["development_trigger_coverage"] is True
    assert resolved["inference_seed_namespace"] == "libero_spatial"
    selected = pilot._load_pairing_key_filter(CANDIDATES)
    assert selected == {
        (0, 1002, 2),
        (1, 1001, 1),
        (5, 1001, 1),
        (5, 1003, 3),
        (6, 1000, 0),
        (6, 1002, 2),
    }
    assert len([trial for trial in trials if (trial["task_id"], trial["seed"], trial["initial_state_id"]) in selected]) == 6
    records = [json.loads(path.read_text()) for path in (output / "episodes").rglob("*.json")]
    assert len(records) == 300 and all(item["status"] == "planned_dry_run" for item in records)
    with pytest.raises(ValueError, match="may execute only Adaptive-v2a"):
        pilot.execute_pilot(
            output_dir=output,
            backend_factory=lambda: pytest.fail("backend must not be constructed"),
            policy_factory=lambda *_: pytest.fail("policy must not be constructed"),
            condition_names={"Static-H20"},
            pairing_keys=selected,
            pairing_key_filter_path=CANDIDATES,
        )


def test_pairing_filter_rejects_outcome_fields(tmp_path: Path) -> None:
    pilot = _pilot()
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "selection_role": "trigger_coverage_development",
                "pairing_keys": [
                    {
                        "task_id": 0,
                        "seed": 1000,
                        "initial_state_id": 0,
                        "success_at_280": True,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="outcome fields are forbidden"):
        pilot._load_pairing_key_filter(path)


def _episode(*, trigger: bool) -> dict[str, object]:
    realized = [5, 1, 20] if trigger else [20]
    events = []
    if trigger:
        events = [
            {
                "model_call_index": 1,
                "env_step": 5,
                "evaluation_order": "trigger_evaluated_before_env_step",
                "dimensions": [0],
                "raw_values": [1.2],
                "bounds": [1.0],
                "excess": [0.2],
                "severity": [0.1],
                "persistence_counts": [1],
                "discarded_actions": 15,
                "horizon_before": 20,
                "horizon_after": 1,
                "recovery_horizon": 20,
                "state_before": "monitoring_h20",
                "state_after": "fallback_h1_pending",
                "cooldown_h20_actions": 20,
            }
        ]
    generated = 50 * len(realized)
    executed = sum(realized)
    horizon_tail = 109 if trigger else 30
    trigger_tail = 15 if trigger else 0
    return {
        "status": "completed",
        "condition": V2_NAME,
        "condition_config": {
            "name": V2_NAME,
            "fixed_h": 20,
            "safety_enabled": True,
            "replan_after_safety_violation": False,
            "adaptive_v2_trigger": True,
            "clip_actions": False,
        },
        "task_id": 0,
        "seed": 1000,
        "initial_state_id": 0,
        "environment_seed": 1000,
        "inference_seed": 123,
        "executed_env_steps": executed,
        "model_invocations": len(realized),
        "model_inference_time_s": 1.0,
        "realized_actions_per_call": realized,
        "generated_actions": generated,
        "unused_actions": generated - executed,
        "chunk_utilization": executed / generated,
        "horizon_tail_discarded_actions": horizon_tail,
        "trigger_tail_discarded_actions": trigger_tail,
        "terminal_tail_unused_actions": 0,
        "range_violations": int(trigger),
        "range_violation_dimension_counts": {
            "0": int(trigger), "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0
        },
        "range_violation_max_excess_by_dimension": {
            "0": 0.2 if trigger else 0.0,
            "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0,
        },
        "trigger_range_violations": int(trigger),
        "gripper_only_range_violations": 0,
        "range_clips": 0,
        "buffer_discards": int(trigger),
        "mean_actual_horizon": executed / len(realized),
        "action_trace_sha256": "a" * 64,
        "git_sha": "a40dc27c3d78dd9c8f647db560ad4cb58510eb61",
        "resolved_config_path": "/tmp/resolved_config.json",
        "adaptive_v2_trigger_events": events,
        "success_at_280": object(),
        "success_step": object(),
        "termination_reason": object(),
    }


def _coverage_fixture(tmp_path: Path, *, trigger: bool) -> tuple[Path, Path]:
    module = _coverage()
    output = tmp_path / ("trigger" if trigger else "inactive")
    episode = _episode(trigger=trigger)
    path = output / "episodes" / "adaptive-v2a-h20-to-h1" / "task_00_seed_1000_state_0.json"
    path.parent.mkdir(parents=True)
    serializable = dict(episode)
    for field in module.FORBIDDEN_OUTCOME_FIELDS:
        serializable[field] = None
    path.write_text(json.dumps(serializable))
    candidate = tmp_path / ("trigger-candidate.json" if trigger else "inactive-candidate.json")
    candidate.write_text(
        json.dumps(
            {
                "selection_role": "trigger_coverage_development",
                "pairing_keys": [{"task_id": 0, "seed": 1000, "initial_state_id": 0}],
            }
        )
    )
    (output / "execution_provenance.json").write_text(
        json.dumps(
            {
                "git_sha": "a40dc27c3d78dd9c8f647db560ad4cb58510eb61",
                "selected_conditions": [V2_NAME],
                "pairing_key_filter_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            }
        )
    )
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.SUMMARY_VALIDATION_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                field: (
                    json.dumps(serializable[field], separators=(",", ":"), sort_keys=True)
                    if isinstance(serializable[field], (dict, list))
                    else serializable[field]
                )
                for field in module.SUMMARY_VALIDATION_FIELDS
            }
        )
    return output, candidate


def test_validator_requires_complete_real_trigger_chain(tmp_path: Path) -> None:
    module = _coverage()
    output, candidate = _coverage_fixture(tmp_path, trigger=True)
    result = module.validate_coverage_output(output, candidate)
    assert result["verdict"] == "V2A_TRIGGER_COVERAGE_PASS"
    assert result["trigger_count"] == result["complete_chain_count"] == 1
    assert result["success_fields_used"] is False


def test_validator_classifies_zero_trigger_as_inactive(tmp_path: Path) -> None:
    module = _coverage()
    output, candidate = _coverage_fixture(tmp_path, trigger=False)
    result = module.validate_coverage_output(output, candidate)
    assert result["verdict"] == "V2A_TRIGGER_INACTIVE"
    assert result["trigger_count"] == result["complete_chain_count"] == 0


def test_confirmatory_seed_manifest_is_new_and_deterministic() -> None:
    payload = json.loads(CONFIRMATORY.read_text())
    records = payload["records"]
    assert len(records) == 50
    new = {item["inference_seed"] for item in records}
    assert len(new) == 50

    def derive(namespace: str, task: int, seed: int, state: int) -> int:
        raw = f"{namespace}|{task}|{seed}|{state}".encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)

    expected = {
        derive(
            payload["inference_seed_namespace"],
            item["task_id"],
            item["environment_seed"],
            item["initial_state_id"],
        )
        for item in records
    }
    legacy = {
        derive("libero_spatial", task, 1000 + state, state)
        for task in range(10)
        for state in range(5)
    }
    observed_v2_namespace = {
        derive("adaptive-v2-heldout-v1|libero_spatial", task, 1000 + state, state)
        for task in range(10)
        for state in range(5)
    }
    assert new == expected
    assert new.isdisjoint(legacy)
    assert new.isdisjoint(observed_v2_namespace)
