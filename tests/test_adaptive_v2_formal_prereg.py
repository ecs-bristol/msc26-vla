from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "a40dc27c3d78dd9c8f647db560ad4cb58510eb61"
PILOT_SCRIPT = ROOT / "scripts" / "analysis" / "libero_spatial_paired_pilot.py"
ANALYSIS_SCRIPT = ROOT / "scripts" / "analysis" / "adaptive_v2_formal_analysis.py"
CONFIG = ROOT / "configs" / "evaluation" / "libero_spatial_adaptive_v2_formal_heldout.yaml"
SEEDS = ROOT / "configs" / "evaluation" / "libero_spatial_adaptive_v2_confirmatory_seed_manifest.json"
BLOCK_A = ROOT / "configs" / "evaluation" / "libero_spatial_adaptive_v2_formal_block_a.json"
BLOCK_B = ROOT / "configs" / "evaluation" / "libero_spatial_adaptive_v2_formal_block_b.json"
SCHEDULE = ROOT / "configs" / "evaluation" / "libero_spatial_adaptive_v2_formal_schedule.json"
BASE_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
STATIC = "Static-H20"
ADAPTIVE = "Adaptive-v2a-H20→H1"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pilot():
    return _load(PILOT_SCRIPT, "adaptive_v2_formal_pilot_test")


def _analysis():
    return _load(ANALYSIS_SCRIPT, "adaptive_v2_formal_analysis_test")


def _snapshot(tmp_path: Path, revision: str) -> Path:
    path = tmp_path / "hf" / "models--test" / "snapshots" / revision
    path.mkdir(parents=True)
    return path


def _function_hash(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return hashlib.sha256(segment.encode()).hexdigest()


def test_coverage_and_formal_prep_do_not_change_frozen_action_path() -> None:
    critical_files = [
        "src/libero_platform/policies/adaptive_v2_trigger.py",
        "src/libero_platform/policies/fixed_h_action_buffer.py",
        "plugins/lerobot_policy_smolvla_adaptive/src/lerobot_policy_smolvla_adaptive/modeling_smolvla_adaptive.py",
        "plugins/lerobot_policy_smolvla_adaptive/src/lerobot_policy_smolvla_adaptive/configuration_smolvla_adaptive.py",
    ]
    for relative in critical_files:
        frozen = subprocess.check_output(
            ["git", "show", f"{BASE_COMMIT}:{relative}"], cwd=ROOT
        )
        assert (ROOT / relative).read_bytes().replace(b"\r\n", b"\n") == frozen.replace(
            b"\r\n", b"\n"
        )
    frozen_evaluator = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:scripts/analysis/libero_spatial_paired_pilot.py"],
        cwd=ROOT,
        text=True,
    )
    current_evaluator = PILOT_SCRIPT.read_text()
    for function in ("_adaptive_v2_trigger_event", "_run_episode"):
        assert _function_hash(current_evaluator, function) == _function_hash(
            frozen_evaluator, function
        )


def test_formal_dry_run_has_only_two_matched_conditions(tmp_path: Path) -> None:
    pilot = _pilot()
    output = tmp_path / "formal-dry-run"
    result = pilot.materialize_dry_run(
        config_path=CONFIG,
        output_dir=output,
        base_snapshot_path=str(_snapshot(tmp_path, BASE_REVISION)),
        vlm_snapshot_path=str(_snapshot(tmp_path, VLM_REVISION)),
    )
    assert result["planned_episodes"] == 100
    resolved, trials = pilot._read_execution_inputs(output)
    assert resolved["formal_adaptive_v2_confirmatory"] is True
    assert resolved["inference_seed_namespace"] == "adaptive-v2-confirmatory-v1|libero_spatial"
    assert [item["name"] for item in resolved["conditions"]] == [STATIC, ADAPTIVE]
    static, adaptive = resolved["conditions"]
    assert {key: value for key, value in static.items() if key not in {"name", "adaptive_v2_trigger"}} == {
        key: value for key, value in adaptive.items() if key not in {"name", "adaptive_v2_trigger"}
    }
    assert static["adaptive_v2_trigger"] is False
    assert adaptive["adaptive_v2_trigger"] is True
    manifest = json.loads((output / "paired_manifest.json").read_text())
    frozen_seeds = json.loads(SEEDS.read_text())["records"]
    assert manifest["inference_seed_records"] == frozen_seeds
    assert len(trials) == 50
    episodes = [json.loads(path.read_text()) for path in (output / "episodes").rglob("*.json")]
    assert len(episodes) == 100 and all(item["status"] == "planned_dry_run" for item in episodes)


def test_blocks_are_exact_deterministic_hash_halves_and_schedule_is_serial() -> None:
    a = json.loads(BLOCK_A.read_text())
    b = json.loads(BLOCK_B.read_text())
    namespace = a["assignment_namespace"]
    ranked = []
    for task in range(10):
        for state in range(5):
            seed = 1000 + state
            digest = hashlib.sha256(f"{namespace}|{task}|{seed}|{state}".encode()).hexdigest()
            ranked.append((digest, {"task_id": task, "seed": seed, "initial_state_id": state}))
    ranked.sort(key=lambda item: item[0])
    assert a["pairing_keys"] == [item[1] for item in ranked[:25]]
    assert b["pairing_keys"] == [item[1] for item in ranked[25:]]
    assert len({tuple(item.values()) for item in a["pairing_keys"] + b["pairing_keys"]}) == 50
    schedule = json.loads(SCHEDULE.read_text())
    assert schedule["parallel_execution_allowed"] is False
    assert schedule["phases"] == [
        {"phase": 1, "block": "A", "condition": STATIC},
        {"phase": 2, "block": "A", "condition": ADAPTIVE},
        {"phase": 3, "block": "B", "condition": ADAPTIVE},
        {"phase": 4, "block": "B", "condition": STATIC},
    ]


def test_new_seeds_are_disjoint_from_every_observed_namespace() -> None:
    payload = json.loads(SEEDS.read_text())
    new = {item["inference_seed"] for item in payload["records"]}

    def derive(namespace: str) -> set[int]:
        result = set()
        for task in range(10):
            for state in range(5):
                raw = f"{namespace}|{task}|{1000 + state}|{state}".encode()
                result.add(
                    int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
                    & ((1 << 63) - 1)
                )
        return result

    assert len(new) == 50
    assert new.isdisjoint(derive("libero_spatial"))
    assert new.isdisjoint(derive("adaptive-v2-heldout-v1|libero_spatial"))


class _PoisonOutcome:
    def __bool__(self) -> bool:
        raise AssertionError("outcome was accessed before completeness")

    def __eq__(self, other: object) -> bool:
        del other
        raise AssertionError("outcome was compared before completeness")


def test_incomplete_analysis_cannot_access_outcome_fields() -> None:
    module = _analysis()
    record = {
        "condition": STATIC,
        "task_id": 0,
        "seed": 1000,
        "initial_state_id": 0,
        "success_at_280": _PoisonOutcome(),
        "success_step": _PoisonOutcome(),
        "termination_reason": _PoisonOutcome(),
    }
    wrapped = module.OutcomeLockedEpisode(record)
    for field in module.FORBIDDEN_BEFORE_COMPLETE:
        with pytest.raises(module.AnalysisLockedError):
            wrapped.get(field)
    source = inspect.getsource(module.validate_complete_without_outcomes)
    constants = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert module.FORBIDDEN_BEFORE_COMPLETE.isdisjoint(constants)
    manifest = {
        "trials": [
            {"task_id": task, "seed": 1000 + state, "initial_state_id": state}
            for task in range(10)
            for state in range(5)
        ],
        "inference_seed_records": [
            {
                "task_id": task,
                "environment_seed": 1000 + state,
                "initial_state_id": state,
                "inference_seed": task * 10 + state,
            }
            for task in range(10)
            for state in range(5)
        ],
    }
    with pytest.raises(module.AnalysisLockedError, match="episode set incomplete"):
        module.validate_complete_without_outcomes(
            raw_records=[record], manifest=manifest, summary_rows=[]
        )


def _formal_record(condition: str, task: int, state: int) -> dict[str, object]:
    adaptive = condition == ADAPTIVE
    rescue = adaptive and task == 0 and state == 0
    static_success = task != 0 or state != 0
    success = rescue or static_success
    event = []
    if rescue:
        event = [
            {
                "task_id": task,
                "seed": 1000 + state,
                "initial_state_id": state,
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
            }
        ]
        realized = [5, 1, 14]
        generated = 150
        horizon_tail = 109
        trigger_tail = 15
        terminal_tail = 6
    elif adaptive:
        realized = [10, 10]
        generated = 100
        horizon_tail = 60
        trigger_tail = 0
        terminal_tail = 20
    else:
        realized = [20]
        generated = 50
        horizon_tail = 30
        trigger_tail = 0
        terminal_tail = 0
    return {
        "status": "completed",
        "condition": condition,
        "condition_config": {
            "name": condition,
            "fixed_h": 20,
            "safety_enabled": True,
            "replan_after_safety_violation": False,
            "adaptive_v2_trigger": adaptive,
            "clip_actions": False,
        },
        "task_id": task,
        "seed": 1000 + state,
        "initial_state_id": state,
        "environment_seed": 1000 + state,
        "inference_seed": task * 10 + state,
        "executed_env_steps": 20,
        "wall_time_to_terminal_s": 2.0 + adaptive,
        "model_invocations": len(realized),
        "model_inference_time_s": 1.0 + adaptive,
        "realized_actions_per_call": realized,
        "generated_actions": generated,
        "unused_actions": generated - 20,
        "chunk_utilization": 20 / generated,
        "horizon_tail_discarded_actions": horizon_tail,
        "trigger_tail_discarded_actions": trigger_tail,
        "terminal_tail_unused_actions": terminal_tail,
        "range_violations": int(rescue),
        "range_violation_dimension_counts": {
            "0": int(rescue), "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0
        },
        "range_violation_max_excess_by_dimension": {
            "0": 0.2 if rescue else 0.0,
            "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0,
        },
        "trigger_range_violations": int(rescue),
        "gripper_only_range_violations": 0,
        "range_clips": 0,
        "buffer_discards": int(rescue),
        "mean_actual_horizon": 20 / len(realized),
        "action_trace_sha256": "a" * 64,
        "git_sha": "f" * 40,
        "resolved_config_path": "/tmp/resolved.json",
        "adaptive_v2_trigger_events": event,
        "success_at_280": success,
        "success_step": 20 if success else None,
        "termination_reason": "success" if success else "max_steps",
    }


def test_frozen_analysis_reports_preregistered_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _analysis()
    monkeypatch.setattr(module, "BOOTSTRAP_REPLICATES", 1000)
    manifest = {
        "trials": [
            {"task_id": task, "seed": 1000 + state, "initial_state_id": state}
            for task in range(10)
            for state in range(5)
        ],
        "inference_seed_records": [
            {
                "task_id": task,
                "environment_seed": 1000 + state,
                "initial_state_id": state,
                "inference_seed": task * 10 + state,
            }
            for task in range(10)
            for state in range(5)
        ],
    }
    records = [
        _formal_record(condition, task, state)
        for condition in (STATIC, ADAPTIVE)
        for task in range(10)
        for state in range(5)
    ]
    summary_rows = [
        {
            field: (
                json.dumps(record[field], separators=(",", ":"), sort_keys=True)
                if isinstance(record[field], (dict, list))
                else str(record[field])
            )
            for field in module.NON_OUTCOME_SUMMARY_FIELDS
        }
        for record in records
    ]
    gate = module.validate_complete_without_outcomes(
        raw_records=records, manifest=manifest, summary_rows=summary_rows
    )
    report = module.analyze_unlocked_outcomes(records, gate)
    assert report["flip_table"] == {
        "both_success": 49,
        "adaptive_only": 1,
        "static_only": 0,
        "both_fail": 0,
    }
    assert report["trigger_count"] == 1
    assert report["trigger_casebook"][0]["effect"] == "rescue"
    assert report["decision_rule"]["added_calls_have_demonstrated_value"] is True
