#!/usr/bin/env python3
"""Outcome-blind selection and validation for Adaptive-v2a trigger coverage.

This development-only utility never selects from rewards or outcomes.  Its
candidate set is the union of frozen Adaptive-v1 episodes with a non-gripper
range violation or trigger telemetry.  It also validates the resulting v2a
mechanism smoke without using outcome labels as acceptance criteria.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


FORBIDDEN_OUTCOME_FIELDS = frozenset(
    {"success_at_280", "success_step", "termination_reason"}
)
V1_CONDITION = "Adaptive-H20→H1"
V2A_CONDITION = "Adaptive-v2a-H20→H1"
SELECTION_ROLE = "trigger_coverage_development"
SUMMARY_VALIDATION_FIELDS = (
    "condition",
    "task_id",
    "seed",
    "initial_state_id",
    "environment_seed",
    "inference_seed",
    "executed_env_steps",
    "model_invocations",
    "model_inference_time_s",
    "realized_actions_per_call",
    "generated_actions",
    "unused_actions",
    "chunk_utilization",
    "horizon_tail_discarded_actions",
    "trigger_tail_discarded_actions",
    "terminal_tail_unused_actions",
    "range_violations",
    "range_violation_dimension_counts",
    "range_violation_max_excess_by_dimension",
    "trigger_range_violations",
    "gripper_only_range_violations",
    "range_clips",
    "buffer_discards",
    "mean_actual_horizon",
    "action_trace_sha256",
    "git_sha",
    "resolved_config_path",
)


class OutcomeFieldAccessError(RuntimeError):
    """Raised on any attempted outcome-label access through the firewall."""


class OutcomeBlindEpisode(Mapping[str, Any]):
    """Read-only mapping that makes accidental outcome access fail loudly."""

    def __init__(self, source: Mapping[str, Any]) -> None:
        self._source = source

    @staticmethod
    def _guard(key: object) -> None:
        if key in FORBIDDEN_OUTCOME_FIELDS:
            raise OutcomeFieldAccessError(f"outcome field access is forbidden: {key}")

    def __getitem__(self, key: str) -> Any:
        self._guard(key)
        return self._source[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key in self._source if key not in FORBIDDEN_OUTCOME_FIELDS)

    def __len__(self) -> int:
        return sum(key not in FORBIDDEN_OUTCOME_FIELDS for key in self._source)

    def __contains__(self, key: object) -> bool:
        self._guard(key)
        return key in self._source

    def get(self, key: str, default: Any = None) -> Any:
        self._guard(key)
        return self._source.get(key, default)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["task_id"]),
        int(record["seed"]),
        int(record["initial_state_id"]),
    )


def select_candidate_pairing_keys(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select all v1 keys with non-gripper violation/trigger telemetry only."""

    selected: dict[tuple[int, int, int], dict[str, Any]] = {}
    for source in records:
        record = OutcomeBlindEpisode(source)
        if record.get("status") != "completed" or record.get("condition") != V1_CONDITION:
            continue
        counts = record["range_violation_dimension_counts"]
        if not isinstance(counts, dict) or set(counts) != {str(i) for i in range(7)}:
            raise ValueError("v1 dimension-count telemetry is incomplete")
        non_gripper_count = sum(int(counts[str(index)]) for index in range(6))
        trigger_count = int(record["trigger_range_violations"])
        if non_gripper_count == 0 and trigger_count == 0:
            continue
        pairing_key = _key(record)
        if pairing_key in selected:
            raise ValueError(f"duplicate frozen v1 pairing key: {pairing_key}")
        selected[pairing_key] = {
            "task_id": pairing_key[0],
            "seed": pairing_key[1],
            "initial_state_id": pairing_key[2],
            "inference_seed": int(record["inference_seed"]),
            "non_gripper_violation_count": non_gripper_count,
            "v1_trigger_count": trigger_count,
            "non_gripper_dimension_counts": {
                str(index): int(counts[str(index)]) for index in range(6)
            },
        }
    return [selected[key] for key in sorted(selected)]


def load_frozen_v1_candidates(source_dir: Path) -> list[dict[str, Any]]:
    episode_root = source_dir.resolve() / "episodes"
    paths = sorted(episode_root.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no frozen v1 episode JSON files under {episode_root}")
    return select_candidate_pairing_keys([_read_json(path) for path in paths])


def candidate_document(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    candidates = load_frozen_v1_candidates(source_dir)
    if not candidates:
        raise ValueError("frozen v1 telemetry produced no trigger-coverage candidates")
    pairing_keys = [
        {
            "task_id": item["task_id"],
            "seed": item["seed"],
            "initial_state_id": item["initial_state_id"],
        }
        for item in candidates
    ]
    return {
        "schema_version": 1,
        "selection_role": SELECTION_ROLE,
        "formal_evaluation": False,
        "source_dir": str(source_dir),
        "source_manifest_sha256": _sha256(source_dir / "paired_manifest.json"),
        "source_condition": V1_CONDITION,
        "selection_rule": (
            "all completed frozen-v1 episodes with any dimension-0..5 range "
            "violation or trigger_range_violations > 0"
        ),
        "forbidden_selection_fields": sorted(FORBIDDEN_OUTCOME_FIELDS),
        "candidate_count": len(candidates),
        "pairing_keys": pairing_keys,
        "selection_evidence": candidates,
    }


def verify_candidate_document(source_dir: Path, candidate_path: Path) -> dict[str, Any]:
    expected = candidate_document(source_dir)
    observed = _read_json(candidate_path.resolve())
    if observed != expected:
        raise ValueError("candidate artifact does not match outcome-blind frozen-v1 selection")
    return {
        "status": "V2A_TRIGGER_COVERAGE_SELECTION_PASS",
        "candidate_count": expected["candidate_count"],
        "candidate_sha256": _sha256(candidate_path.resolve()),
    }


def _summary_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def _episode_path(output_dir: Path, item: Mapping[str, Any]) -> Path:
    return (
        output_dir
        / "episodes"
        / "adaptive-v2a-h20-to-h1"
        / (
            f"task_{int(item['task_id']):02d}_seed_{int(item['seed'])}"
            f"_state_{int(item['initial_state_id'])}.json"
        )
    )


def _event_validation(
    event: Mapping[str, Any], realized: list[int]
) -> tuple[list[str], bool]:
    errors: list[str] = []
    required = {
        "model_call_index",
        "env_step",
        "evaluation_order",
        "dimensions",
        "raw_values",
        "bounds",
        "excess",
        "severity",
        "persistence_counts",
        "discarded_actions",
        "horizon_before",
        "horizon_after",
        "recovery_horizon",
        "state_before",
        "state_after",
        "cooldown_h20_actions",
    }
    missing = sorted(required - set(event))
    if missing:
        return [f"missing event fields: {missing}"], False
    if event["evaluation_order"] != "trigger_evaluated_before_env_step":
        errors.append("trigger event was not persisted before env.step")
    dimensions = [int(value) for value in event["dimensions"]]
    columns = [
        dimensions,
        list(event["raw_values"]),
        list(event["bounds"]),
        list(event["excess"]),
        list(event["severity"]),
        list(event["persistence_counts"]),
    ]
    if not dimensions or any(dimension == 6 for dimension in dimensions):
        errors.append("trigger event includes no non-gripper dimension")
    if len({len(column) for column in columns}) != 1:
        errors.append("per-dimension trigger columns have unequal lengths")
    else:
        for raw, bound, excess, severity, persistence in zip(*columns[1:], strict=True):
            expected_bound = 1.0 if float(raw) > 0.0 else -1.0
            expected_excess = max(abs(float(raw)) - 1.0, 0.0)
            if float(bound) != expected_bound:
                errors.append("trigger bound does not match raw action sign")
            if abs(float(excess) - expected_excess) > 1e-6:
                errors.append("trigger excess does not match the action bound")
            if abs(float(severity) - expected_excess / 2.0) > 1e-6:
                errors.append("trigger severity does not equal excess/action-span")
            if int(persistence) < 1:
                errors.append("trigger persistence count is invalid")
    if (
        int(event["horizon_before"]) != 20
        or int(event["horizon_after"]) != 1
        or int(event["recovery_horizon"]) != 20
        or int(event["cooldown_h20_actions"]) != 20
        or event["state_before"] != "monitoring_h20"
        or event["state_after"] != "fallback_h1_pending"
    ):
        errors.append("trigger event has an invalid H20/H1 state transition")
    call_index = int(event["model_call_index"])
    zero_based = call_index - 1
    if zero_based < 0 or zero_based >= len(realized):
        errors.append("trigger model-call index is outside realized call telemetry")
        return errors, False
    expected_discard = 20 - int(realized[zero_based])
    if int(event["discarded_actions"]) != expected_discard:
        errors.append("trigger discard does not match the realized trigger-call tail")
    fallback_index = zero_based + 1
    cooldown_index = zero_based + 2
    full_chain = (
        fallback_index < len(realized)
        and int(realized[fallback_index]) == 1
        and cooldown_index < len(realized)
        and int(realized[cooldown_index]) == 20
    )
    return errors, full_chain


def validate_coverage_output(
    output_dir: Path, candidate_path: Path
) -> dict[str, Any]:
    """Validate telemetry without consulting reward, success, or termination labels."""

    output_dir = output_dir.resolve()
    candidates = _read_json(candidate_path.resolve())
    if candidates.get("selection_role") != SELECTION_ROLE:
        raise ValueError("candidate artifact is not development trigger coverage")
    pairing_keys = candidates.get("pairing_keys")
    if not isinstance(pairing_keys, list) or not pairing_keys:
        raise ValueError("candidate artifact has no pairing keys")
    provenance = _read_json(output_dir / "execution_provenance.json")
    expected_git_sha = provenance.get("git_sha")
    if not isinstance(expected_git_sha, str) or len(expected_git_sha) != 40:
        raise ValueError("execution provenance has no valid Git SHA")
    if provenance.get("selected_conditions") != [V2A_CONDITION]:
        raise ValueError("execution provenance is not restricted to Adaptive-v2a")
    if provenance.get("pairing_key_filter_sha256") != _sha256(candidate_path.resolve()):
        raise ValueError("execution provenance pairing-key filter hash mismatch")
    summary_rows = list(csv.DictReader((output_dir / "summary.csv").open(encoding="utf-8")))
    summary_map = {
        (
            row["condition"],
            int(row["task_id"]),
            int(row["seed"]),
            int(row["initial_state_id"]),
        ): row
        for row in summary_rows
    }
    errors: list[str] = []
    episode_reports: list[dict[str, Any]] = []
    trigger_count = 0
    complete_chain_count = 0
    for item in pairing_keys:
        path = _episode_path(output_dir, item)
        if not path.is_file():
            errors.append(f"missing candidate episode: {path.name}")
            continue
        episode = OutcomeBlindEpisode(_read_json(path))
        if episode.get("status") != "completed":
            errors.append(f"candidate episode is not completed: {path.name}")
            continue
        if episode["condition"] != V2A_CONDITION or episode["git_sha"] != expected_git_sha:
            errors.append(f"candidate identity mismatch: {path.name}")
        expected_condition = {
            "name": V2A_CONDITION,
            "fixed_h": 20,
            "safety_enabled": True,
            "replan_after_safety_violation": False,
            "adaptive_v2_trigger": True,
            "clip_actions": False,
        }
        if episode["condition_config"] != expected_condition:
            errors.append(f"candidate protocol mismatch: {path.name}")
        realized = [int(value) for value in episode["realized_actions_per_call"]]
        accounting = {
            "generated": int(episode["generated_actions"])
            == 50 * int(episode["model_invocations"]),
            "realized": sum(realized) == int(episode["executed_env_steps"]),
            "calls": len(realized) == int(episode["model_invocations"]),
            "unused": int(episode["unused_actions"])
            == int(episode["generated_actions"]) - int(episode["executed_env_steps"]),
            "tails": int(episode["unused_actions"])
            == int(episode["horizon_tail_discarded_actions"])
            + int(episode["trigger_tail_discarded_actions"])
            + int(episode["terminal_tail_unused_actions"]),
        }
        if not all(accounting.values()):
            errors.append(f"candidate accounting mismatch: {path.name}")
        if int(episode["range_clips"]) != 0:
            errors.append(f"candidate applied action clipping: {path.name}")
        events = list(episode.get("adaptive_v2_trigger_events", []))
        event_reports = []
        discarded_sum = 0
        for event in events:
            event_errors, full_chain = _event_validation(event, realized)
            event_reports.append(
                {
                    "model_call_index": event.get("model_call_index"),
                    "env_step": event.get("env_step"),
                    "errors": event_errors,
                    "full_chain": full_chain,
                }
            )
            errors.extend(f"{path.name}: {message}" for message in event_errors)
            trigger_count += 1
            complete_chain_count += int(full_chain and not event_errors)
            discarded_sum += int(event.get("discarded_actions", 0))
        if int(episode["trigger_range_violations"]) != len(events):
            errors.append(f"trigger aggregate/event count mismatch: {path.name}")
        if int(episode["buffer_discards"]) != len(events):
            errors.append(f"buffer discard/event count mismatch: {path.name}")
        if int(episode["trigger_tail_discarded_actions"]) != discarded_sum:
            errors.append(f"trigger tail/event sum mismatch: {path.name}")
        row = summary_map.get((V2A_CONDITION, *_key(item)))
        if row is None:
            errors.append(f"summary row missing: {path.name}")
        else:
            for field in SUMMARY_VALIDATION_FIELDS:
                if row[field] != _summary_value(episode.get(field)):
                    errors.append(f"summary mismatch {field}: {path.name}")
        episode_reports.append(
            {
                "task_id": item["task_id"],
                "seed": item["seed"],
                "initial_state_id": item["initial_state_id"],
                "trigger_count": len(events),
                "event_reports": event_reports,
                "accounting": accounting,
            }
        )
    selected_keys = {
        (V2A_CONDITION, int(item["task_id"]), int(item["seed"]), int(item["initial_state_id"]))
        for item in pairing_keys
    }
    extra_completed = []
    for path in (output_dir / "episodes").rglob("*.json"):
        raw = _read_json(path)
        if raw.get("status") in {"completed", "failed"}:
            observed = (
                raw.get("condition"),
                int(raw["task_id"]),
                int(raw["seed"]),
                int(raw["initial_state_id"]),
            )
            if observed not in selected_keys:
                extra_completed.append(observed)
    if extra_completed:
        errors.append(f"non-candidate terminal episodes exist: {sorted(extra_completed)}")
    all_candidates_completed = len(episode_reports) == len(pairing_keys)
    if errors or not all_candidates_completed:
        verdict = "V2A_TRIGGER_COVERAGE_VALIDATION_FAIL"
    elif trigger_count == 0:
        verdict = "V2A_TRIGGER_INACTIVE"
    elif complete_chain_count >= 1:
        verdict = "V2A_TRIGGER_COVERAGE_PASS"
    else:
        verdict = "V2A_TRIGGER_COVERAGE_VALIDATION_FAIL"
        errors.append("trigger occurred but no complete H20/H1/cooldown/H20 chain was observed")
    return {
        "schema_version": 1,
        "verdict": verdict,
        "formal_evaluation": False,
        "success_fields_used": False,
        "candidate_count": len(pairing_keys),
        "completed_candidate_count": len(episode_reports),
        "trigger_count": trigger_count,
        "complete_chain_count": complete_chain_count,
        "errors": errors,
        "episodes": episode_reports,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--verify-selection", action="store_true")
    parser.add_argument("--validate-output", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    if args.verify_selection == (args.validate_output is not None):
        raise SystemExit("choose exactly one of --verify-selection or --validate-output")
    if args.verify_selection:
        if args.source_dir is None or args.report_json is not None:
            raise SystemExit("selection verification requires --source-dir only")
        result = verify_candidate_document(args.source_dir, args.candidate_file)
    else:
        if args.source_dir is not None:
            raise SystemExit("output validation does not accept --source-dir")
        result = validate_coverage_output(args.validate_output, args.candidate_file)
        if args.report_json is not None:
            _atomic_write_json(args.report_json, result)
    print(json.dumps(result, sort_keys=True))
    if result.get("verdict") == "V2A_TRIGGER_COVERAGE_VALIDATION_FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
