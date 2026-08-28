#!/usr/bin/env python3
"""Frozen analysis for the paired Adaptive-v2a formal held-out experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np


STATIC = "Static-H20"
ADAPTIVE = "Adaptive-v2a-H20→H1"
CONDITIONS = (STATIC, ADAPTIVE)
FORBIDDEN_BEFORE_COMPLETE = frozenset(
    {"success_at_280", "success_step", "termination_reason"}
)
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20_260_828
NON_OUTCOME_SUMMARY_FIELDS = (
    "condition",
    "task_id",
    "seed",
    "initial_state_id",
    "environment_seed",
    "inference_seed",
    "executed_env_steps",
    "wall_time_to_terminal_s",
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


class AnalysisLockedError(RuntimeError):
    """Raised before any outcome field may be inspected."""


class OutcomeLockedEpisode(Mapping[str, Any]):
    """Mapping used by the completeness gate to reject outcome access."""

    def __init__(self, source: Mapping[str, Any]) -> None:
        self._source = source

    @staticmethod
    def _guard(key: object) -> None:
        if key in FORBIDDEN_BEFORE_COMPLETE:
            raise AnalysisLockedError(f"outcome access before completeness: {key}")

    def __getitem__(self, key: str) -> Any:
        self._guard(key)
        return self._source[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key in self._source if key not in FORBIDDEN_BEFORE_COMPLETE)

    def __len__(self) -> int:
        return sum(key not in FORBIDDEN_BEFORE_COMPLETE for key in self._source)

    def __contains__(self, key: object) -> bool:
        self._guard(key)
        return key in self._source

    def get(self, key: str, default: Any = None) -> Any:
        self._guard(key)
        return self._source.get(key, default)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["task_id"]),
        int(record["seed"]),
        int(record["initial_state_id"]),
    )


def _summary_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def validate_complete_without_outcomes(
    *,
    raw_records: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    summary_rows: list[Mapping[str, str]],
) -> dict[str, Any]:
    """Hard gate that cannot inspect success, success step, or termination reason."""

    trials = manifest.get("trials")
    seed_records = manifest.get("inference_seed_records")
    if not isinstance(trials, list) or not isinstance(seed_records, list):
        raise AnalysisLockedError("formal manifest is missing trials or inference seeds")
    expected_keys = {_key(item) for item in trials}
    if len(trials) != 50 or len(expected_keys) != 50:
        raise AnalysisLockedError("formal manifest must contain 50 unique pairing keys")
    expected_seeds = {
        (
            int(item["task_id"]),
            int(item["environment_seed"]),
            int(item["initial_state_id"]),
        ): int(item["inference_seed"])
        for item in seed_records
    }
    if set(expected_seeds) != expected_keys:
        raise AnalysisLockedError("manifest seed records do not match pairing keys")
    records: dict[tuple[str, int, int, int], OutcomeLockedEpisode] = {}
    for raw in raw_records:
        record = OutcomeLockedEpisode(raw)
        identity = (str(record["condition"]), *_key(record))
        if identity in records:
            raise AnalysisLockedError(f"duplicate formal episode: {identity}")
        records[identity] = record
    expected_identities = {
        (condition, *key) for condition in CONDITIONS for key in expected_keys
    }
    if set(records) != expected_identities:
        missing = sorted(expected_identities - set(records))
        extra = sorted(set(records) - expected_identities)
        raise AnalysisLockedError(f"formal episode set incomplete; missing={missing}, extra={extra}")
    if any(record.get("status") != "completed" for record in records.values()):
        raise AnalysisLockedError("all 100 formal episodes must be completed before analysis")
    git_shas = {str(record["git_sha"]) for record in records.values()}
    if len(git_shas) != 1:
        raise AnalysisLockedError("formal episodes do not share one Git SHA")
    expected_configs = {
        STATIC: {
            "name": STATIC,
            "fixed_h": 20,
            "safety_enabled": True,
            "replan_after_safety_violation": False,
            "adaptive_v2_trigger": False,
            "clip_actions": False,
        },
        ADAPTIVE: {
            "name": ADAPTIVE,
            "fixed_h": 20,
            "safety_enabled": True,
            "replan_after_safety_violation": False,
            "adaptive_v2_trigger": True,
            "clip_actions": False,
        },
    }
    accounting_errors: list[str] = []
    for identity, record in records.items():
        condition, task, seed, state = identity
        if record["condition_config"] != expected_configs[condition]:
            accounting_errors.append(f"condition config mismatch: {identity}")
        if int(record["inference_seed"]) != expected_seeds[(task, seed, state)]:
            accounting_errors.append(f"inference seed mismatch: {identity}")
        realized = [int(value) for value in record["realized_actions_per_call"]]
        checks = (
            int(record["generated_actions"]) == 50 * int(record["model_invocations"]),
            len(realized) == int(record["model_invocations"]),
            sum(realized) == int(record["executed_env_steps"]),
            int(record["unused_actions"])
            == int(record["generated_actions"]) - int(record["executed_env_steps"]),
            int(record["unused_actions"])
            == int(record["horizon_tail_discarded_actions"])
            + int(record["trigger_tail_discarded_actions"])
            + int(record["terminal_tail_unused_actions"]),
            int(record["range_clips"]) == 0,
        )
        if not all(checks):
            accounting_errors.append(f"action/accounting mismatch: {identity}")
        events = list(record.get("adaptive_v2_trigger_events", []))
        event_errors = [
            error
            for event in events
            for error in _validate_trigger_event(event)
        ]
        if event_errors:
            accounting_errors.append(f"trigger event mismatch {identity}: {event_errors}")
        if condition == STATIC and events:
            accounting_errors.append(f"Static-H20 contains trigger events: {identity}")
        if condition == ADAPTIVE:
            if int(record["trigger_range_violations"]) != len(events):
                accounting_errors.append(f"trigger aggregate mismatch: {identity}")
            if int(record["buffer_discards"]) != len(events):
                accounting_errors.append(f"buffer discard aggregate mismatch: {identity}")
            if int(record["trigger_tail_discarded_actions"]) != sum(
                int(event["discarded_actions"]) for event in events
            ):
                accounting_errors.append(f"trigger tail aggregate mismatch: {identity}")
    if accounting_errors:
        raise AnalysisLockedError("; ".join(accounting_errors))
    summary_map = {
        (
            str(row["condition"]),
            int(row["task_id"]),
            int(row["seed"]),
            int(row["initial_state_id"]),
        ): row
        for row in summary_rows
    }
    if set(summary_map) != expected_identities:
        raise AnalysisLockedError("formal summary does not contain exactly 100 paired rows")
    for identity, record in records.items():
        row = summary_map[identity]
        for field in NON_OUTCOME_SUMMARY_FIELDS:
            if row[field] != _summary_value(record.get(field)):
                raise AnalysisLockedError(f"summary mismatch {field}: {identity}")
    return {
        "pair_count": 50,
        "episode_count": 100,
        "git_sha": next(iter(git_shas)),
        "outcome_fields_read": False,
        "pairing_gate_passed": True,
    }


def _cluster_bootstrap(values: dict[int, list[float]]) -> dict[str, float]:
    tasks = sorted(values)
    if tasks != list(range(10)) or any(len(values[task]) != 5 for task in tasks):
        raise ValueError("task-cluster bootstrap requires ten tasks with five pairs each")
    matrix = np.asarray([values[task] for task in tasks], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(tasks), size=(BOOTSTRAP_REPLICATES, len(tasks)))
    estimates = matrix[draws].mean(axis=(1, 2))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean": float(matrix.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _mcnemar_exact(adaptive_only: int, static_only: int) -> float:
    discordant = adaptive_only + static_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(adaptive_only, static_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _validate_trigger_event(event: Mapping[str, Any]) -> list[str]:
    required = {
        "task_id",
        "initial_state_id",
        "seed",
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
    }
    errors = [f"missing:{field}" for field in sorted(required - set(event))]
    if not errors:
        if event["evaluation_order"] != "trigger_evaluated_before_env_step":
            errors.append("not_pre_execution")
        if not event["dimensions"] or any(int(value) == 6 for value in event["dimensions"]):
            errors.append("invalid_trigger_dimensions")
        lengths = {
            len(event[field])
            for field in (
                "dimensions",
                "raw_values",
                "bounds",
                "excess",
                "severity",
                "persistence_counts",
            )
        }
        if lengths != {len(event["dimensions"])}:
            errors.append("unaligned_per_dimension_fields")
        if (
            int(event["horizon_before"]) != 20
            or int(event["horizon_after"]) != 1
            or int(event["recovery_horizon"]) != 20
        ):
            errors.append("invalid_horizon_transition")
    return errors


def analyze_unlocked_outcomes(
    raw_records: list[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Execute only after the success-blind 100-episode gate has passed."""

    if gate.get("pairing_gate_passed") is not True or gate.get("episode_count") != 100:
        raise AnalysisLockedError("outcome analysis is locked until all 100 episodes pass")
    by_condition = {
        condition: {_key(record): record for record in raw_records if record["condition"] == condition}
        for condition in CONDITIONS
    }
    keys = sorted(by_condition[STATIC])
    flips = {"both_success": 0, "adaptive_only": 0, "static_only": 0, "both_fail": 0}
    success_values: dict[int, list[float]] = {task: [] for task in range(10)}
    resource_values = {
        field: {task: [] for task in range(10)}
        for field in ("model_invocations", "model_inference_time_s", "wall_time_to_terminal_s")
    }
    trigger_casebook: list[dict[str, Any]] = []
    trigger_errors: list[str] = []
    rescue_keys: set[tuple[int, int, int]] = set()
    valid_trigger_keys: set[tuple[int, int, int]] = set()
    for key in keys:
        static = by_condition[STATIC][key]
        adaptive = by_condition[ADAPTIVE][key]
        static_success = bool(static["success_at_280"])
        adaptive_success = bool(adaptive["success_at_280"])
        if static_success and adaptive_success:
            flip = "both_success"
        elif adaptive_success:
            flip = "adaptive_only"
        elif static_success:
            flip = "static_only"
        else:
            flip = "both_fail"
        flips[flip] += 1
        if flip == "adaptive_only":
            rescue_keys.add(key)
        success_values[key[0]].append(float(adaptive_success) - float(static_success))
        for field in resource_values:
            resource_values[field][key[0]].append(float(adaptive[field]) - float(static[field]))
        for ordinal, event in enumerate(adaptive.get("adaptive_v2_trigger_events", []), start=1):
            errors = _validate_trigger_event(event)
            trigger_errors.extend(f"{key}/trigger{ordinal}:{error}" for error in errors)
            if not errors:
                valid_trigger_keys.add(key)
            trigger_casebook.append(
                {
                    "task_id": key[0],
                    "seed": key[1],
                    "initial_state_id": key[2],
                    "trigger_ordinal": ordinal,
                    "model_call_index": event.get("model_call_index"),
                    "env_step": event.get("env_step"),
                    "dimensions": event.get("dimensions"),
                    "raw_values": event.get("raw_values"),
                    "bounds": event.get("bounds"),
                    "excess": event.get("excess"),
                    "severity": event.get("severity"),
                    "persistence_counts": event.get("persistence_counts"),
                    "discarded_actions": event.get("discarded_actions"),
                    "static_success": static_success,
                    "adaptive_success": adaptive_success,
                    "effect": (
                        "rescue"
                        if adaptive_success and not static_success
                        else "loss"
                        if static_success and not adaptive_success
                        else "no_flip"
                    ),
                    "event_errors": errors,
                }
            )
    success_effect = _cluster_bootstrap(success_values)
    resources = {
        field: _cluster_bootstrap(values) for field, values in resource_values.items()
    }
    evidence_ok = not trigger_errors and rescue_keys <= valid_trigger_keys
    decision = {
        "at_least_one_rescue": flips["adaptive_only"] >= 1,
        "rescues_outnumber_losses": flips["adaptive_only"] > flips["static_only"],
        "trigger_evidence_complete": evidence_ok,
    }
    decision["added_calls_have_demonstrated_value"] = all(decision.values())
    return {
        "schema_version": 1,
        "analysis_preregistered": True,
        "outcome_gate": dict(gate),
        "flip_table": flips,
        "paired_success_difference": success_effect,
        "mcnemar_exact_p": _mcnemar_exact(
            flips["adaptive_only"], flips["static_only"]
        ),
        "paired_resource_differences_adaptive_minus_static": resources,
        "trigger_count": len(trigger_casebook),
        "trigger_casebook": trigger_casebook,
        "trigger_validation_errors": trigger_errors,
        "decision_rule": decision,
        "decision_conclusion": (
            "added calls have demonstrated value under preregistered descriptive gates"
            if decision["added_calls_have_demonstrated_value"]
            else "no demonstrated value for the added calls"
        ),
        "bootstrap": {
            "cluster": "task_id",
            "replicates": BOOTSTRAP_REPLICATES,
            "rng_seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
    }


def analyze_output(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest = _read_json(output_dir / "paired_manifest.json")
    raw_records = [
        _read_json(path) for path in sorted((output_dir / "episodes").rglob("*.json"))
    ]
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    gate = validate_complete_without_outcomes(
        raw_records=raw_records,
        manifest=manifest,
        summary_rows=summary_rows,
    )
    return analyze_unlocked_outcomes(raw_records, gate)


def _atomic_write(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _markdown(report: Mapping[str, Any]) -> str:
    flips = report["flip_table"]
    effect = report["paired_success_difference"]
    lines = [
        "# Adaptive-v2a formal held-out paired analysis",
        "",
        "Analysis unlocked only after all 100 episodes passed the outcome-blind completeness gate.",
        "",
        "| both success | Adaptive only | Static only | both fail |",
        "|---:|---:|---:|---:|",
        f"| {flips['both_success']} | {flips['adaptive_only']} | {flips['static_only']} | {flips['both_fail']} |",
        "",
        f"Paired success difference: {effect['mean']:.6f} "
        f"(task-cluster bootstrap 95% CI {effect['ci95_low']:.6f}, {effect['ci95_high']:.6f}).",
        f"McNemar exact p: {report['mcnemar_exact_p']:.12g}.",
        f"Trigger count: {report['trigger_count']}.",
        f"Decision: {report['decision_conclusion']}.",
        "",
        "No trigger parameter or decision rule is modified by this analysis.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_output(args.output_dir)
    _atomic_write(args.report_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write(args.report_md, _markdown(report))
    print(json.dumps({"analysis": "complete", "decision": report["decision_conclusion"]}, sort_keys=True))


if __name__ == "__main__":
    main()
