#!/usr/bin/env python3
"""Export frozen XS episode/diagnostic evidence without running a policy.

Run this script with the repository's WSL Python. It reads only completed JSON
artifacts and writes compact CSV/JSON inputs for the manuscript figures and
tables. It never imports a model, LeRobot, LIBERO, or an evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEV_SHA = "b5cbf91a7c1a47ac48b7d30ed318ecd2ea252d1a"
HELDOUT_SHA = "a9afdc0b4feee120f5c3c71f22d84c691ed85ba6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def episode_rows(sources: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[Path]]:
    fields = [
        "cohort", "condition", "task_id", "initial_state_id", "environment_seed",
        "inference_seed", "success_at_280", "success_step", "executed_env_steps",
        "model_invocations", "model_inference_time_s", "wall_time_to_terminal_s",
        "generated_actions", "unused_actions", "horizon_tail_discarded_actions",
        "trigger_tail_discarded_actions", "terminal_tail_unused_actions",
        "chunk_utilization", "mean_actual_horizon", "range_violations",
        "trigger_range_violations", "gripper_only_range_violations", "range_clips",
        "git_sha", "source_episode_json",
    ]
    rows: list[dict[str, Any]] = []
    inputs: list[Path] = []
    for source in sources:
        directory = Path(source["directory"])
        files = sorted(directory.glob("task_*_seed_*_state_*.json"))
        if len(files) != 50:
            raise ValueError(f"{source['condition']} must contain exactly 50 episodes, got {len(files)}")
        expected_sha = source["git_sha"]
        keys: set[tuple[int, int]] = set()
        for path in files:
            record = read_json(path)
            if record.get("status") != "completed" or record.get("git_sha") != expected_sha:
                raise ValueError(f"invalid frozen episode identity: {path}")
            task = int(record["task_id"])
            state = int(record["initial_state_id"])
            keys.add((task, state))
            generated = int(record["generated_actions"])
            executed = int(record["executed_env_steps"])
            unused = int(record["unused_actions"])
            parts = (
                int(record["horizon_tail_discarded_actions"])
                + int(record["trigger_tail_discarded_actions"])
                + int(record["terminal_tail_unused_actions"])
            )
            if generated != 50 * int(record["model_invocations"]):
                raise ValueError(f"G=C*M fails: {path}")
            if unused != generated - executed or unused != parts:
                raise ValueError(f"action conservation fails: {path}")
            row = {field: record.get(field) for field in fields}
            row.update({
                "cohort": source["cohort"],
                "condition": source["condition"],
                "success_at_280": int(bool(record["success_at_280"])),
                "source_episode_json": str(path),
            })
            rows.append(row)
            inputs.append(path)
        if keys != {(task, state) for task in range(10) for state in range(5)}:
            raise ValueError(f"incomplete task-state grid: {source['condition']}")
    rows.sort(key=lambda row: (row["cohort"], row["condition"], row["task_id"], row["initial_state_id"]))
    return rows, inputs


def outcome_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {
        (row["condition"], int(row["task_id"]), int(row["initial_state_id"])): row
        for row in episodes if row["cohort"] == "development"
    }
    rows: list[dict[str, Any]] = []
    counts = {"both_success": 0, "h1_only": 0, "h20_only": 0, "both_fail": 0}
    for task in range(10):
        for state in range(5):
            h1 = int(selected[("Static-H1", task, state)]["success_at_280"])
            h20 = int(selected[("Static-H20", task, state)]["success_at_280"])
            category = (
                "both_success" if h1 and h20 else
                "h1_only" if h1 else
                "h20_only" if h20 else
                "both_fail"
            )
            counts[category] += 1
            rows.append({
                "task_id": task,
                "initial_state_id": state,
                "h1_success": h1,
                "h20_success": h20,
                "outcome_category": category,
            })
    if counts != {"both_success": 26, "h1_only": 7, "h20_only": 6, "both_fail": 11}:
        raise ValueError(f"unexpected H1/H20 flip table: {counts}")
    return rows


def trigger_rows(dev_analysis: Path, heldout_analysis: Path) -> list[dict[str, Any]]:
    dev = read_json(dev_analysis)
    heldout = read_json(heldout_analysis)
    rows: list[dict[str, Any]] = []
    for case in dev["adaptive_trigger_summary"]["casebook"]:
        effect = (
            "loss" if case["adaptive_lost_static_success"] else
            "rescue" if case["adaptive_rescued_static_failure"] else
            "no_flip"
        )
        rows.append({
            "cohort": "development",
            "mechanism": "Adaptive-v1",
            "case_id": case["case_id"],
            "task_id": case["task_id"],
            "initial_state_id": case["initial_state_id"],
            "environment_seed": case["seed"],
            "env_step": case.get("trigger_step") or case["trigger_step_candidates"],
            "dimensions": case["violation_dimension"],
            "severity": "NA-v1",
            "excess": case.get("excess") if case.get("excess") is not None else case["excess_evidence"],
            "persistence": "none",
            "discarded_actions": case.get("discarded_actions") if case.get("discarded_actions") is not None else case["discarded_actions_candidates"],
            "state_chain": "H20-trigger-H1-H20",
            "effect": effect,
            "evidence_status": case["sequence_evidence_status"],
        })
    for case in heldout["trigger_casebook"]:
        rows.append({
            "cohort": "held-out",
            "mechanism": "Adaptive-v2a",
            "case_id": case["trigger_ordinal"],
            "task_id": case["task_id"],
            "initial_state_id": case["initial_state_id"],
            "environment_seed": case["seed"],
            "env_step": case["env_step"],
            "dimensions": "|".join(map(str, case["dimensions"])),
            "severity": "|".join(f"{value:.9f}" for value in case["severity"]),
            "excess": "|".join(f"{value:.9f}" for value in case["excess"]),
            "persistence": "|".join(map(str, case["persistence_counts"])),
            "discarded_actions": case["discarded_actions"],
            "state_chain": "H20-trigger-H1-H20-cooldown",
            "effect": case["effect"],
            "evidence_status": "exact",
        })
    if len(rows) != 8:
        raise ValueError(f"expected eight trigger events, got {len(rows)}")
    return rows


def variable_chunk_rows(report_path: Path) -> list[dict[str, Any]]:
    report = read_json(report_path)
    if report.get("classification") != "VARIABLE_CHUNK_RUNS_BUT_CHANGES_POLICY":
        raise ValueError("unexpected variable-chunk classification")
    comparisons: dict[int, list[dict[str, Any]]] = {}
    for item in report["comparisons"]:
        comparisons.setdefault(int(item["requested_chunk_size"]), []).append(item)
    rows: list[dict[str, Any]] = []
    for chunk in (1, 10, 20, 25, 30, 50):
        stage = report["stages"][str(chunk)]
        items = comparisons.get(chunk, [])
        rows.append({
            "chunk_size": chunk,
            "status": stage["status"],
            "strict_load": int(bool(stage["strict_load"]["success"])),
            "output_shape": "x".join(map(str, stage["output_shape"])),
            "finite": int(bool(stage["finite"])),
            "median_latency_ms": stage["latency_ms"]["median"],
            "mean_latency_ms": stage["latency_ms"]["mean"],
            "p95_latency_ms": stage["latency_ms"]["p95"],
            "c50_prefix_allclose": 1 if chunk == 50 else int(all(x["allclose"] for x in items)),
            "prefix_comparisons": len(items) if chunk != 50 else 3,
            "max_abs_difference": 0.0 if chunk == 50 else max(x["max_abs_difference"] for x in items),
            "max_rmse": 0.0 if chunk == 50 else max(x["rmse"] for x in items),
        })
    return rows


def parity_rows() -> list[dict[str, str]]:
    return [
        {"check": "environment input", "parity_error": "256x256", "official_or_repaired": "360x360", "validation": "official processor path"},
        {"check": "image orientation", "parity_error": "height flip only", "official_or_repaired": "height and width flip", "validation": "observation hashes equal"},
        {"check": "reset settle", "parity_error": "0 no-op steps", "official_or_repaired": "10 no-op steps", "validation": "official reset sequence"},
        {"check": "robot state", "parity_error": "normalized quaternion; force w>=0", "official_or_repaired": "official quaternion-to-axis-angle", "validation": "state hash equal; max diff 0"},
        {"check": "action handling", "parity_error": "clip to [-1,1]", "official_or_repaired": "native action; clipping off", "validation": "chunk hash equal; max diff 0"},
        {"check": "hash gate", "parity_error": "not available", "official_or_repaired": "agentview, wrist, state and action SHA256", "validation": "all exact"},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--output-dir", type=Path, default=root / "figures" / "xs_adaptive_extended" / "source")
    parser.add_argument("--wsl-vla-root", type=Path, default=Path("/home/xinrui_shen/vla"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wsl = args.wsl_vla_root

    sources = [
        {"cohort": "development", "condition": "Static-H1", "git_sha": DEV_SHA, "directory": str(wsl / "runs/parity-hardening/static-h1-original-50-b5cbf91-20260827/episodes/static-h1-original")},
        {"cohort": "development", "condition": "Static-H10", "git_sha": DEV_SHA, "directory": str(wsl / "runs/parity-hardening/static-h10-50-b5cbf91-20260827/episodes/static-h10")},
        {"cohort": "development", "condition": "Static-H20", "git_sha": DEV_SHA, "directory": str(wsl / "runs/parity-hardening/h20-paired-100-b5cbf91-20260827/episodes/static-h20")},
        {"cohort": "development", "condition": "Adaptive-v1", "git_sha": DEV_SHA, "directory": str(wsl / "runs/parity-hardening/h20-paired-100-b5cbf91-20260827/episodes/adaptive-h20-to-h1")},
        {"cohort": "held-out", "condition": "Static-H20", "git_sha": HELDOUT_SHA, "directory": str(wsl / "runs/adaptive-v2-prereg/adaptive-v2a-formal-heldout-100-a9afdc0-20260828/episodes/static-h20")},
        {"cohort": "held-out", "condition": "Adaptive-v2a", "git_sha": HELDOUT_SHA, "directory": str(wsl / "runs/adaptive-v2-prereg/adaptive-v2a-formal-heldout-100-a9afdc0-20260828/episodes/adaptive-v2a-h20-to-h1")},
    ]
    episodes, episode_inputs = episode_rows(sources)
    episode_fields = list(episodes[0])
    write_csv(out / "xs_episode_metrics.csv", episodes, episode_fields)
    write_csv(out / "xs_h1_h20_task_state_outcomes.csv", outcome_rows(episodes),
              ["task_id", "initial_state_id", "h1_success", "h20_success", "outcome_category"])

    dev_analysis = wsl / "runs/parity-hardening/h20-paired-100-b5cbf91-20260827/analysis/four_condition_paired_analysis.json"
    heldout_analysis = wsl / "runs/adaptive-v2-prereg/adaptive-v2a-formal-heldout-100-a9afdc0-20260828/analysis/adaptive_v2a_formal_analysis.json"
    casebook = trigger_rows(dev_analysis, heldout_analysis)
    write_csv(out / "xs_trigger_casebook.csv", casebook, list(casebook[0]))

    variable_report = wsl / "runs/parity-hardening/variable-chunk-feasibility-v2-b5cbf91-20260827/variable_chunk_v2_report.json"
    variable = variable_chunk_rows(variable_report)
    write_csv(out / "xs_variable_chunk_v2.csv", variable, list(variable[0]))
    parity = parity_rows()
    write_csv(out / "xs_parity_validation.csv", parity, list(parity[0]))

    fixed_inputs = [
        root / "analysis/final_vla_results.csv",
        root / "analysis/final_vla_statistics.json",
        root / "docs/BASELINE_PARITY_AUDIT.md",
        root / "docs/PARITY_HARDENING.md",
        root / "docs/ADAPTIVE_V2_PREREG.md",
        root / "docs/ADAPTIVE_V2_FORMAL_HELDOUT_PREREG.md",
        root / "src/libero_platform/policies/fixed_h_action_buffer.py",
        root / "src/libero_platform/policies/adaptive_v2_trigger.py",
        dev_analysis, heldout_analysis, variable_report,
    ]
    all_inputs = [*fixed_inputs, *episode_inputs]
    manifest = {
        "schema_version": 1,
        "read_only_sources": True,
        "model_loaded": False,
        "environment_created": False,
        "rollout_executed": False,
        "cohort_pooling": False,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in all_inputs],
        "outputs": [
            {"path": path.name, "sha256": sha256(path)}
            for path in sorted(out.glob("xs_*.csv"))
        ],
    }
    (out / "xs_frozen_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "episodes": len(episodes), "triggers": len(casebook), "output_dir": str(out)}))


if __name__ == "__main__":
    main()
