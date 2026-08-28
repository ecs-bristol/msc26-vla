#!/usr/bin/env python3
"""Read-only integrity gate for the frozen final VLA result package."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


DEVELOPMENT = "parity-corrected-development"
CONFIRMATORY = "untouched-heldout-confirmatory"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_close(actual: str | float, expected: float, *, field: str) -> None:
    if abs(float(actual) - float(expected)) > 1e-12:
        raise AssertionError(f"{field}: {actual} != {expected}")


def _verify_external_sources(manifest: dict[str, object]) -> None:
    cohorts = manifest["cohorts"]
    development = cohorts["parity_corrected_development"]
    for source in development["sources"]:
        result_dir = Path(source["result_directory"])
        if _sha256(result_dir / "summary.csv") != source["summary_csv_sha256"]:
            raise AssertionError(f"development summary changed: {result_dir}")
        analysis_path = Path(source["analysis_path"])
        if _sha256(analysis_path) != source["analysis_sha256"]:
            raise AssertionError(f"development analysis changed: {analysis_path}")
        if _sha256(result_dir / "paired_manifest.json") != development["manifest_sha256"]:
            raise AssertionError(f"development manifest changed: {result_dir}")

    confirmatory = cohorts["untouched_heldout_confirmatory"]
    result_dir = Path(confirmatory["result_directory"])
    checks = {
        result_dir / "summary.csv": confirmatory["summary_csv_sha256"],
        result_dir / "paired_manifest.json": confirmatory["manifest_sha256"],
        Path(confirmatory["preregistered_analysis_path"]): confirmatory["preregistered_analysis_sha256"],
        Path(confirmatory["independent_validation_path"]): confirmatory["independent_validation_sha256"],
    }
    for path, expected in checks.items():
        if _sha256(path) != expected:
            raise AssertionError(f"confirmatory source changed: {path}")


def _verify_repo_hashes(project_root: Path, manifest: dict[str, object]) -> None:
    for section in ("frozen_method_files", "generated_artifacts"):
        for relative_path, expected in manifest[section].items():
            path = project_root / relative_path
            if not path.is_file():
                raise AssertionError(f"missing {section} file: {relative_path}")
            if _sha256(path) != expected:
                raise AssertionError(f"SHA mismatch for {relative_path}")


def _verify_tables(project_root: Path, statistics: dict[str, object]) -> None:
    results = _read_csv(project_root / "analysis/final_vla_results.csv")
    if len(results) != 6:
        raise AssertionError("final_vla_results.csv must contain six condition rows")
    if {row["cohort"] for row in results} != {DEVELOPMENT, CONFIRMATORY}:
        raise AssertionError("result cohorts are missing or merged")
    if sum(row["cohort"] == DEVELOPMENT for row in results) != 4:
        raise AssertionError("development cohort must contain four rows")
    if sum(row["cohort"] == CONFIRMATORY for row in results) != 2:
        raise AssertionError("confirmatory cohort must contain two rows")

    cohort_map = {"development": DEVELOPMENT, "confirmatory": CONFIRMATORY}
    index = {(row["cohort"], row["condition"]): row for row in results}
    if len(index) != 6:
        raise AssertionError("duplicate final result row")
    for condition in statistics["condition_results"]:
        row = index[(cohort_map[condition["cohort"]], condition["condition"])]
        if int(row["episodes"]) != condition["episodes"]:
            raise AssertionError("episode count mismatch")
        if int(row["successes"]) != condition["successes"]:
            raise AssertionError("success count mismatch")
        _assert_close(row["success_rate"], condition["success_rate"], field="success_rate")
        _assert_close(row["model_calls_mean"], condition["model_calls_mean"], field="model_calls_mean")
        _assert_close(row["wall_time_mean_s"], condition["wall_time_mean_s"], field="wall_time_mean_s")
        _assert_close(
            row["chunk_utilization_aggregate"],
            condition["chunk_utilization_aggregate"],
            field="chunk_utilization_aggregate",
        )

    display_map = {
        "parity-corrected development": DEVELOPMENT,
        "untouched held-out confirmatory": CONFIRMATORY,
    }
    figure_specs = {
        "final_vla_success.csv": ("success_rate", "success_rate"),
        "final_vla_model_calls.csv": ("mean_model_calls", "model_calls_mean"),
        "final_vla_wall_time.csv": ("mean_wall_time_s", "wall_time_mean_s"),
    }
    label_normalization = {
        "Adaptive-v1 H20→H1": "Adaptive-v1-H20→H1",
        "Adaptive-v2a H20→H1": "Adaptive-v2a-H20→H1",
    }
    for filename, (figure_field, result_field) in figure_specs.items():
        rows = _read_csv(project_root / "analysis/figures" / filename)
        if len(rows) != 6:
            raise AssertionError(f"{filename} must contain six rows")
        for figure_row in rows:
            key = (
                display_map[figure_row["cohort"]],
                label_normalization.get(figure_row["condition"], figure_row["condition"]),
            )
            _assert_close(figure_row[figure_field], index[key][result_field], field=filename)

    if statistics["cohort_separation"]["cross_cohort_pairing_or_pooling"] is not False:
        raise AssertionError("cross-cohort pooling must remain disabled")
    if set(statistics["paired_comparisons"]) != {"development", "confirmatory"}:
        raise AssertionError("paired comparisons must remain cohort-scoped")
    if statistics["exclusions"][0]["included_in_final_performance_results"] is not False:
        raise AssertionError("parity-error results must remain excluded")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    current_head = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = _read_json(project_root / "analysis/final_reproducibility_manifest.json")
    statistics = _read_json(project_root / "analysis/final_vla_statistics.json")
    if current_head != manifest["repository_base_commit"]:
        raise AssertionError(f"HEAD {current_head} does not match frozen base commit")
    _verify_external_sources(manifest)
    _verify_repo_hashes(project_root, manifest)
    _verify_tables(project_root, statistics)

    report = (project_root / "docs/FINAL_VLA_EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    table = (project_root / "docs/FINAL_VLA_RESULTS_TABLE.md").read_text(encoding="utf-8")
    for required in (
        "Stage A — parity-corrected development cohort",
        "Stage B — untouched held-out confirmatory cohort",
        "No demonstrated value for the added calls.",
    ):
        if required not in report:
            raise AssertionError(f"final report is missing: {required}")
    if "Statistical firewall" not in table:
        raise AssertionError("results table lacks the cohort firewall")
    print("FINAL_VLA_RESULTS_FREEZE_PASS")


if __name__ == "__main__":
    main()
