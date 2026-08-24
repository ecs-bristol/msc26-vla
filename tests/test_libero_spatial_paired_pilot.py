from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "analysis" / "libero_spatial_paired_pilot.py"
SMOLVLA_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SMOLVLM2_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
SUMMARY_FIELDS = {
    "condition",
    "task_id",
    "seed",
    "initial_state_id",
    "success_at_280",
    "success_step",
    "executed_env_steps",
    "wall_time_to_terminal_s",
    "model_invocations",
    "model_inference_time_s",
    "range_violations",
    "range_clips",
    "buffer_discards",
    "mean_actual_horizon",
    "termination_reason",
    "git_sha",
    "resolved_config_path",
}


def _snapshot(tmp_path: Path, revision: str) -> Path:
    path = tmp_path / "hf" / "hub" / "models--example" / "snapshots" / revision
    path.mkdir(parents=True)
    return path


def _command(output_dir: Path, base_snapshot: Path, vlm_snapshot: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--dry-run",
        "--output-dir",
        str(output_dir),
        "--base-snapshot-path",
        str(base_snapshot),
        "--vlm-snapshot-path",
        str(vlm_snapshot),
        *extra,
    ]


def test_dry_run_materializes_six_strictly_paired_conditions(tmp_path: Path) -> None:
    output_dir = tmp_path / "pilot"
    base_snapshot = _snapshot(tmp_path, SMOLVLA_REVISION)
    vlm_snapshot = _snapshot(tmp_path, SMOLVLM2_REVISION)

    subprocess.run(_command(output_dir, base_snapshot, vlm_snapshot), check=True)

    manifest = json.loads((output_dir / "paired_manifest.json").read_text())
    assert manifest["pairing_key"] == ["task_id", "seed", "initial_state_id"]
    assert len(manifest["trials"]) == 50
    assert manifest["trials"][0] == {
        "task_id": 0,
        "seed": 1000,
        "initial_state_id": 0,
        "episode_index": 0,
    }
    assert manifest["trials"][-1] == {
        "task_id": 9,
        "seed": 1004,
        "initial_state_id": 4,
        "episode_index": 4,
    }

    resolved = json.loads((output_dir / "resolved_config.json").read_text())
    assert resolved["suite"] == "libero_spatial"
    assert resolved["episode_cap"] == 280
    assert resolved["batch_size"] == 1
    assert resolved["model"]["local_files_only"] is True
    assert resolved["model"]["num_steps"] == 2
    assert resolved["model"]["chunk_size"] == 50
    assert resolved["model"]["base_revision"] == SMOLVLA_REVISION
    assert resolved["model"]["vlm_revision"] == SMOLVLM2_REVISION
    assert [condition["name"] for condition in resolved["conditions"]] == [
        "Static-H1",
        "Static-H5",
        "Static-H10",
        "Static-H20",
        "Static-H50",
        "Adaptive-H20→H1",
    ]

    episode_files = list((output_dir / "episodes").rglob("*.json"))
    assert len(episode_files) == 300
    adaptive = json.loads(
        (output_dir / "episodes" / "adaptive-h20-to-h1" / "task_00_seed_1000_state_0.json").read_text()
    )
    assert adaptive["condition_config"]["replan_after_safety_violation"] is True
    assert adaptive["termination_reason"] == "not_started_dry_run"
    assert adaptive["resolved_config_path"] == str((output_dir / "resolved_config.json").resolve())

    with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert set(rows[0]) == SUMMARY_FIELDS
    assert len(rows) == 300
    assert {row["condition"] for row in rows} == {
        "Static-H1",
        "Static-H5",
        "Static-H10",
        "Static-H20",
        "Static-H50",
        "Adaptive-H20→H1",
    }


def test_pilot_tool_refuses_a_non_dry_run_invocation(tmp_path: Path) -> None:
    base_snapshot = _snapshot(tmp_path, SMOLVLA_REVISION)
    vlm_snapshot = _snapshot(tmp_path, SMOLVLM2_REVISION)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(tmp_path / "pilot"),
            "--base-snapshot-path",
            str(base_snapshot),
            "--vlm-snapshot-path",
            str(vlm_snapshot),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "currently supports --dry-run only" in completed.stderr
