from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "analysis" / "capture_official_eval_provenance.py"


def _command(output_dir: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--output-dir",
        str(output_dir),
        "--project-root",
        str(PROJECT_ROOT),
        "--suite",
        "libero_spatial",
        "--episodes-per-task",
        "2",
        "--seed",
        "1000",
        "--checkpoint",
        "HuggingFaceVLA/smolvla_libero",
        "--checkpoint-revision",
        "revision",
        "--num-steps",
        "2",
        "--episode-length",
        "280",
        *extra,
    ]


def test_provenance_writes_paired_seed_manifest_and_metadata(tmp_path: Path) -> None:
    subprocess.run(_command(tmp_path), check=True)

    manifest = json.loads((tmp_path / "paired_seed_manifest.json").read_text())
    assert len(manifest["trials"]) == 20
    assert manifest["trials"][0] == {
        "task_id": 0,
        "episode_index": 0,
        "seed": 1000,
        "initial_state_id": None,
        "initial_state_provenance": "LeRobot LIBERO reset(seed); state ID is not exposed by v0.6.1",
    }
    assert manifest["trials"][1]["seed"] == 1001
    assert manifest["trials"][2]["task_id"] == 1

    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["num_steps"] == 2
    assert provenance["seed"] == 1000
    assert provenance["exit_code"] is None


def test_provenance_derives_summary_only_from_existing_eval_info(tmp_path: Path) -> None:
    (tmp_path / "eval_info.json").write_text(
        json.dumps(
            {
                "per_task": [
                    {
                        "task_group": "libero_spatial",
                        "task_id": 3,
                        "metrics": {"successes": [True, False]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(_command(tmp_path, "--exit-code", "0"), check=True)

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {
                "task_group": "libero_spatial",
                "task_id": "3",
                "episodes": "2",
                "successes": "1",
                "success_rate": "0.5",
            }
        ]


def test_provenance_reuses_verified_manifest_and_records_its_hash(tmp_path: Path) -> None:
    shared_manifest = tmp_path / "paired-inputs" / "spatial-seed-1000.json"
    fixed_output = tmp_path / "fixed"
    adaptive_output = tmp_path / "adaptive"

    subprocess.run(_command(fixed_output, "--manifest-path", str(shared_manifest)), check=True)
    subprocess.run(_command(adaptive_output, "--manifest-path", str(shared_manifest)), check=True)

    expected_hash = hashlib.sha256(shared_manifest.read_bytes()).hexdigest()
    for output_dir in (fixed_output, adaptive_output):
        provenance = json.loads((output_dir / "provenance.json").read_text())
        assert provenance["paired_seed_manifest_path"] == str(shared_manifest.resolve())
        assert provenance["paired_seed_manifest_sha256"] == expected_hash
        assert (output_dir / "paired_seed_manifest.json").read_bytes() == shared_manifest.read_bytes()
