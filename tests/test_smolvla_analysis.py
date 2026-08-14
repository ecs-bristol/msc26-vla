from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTOR_SCRIPT = (
    REPO_ROOT / "scripts" / "analysis" / "select_smolvla_candidates.py"
)
GATE_SCRIPT = (
    REPO_ROOT / "scripts" / "analysis" / "assess_smolvla_capability_gate.py"
)
LORA_MANIFEST = (
    REPO_ROOT / "configs" / "training" / "smolvla_libero_spatial_lora.yaml"
)


TRIAL_FIELDS = [
    "task_id",
    "initial_state_id",
    "episode_id",
    "seed",
    "reset_seed",
    "success",
    "action_valid",
]


def _write_trials_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_selected_five_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "task_ids": [3, 0, 1, 2, 4],
                    "initial_state_ids": [0],
                },
                "execution": {
                    "episodes_per_initial_state": 5,
                    "seed": 42,
                },
            }
        ),
        encoding="utf-8",
    )


def _selected_five_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    episode_id = 0
    for task_id in [3, 0, 1, 2, 4]:
        for repetition in range(5):
            rows.append(
                {
                    "task_id": str(task_id),
                    "initial_state_id": "0",
                    "episode_id": str(episode_id),
                    "seed": str(42 + episode_id),
                    "reset_seed": str(42 + episode_id),
                    "success": str(task_id in {3, 0, 1}),
                    "action_valid": "True",
                }
            )
            episode_id += 1
    return rows


def _run_gate(
    trials_path: Path, config_path: Path, output_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--trials",
            str(trials_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


def test_selector_writes_successes_first_then_smallest_remaining_ids(
    tmp_path: Path,
) -> None:
    trials_path = tmp_path / "trials.csv"
    output_path = tmp_path / "selected.json"
    _write_trials_csv(
        trials_path,
        [
            {"task_id": "0", "success": "False"},
            {"task_id": "1", "success": "False"},
            {"task_id": "2", "success": "False"},
            {"task_id": "3", "success": "True"},
            {"task_id": "8", "success": "True"},
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SELECTOR_SCRIPT),
            "--trials",
            str(trials_path),
            "--output",
            str(output_path),
            "--minimum-tasks",
            "5",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == [3, 8, 0, 1, 2]


def test_selector_fails_when_minimum_task_count_cannot_be_met(tmp_path: Path) -> None:
    trials_path = tmp_path / "trials.csv"
    output_path = tmp_path / "selected.json"
    _write_trials_csv(
        trials_path,
        [
            {"task_id": "3", "success": "True"},
            {"task_id": "8", "success": "False"},
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SELECTOR_SCRIPT),
            "--trials",
            str(trials_path),
            "--output",
            str(output_path),
            "--minimum-tasks",
            "5",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert completed.returncode != 0
    assert "minimum 5 task IDs" in completed.stderr
    assert not output_path.exists()


def test_gate_marks_capability_pass_when_three_tasks_succeed(tmp_path: Path) -> None:
    trials_path = tmp_path / "trials.csv"
    config_path = tmp_path / "selected.yaml"
    output_path = tmp_path / "gate.json"
    _write_selected_five_config(config_path)
    _write_trials_csv(trials_path, _selected_five_rows())

    completed = _run_gate(trials_path, config_path, output_path)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "successful_task_count": 3,
        "mean_success_rate": 0.6,
        "passed": True,
        "fine_tuning_required": False,
    }


def test_gate_rejects_empty_trial_csv_with_actionable_error(tmp_path: Path) -> None:
    trials_path = tmp_path / "trials.csv"
    config_path = tmp_path / "selected.yaml"
    output_path = tmp_path / "gate.json"
    _write_selected_five_config(config_path)
    _write_trials_csv(trials_path, [])

    completed = _run_gate(trials_path, config_path, output_path)

    assert completed.returncode != 0
    assert "25 trials" in completed.stderr
    assert not output_path.exists()


def test_gate_rejects_incomplete_selected_five_protocol(tmp_path: Path) -> None:
    trials_path = tmp_path / "trials.csv"
    config_path = tmp_path / "selected.yaml"
    output_path = tmp_path / "gate.json"
    _write_selected_five_config(config_path)
    _write_trials_csv(trials_path, _selected_five_rows()[:-1])

    completed = _run_gate(trials_path, config_path, output_path)

    assert completed.returncode != 0
    assert "25 trials" in completed.stderr
    assert not output_path.exists()


def test_gate_rejects_invalid_actions(tmp_path: Path) -> None:
    trials_path = tmp_path / "trials.csv"
    config_path = tmp_path / "selected.yaml"
    output_path = tmp_path / "gate.json"
    rows = _selected_five_rows()
    rows[7]["action_valid"] = "False"
    _write_selected_five_config(config_path)
    _write_trials_csv(trials_path, rows)

    completed = _run_gate(trials_path, config_path, output_path)

    assert completed.returncode != 0
    assert "action_valid" in completed.stderr
    assert not output_path.exists()


def test_gate_rejects_unexpected_seed_structure(tmp_path: Path) -> None:
    trials_path = tmp_path / "trials.csv"
    config_path = tmp_path / "selected.yaml"
    output_path = tmp_path / "gate.json"
    rows = _selected_five_rows()
    rows[7]["seed"] = "999"
    _write_selected_five_config(config_path)
    _write_trials_csv(trials_path, rows)

    completed = _run_gate(trials_path, config_path, output_path)

    assert completed.returncode != 0
    assert "seed structure" in completed.stderr
    assert not output_path.exists()


def test_lora_manifest_declares_non_executable_request_defaults() -> None:
    assert LORA_MANIFEST.is_file()

    manifest = yaml.safe_load(LORA_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["base_checkpoint"] == "lerobot/smolvla_libero"
    assert manifest["training_device"] == "pc_or_gpu_host"
    assert manifest["adapter_type"] == "lora"
    assert manifest["evaluation_protocol"] == "engineering_demo_settle_0"
    assert manifest["seed"] == 42
