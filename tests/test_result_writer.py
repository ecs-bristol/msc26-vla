from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from libero_platform.result_schema import DeviceProfile, TrialRecord, summarize_trials
from libero_platform.result_writer import write_required_artifacts


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run_1"


def test_writes_all_required_files_with_empty_failures(run_dir: Path) -> None:
    trial = TrialRecord.example(success=True)

    paths = write_required_artifacts(
        run_dir=run_dir,
        metadata={"run_id": "run_1"},
        device_profile={"device_model": "test"},
        trials=[trial],
    )

    assert set(paths) == {
        "metadata",
        "trials_jsonl",
        "trials_csv",
        "summary_csv",
        "failures_csv",
        "device_profile",
    }
    with paths["failures_csv"].open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
    assert paths["failures_csv"].read_text(encoding="utf-8").strip()
    assert json.loads(paths["metadata"].read_text(encoding="utf-8"))["run_id"] == "run_1"


def test_preserves_existing_recorder_trials_jsonl(run_dir: Path) -> None:
    run_dir.mkdir()
    trials_jsonl = run_dir / "trials.jsonl"
    original = b'{"episode_id":0,"success":true}\n'
    trials_jsonl.write_bytes(original)

    paths = write_required_artifacts(
        run_dir=run_dir,
        metadata={"run_id": "run_1"},
        device_profile={"device_model": "test"},
        trials=[TrialRecord.example(success=False)],
        preserve_existing_trials_jsonl=True,
    )

    assert paths["trials_jsonl"].read_bytes() == original


def test_summary_groups_by_v1_dimensions_and_interpolates_p95() -> None:
    first = TrialRecord.example(
        success=True,
        policy_latency_mean_ms=0.0,
        end_to_end_mean_ms=2.0,
    )
    second = TrialRecord.example(
        success=False,
        policy_latency_mean_ms=10.0,
        end_to_end_mean_ms=12.0,
        oom=True,
        load_success=False,
        failure_type="oom",
        episode_id=1,
    )
    separate_group = TrialRecord.example(
        success=True,
        suite="libero_object",
        task_id=99,
        policy_key="other_policy",
        deployment_mode="jetson_local",
        precision="fp16",
        quantization="int8",
    )

    summaries = summarize_trials([first, second, separate_group])

    first_summary = summaries[0]
    assert first_summary["suite"] == "libero_object"
    assert first_summary["policy_key"] == "other_policy"

    grouped_summary = summaries[1]
    assert grouped_summary["suite"] == "libero_spatial"
    assert grouped_summary["task_id"] == 0
    assert grouped_summary["policy_key"] == "zero_policy"
    assert grouped_summary["deployment_mode"] == "pc_local"
    assert grouped_summary["precision"] == "none"
    assert grouped_summary["quantization"] == "none"
    assert grouped_summary["trials"] == 2
    assert grouped_summary["policy_latency_p95_ms"] == 9.5
    assert grouped_summary["end_to_end_p95_ms"] == 11.5
    assert grouped_summary["oom_count"] == 1
    assert grouped_summary["load_failure_count"] == 1


def test_device_profile_preserves_unavailable_metrics_as_json_null(run_dir: Path) -> None:
    profile = DeviceProfile(
        device_model="test-device",
        peak_host_memory_mb=None,
        peak_device_memory_mb=None,
    )

    paths = write_required_artifacts(
        run_dir=run_dir,
        metadata={"run_id": "run_1"},
        device_profile=profile,
        trials=[],
    )

    saved_profile = json.loads(paths["device_profile"].read_text(encoding="utf-8"))
    assert saved_profile["peak_host_memory_mb"] is None
    assert saved_profile["peak_device_memory_mb"] is None


def test_json_artifacts_normalize_non_finite_values(run_dir: Path) -> None:
    trial = TrialRecord.example(
        success=True,
        policy_latency_mean_ms=math.nan,
        end_to_end_mean_ms=math.inf,
    )

    paths = write_required_artifacts(
        run_dir=run_dir,
        metadata={"metrics": [math.nan, {"infinite": -math.inf}]},
        device_profile={"device_model": "test", "memory_mb": math.nan},
        trials=[trial],
    )

    metadata = _strict_json(paths["metadata"].read_text(encoding="utf-8"))
    device_profile = _strict_json(paths["device_profile"].read_text(encoding="utf-8"))
    trials = [
        _strict_json(line)
        for line in paths["trials_jsonl"].read_text(encoding="utf-8").splitlines()
    ]

    assert metadata["metrics"] == [None, {"infinite": None}]
    assert device_profile["memory_mb"] is None
    assert trials[0]["policy_latency_mean_ms"] is None
    assert trials[0]["end_to_end_mean_ms"] is None


def test_csv_roundtrip_coerces_boolean_strings_for_summary_and_failures(run_dir: Path) -> None:
    successful = TrialRecord.example(success=True, action_valid=True, episode_id=0)
    failed = TrialRecord.example(
        success=False,
        action_valid=False,
        oom=True,
        load_success=False,
        episode_id=1,
        failure_type="oom",
    )
    original_paths = write_required_artifacts(
        run_dir=run_dir / "original",
        metadata={"run_id": "run_1"},
        device_profile={"device_model": "test"},
        trials=[successful, failed],
    )
    with original_paths["trials_csv"].open(newline="", encoding="utf-8") as handle:
        csv_trials = list(csv.DictReader(handle))

    paths = write_required_artifacts(
        run_dir=run_dir / "roundtrip",
        metadata={"run_id": "run_1"},
        device_profile={"device_model": "test"},
        trials=csv_trials,
    )
    with paths["summary_csv"].open(newline="", encoding="utf-8") as handle:
        summary = next(csv.DictReader(handle))
    with paths["failures_csv"].open(newline="", encoding="utf-8") as handle:
        failures = list(csv.DictReader(handle))

    assert summary["success_rate"] == "0.5"
    assert summary["action_valid_rate"] == "0.5"
    assert summary["oom_count"] == "1"
    assert summary["load_failure_count"] == "1"
    assert [failure["episode_id"] for failure in failures] == ["1"]


def _strict_json(text: str) -> object:
    return json.loads(text, parse_constant=_reject_non_standard_json_constant)


def _reject_non_standard_json_constant(value: str) -> object:
    raise AssertionError(f"non-standard JSON constant: {value}")
