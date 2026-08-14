from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .result_schema import (
    DeviceProfile,
    FAILURE_FIELDS,
    SUMMARY_FIELDS,
    TRIAL_RECORD_FIELDS,
    TrialRecord,
    failure_rows,
    summarize_trials,
)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(dict(metadata)), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_required_artifacts(
    *,
    run_dir: Path,
    metadata: Mapping[str, Any],
    device_profile: DeviceProfile | Mapping[str, Any],
    trials: Sequence[TrialRecord | Mapping[str, Any]],
    preserve_existing_trials_jsonl: bool = False,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    trial_rows = [_json_safe(_trial_row(trial)) for trial in trials]
    device_profile_row = _json_safe(_device_profile_row(device_profile))

    paths = {
        "metadata": run_dir / "metadata.json",
        "trials_jsonl": run_dir / "trials.jsonl",
        "trials_csv": run_dir / "trials.csv",
        "summary_csv": run_dir / "summary.csv",
        "failures_csv": run_dir / "failures.csv",
        "device_profile": run_dir / "device_profile.json",
    }
    paths["device_profile"].write_text(
        json.dumps(device_profile_row, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if preserve_existing_trials_jsonl:
        if not paths["trials_jsonl"].is_file():
            raise FileNotFoundError("recorder trials.jsonl is missing")
    else:
        write_jsonl(paths["trials_jsonl"], trial_rows)
    write_csv(paths["trials_csv"], trial_rows, TRIAL_RECORD_FIELDS)
    write_csv(paths["summary_csv"], summarize_trials(trial_rows), SUMMARY_FIELDS)
    write_csv(paths["failures_csv"], failure_rows(trial_rows), FAILURE_FIELDS)
    write_metadata(paths["metadata"], metadata)
    return paths


def _trial_row(trial: TrialRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trial, TrialRecord):
        return trial.to_dict()
    return {field: trial.get(field) for field in TRIAL_RECORD_FIELDS}


def _device_profile_row(
    device_profile: DeviceProfile | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(device_profile, DeviceProfile):
        return device_profile.to_dict()
    return dict(device_profile)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
