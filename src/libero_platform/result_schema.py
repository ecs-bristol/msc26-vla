from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from statistics import mean
from typing import Any, Mapping, Sequence


STEP_RECORD_FIELDS = [
    "run_id",
    "episode_id",
    "step_id",
    "policy_latency_ms",
    "service_latency_ms",
    "transport_latency_ms",
    "end_to_end_ms",
    "action",
    "raw_action",
    "action_transform",
    "action_clipped",
    "action_valid",
    "reward",
    "done",
    "success",
]

TRIAL_RECORD_FIELDS = [
    "run_id",
    "suite",
    "task_id",
    "task_name",
    "initial_state_id",
    "episode_id",
    "seed",
    "reset_seed",
    "reset_initial_state_source",
    "reset_settle_steps",
    "reset_fingerprint",
    "instruction",
    "policy_key",
    "checkpoint",
    "deployment_mode",
    "device_profile",
    "precision",
    "quantization",
    "load_success",
    "success",
    "steps",
    "termination_reason",
    "action_valid",
    "policy_latency_mean_ms",
    "policy_latency_p95_ms",
    "end_to_end_mean_ms",
    "end_to_end_p95_ms",
    "peak_host_memory_mb",
    "peak_device_memory_mb",
    "oom",
    "failure_type",
    "error_summary",
    "video_path",
    "frame_directory",
]

SUMMARY_FIELDS = [
    "suite",
    "task_id",
    "policy_key",
    "deployment_mode",
    "precision",
    "quantization",
    "trials",
    "success_rate",
    "action_valid_rate",
    "oom_count",
    "load_failure_count",
    "policy_latency_mean_ms",
    "policy_latency_p95_ms",
    "end_to_end_mean_ms",
    "end_to_end_p95_ms",
    "peak_host_memory_max_mb",
    "peak_device_memory_max_mb",
]

FAILURE_FIELDS = [
    "run_id",
    "suite",
    "task_id",
    "task_name",
    "initial_state_id",
    "episode_id",
    "seed",
    "policy_key",
    "checkpoint",
    "deployment_mode",
    "device_profile",
    "precision",
    "quantization",
    "load_success",
    "oom",
    "termination_reason",
    "failure_type",
    "error_summary",
    "video_path",
    "frame_directory",
]


@dataclass(frozen=True)
class StepRecord:
    run_id: str
    episode_id: int
    step_id: int
    policy_latency_ms: float | None
    transport_latency_ms: float | None
    end_to_end_ms: float | None
    action: list[float] | None
    action_valid: bool
    reward: float | None
    done: bool
    success: bool | None
    service_latency_ms: float | None = None
    raw_action: list[float] | None = None
    action_transform: str = ""
    action_clipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrialRecord:
    run_id: str
    suite: str
    task_id: int
    task_name: str
    initial_state_id: int
    episode_id: int
    seed: int
    instruction: str
    policy_key: str
    checkpoint: str
    deployment_mode: str
    device_profile: str
    precision: str
    quantization: str
    load_success: bool
    success: bool
    steps: int
    termination_reason: str
    action_valid: bool
    policy_latency_mean_ms: float | None
    policy_latency_p95_ms: float | None
    end_to_end_mean_ms: float | None
    end_to_end_p95_ms: float | None
    peak_host_memory_mb: float | None
    peak_device_memory_mb: float | None
    oom: bool
    failure_type: str
    error_summary: str
    video_path: str | None
    frame_directory: str | None
    reset_seed: int | None = None
    reset_initial_state_source: str = ""
    reset_settle_steps: int = 0
    reset_fingerprint: str = ""

    @classmethod
    def example(cls, **overrides: Any) -> TrialRecord:
        record = cls(
            run_id="run_1",
            suite="libero_spatial",
            task_id=0,
            task_name="pick_up_the_black_bowl",
            initial_state_id=0,
            episode_id=0,
            seed=42,
            instruction="pick up the black bowl",
            policy_key="zero_policy",
            checkpoint="none",
            deployment_mode="pc_local",
            device_profile="pc_default",
            precision="none",
            quantization="none",
            load_success=True,
            success=False,
            steps=1,
            termination_reason="max_steps",
            action_valid=True,
            policy_latency_mean_ms=None,
            policy_latency_p95_ms=None,
            end_to_end_mean_ms=None,
            end_to_end_p95_ms=None,
            peak_host_memory_mb=None,
            peak_device_memory_mb=None,
            oom=False,
            failure_type="",
            error_summary="",
            video_path=None,
            frame_directory=None,
            reset_seed=42,
            reset_initial_state_source="fake",
            reset_settle_steps=0,
            reset_fingerprint="0123456789abcdef",
        )
        return replace(record, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceProfile:
    device_model: str
    host_os: str | None = None
    cpu_model: str | None = None
    cpu_count: int | None = None
    host_memory_total_mb: float | None = None
    device_name: str | None = None
    device_memory_total_mb: float | None = None
    peak_host_memory_mb: float | None = None
    peak_device_memory_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: Sequence[float], percentile_value: int | float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(percentile_value) / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def summarize_trials(
    trials: Sequence[TrialRecord | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str, str, str], list[dict[str, Any]]] = {}
    for trial in trials:
        row = _trial_row(trial)
        key = (
            str(row["suite"]),
            int(row["task_id"]),
            str(row["policy_key"]),
            str(row["deployment_mode"]),
            str(row["precision"]),
            str(row["quantization"]),
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        policy_latencies = _available_floats(rows, "policy_latency_mean_ms")
        end_to_end_latencies = _available_floats(rows, "end_to_end_mean_ms")
        host_memories = _available_floats(rows, "peak_host_memory_mb")
        device_memories = _available_floats(rows, "peak_device_memory_mb")
        summaries.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "policy_key": key[2],
                "deployment_mode": key[3],
                "precision": key[4],
                "quantization": key[5],
                "trials": len(rows),
                "success_rate": sum(_as_bool(row["success"]) for row in rows) / len(rows),
                "action_valid_rate": sum(_as_bool(row["action_valid"]) for row in rows)
                / len(rows),
                "oom_count": sum(_as_bool(row["oom"]) for row in rows),
                "load_failure_count": sum(not _as_bool(row["load_success"]) for row in rows),
                "policy_latency_mean_ms": mean(policy_latencies) if policy_latencies else None,
                "policy_latency_p95_ms": percentile(policy_latencies, 95),
                "end_to_end_mean_ms": mean(end_to_end_latencies)
                if end_to_end_latencies
                else None,
                "end_to_end_p95_ms": percentile(end_to_end_latencies, 95),
                "peak_host_memory_max_mb": max(host_memories) if host_memories else None,
                "peak_device_memory_max_mb": max(device_memories) if device_memories else None,
            }
        )
    return summaries


def failure_rows(
    trials: Sequence[TrialRecord | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {field: row.get(field) for field in FAILURE_FIELDS}
        for trial in trials
        for row in [_trial_row(trial)]
        if not _as_bool(row["success"])
    ]


def _available_floats(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _trial_row(trial: TrialRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trial, TrialRecord):
        return trial.to_dict()
    return {field: trial.get(field) for field in TRIAL_RECORD_FIELDS}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
