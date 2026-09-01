from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .action_schema import ACTION_VOCABULARY


DEPLOYMENT_MODES = [
    "pc_local",
    "pc_remote_server",
    "jetson_local",
    "jetson_quantized",
    "jetson_remote_client",
    "remote_server",
    "mock",
]


TRIAL_RECORD_FIELDS = [
    "run_id",
    "experiment",
    "timestamp",
    "repeat_idx",
    "task_id",
    "image_path",
    "instruction",
    "expected_target",
    "expected_action",
    "model_key",
    "model_id",
    "adapter",
    "deployment_mode",
    "device_profile",
    "runtime_precision",
    "quantization",
    "local_files_only",
    "load_success",
    "oom",
    "success",
    "failure_type",
    "error",
    "target",
    "action_text",
    "action_vector",
    "parsed_actions",
    "action_valid",
    "expected_action_found",
    "target_mentioned",
    "auto_score_pass",
    "confidence",
    "raw_output",
    "latency_ms",
    "model_load_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "end_to_end_ms",
    "peak_memory_mb",
    "cpu_percent",
    "gpu_percent",
    "power_w",
    "notes",
]


@dataclass(frozen=True)
class AdapterRequest:
    task_id: str
    image_path: Path
    instruction: str
    expected_target: str
    expected_action: str
    model_key: str
    model_id: str
    adapter: str
    deployment_mode: str
    runtime_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterResponse:
    model_key: str
    adapter: str
    deployment_mode: str
    task_id: str
    target: str = ""
    action_text: str = ""
    action_vector: list[float] | None = None
    confidence: float | None = None
    raw_output: str = ""
    parsed_actions: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    model_load_ms: float | None = None
    preprocess_ms: float | None = None
    inference_ms: float | None = None
    postprocess_ms: float | None = None
    end_to_end_ms: float | None = None
    peak_memory_mb: float | None = None
    success: bool = False
    load_success: bool = True
    oom: bool = False
    failure_type: str = ""
    error: str = ""


def parse_action_tokens(text: str) -> list[str]:
    lowered = text.lower()
    return [
        action
        for action in ACTION_VOCABULARY
        if _contains_token(lowered, action) or _contains_token(lowered, action.replace("_", " "))
    ]


def score_text_output(
    *,
    expected_target: str,
    expected_action: str,
    output_text: str,
    parsed_actions: list[str] | None = None,
) -> dict[str, Any]:
    lowered = output_text.lower()
    actions = parsed_actions or parse_action_tokens(output_text)
    expected_action_norm = expected_action.lower().strip()
    expected_target_norm = expected_target.lower().strip()
    expected_action_found = bool(expected_action_norm and expected_action_norm in actions)
    target_mentioned = bool(expected_target_norm and _contains_token(lowered, expected_target_norm))
    action_valid = bool(actions)
    return {
        "parsed_actions": actions,
        "action_valid": action_valid,
        "expected_action_found": expected_action_found,
        "target_mentioned": target_mentioned,
        "auto_score_pass": bool(
            action_valid
            and (not expected_action_norm or expected_action_found)
            and (not expected_target_norm or target_mentioned)
        ),
    }


def _contains_token(text: str, token: str) -> bool:
    escaped = re.escape(token.strip().lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", text))


def _join_actions(actions: list[str]) -> str:
    return "|".join(actions)


def _join_vector(vector: list[float] | None) -> str:
    if vector is None:
        return ""
    return "|".join(str(value) for value in vector)


def make_trial_record(
    *,
    run_id: str,
    experiment: str,
    timestamp: str,
    repeat_idx: int,
    request: AdapterRequest,
    response: AdapterResponse,
    device_profile: str,
    notes: str = "",
) -> dict[str, Any]:
    output_for_score = response.raw_output or response.action_text
    score = score_text_output(
        expected_target=request.expected_target,
        expected_action=request.expected_action,
        output_text=output_for_score,
        parsed_actions=response.parsed_actions,
    )
    runtime_config = request.runtime_config
    record = {
        "run_id": run_id,
        "experiment": experiment,
        "timestamp": timestamp,
        "repeat_idx": repeat_idx,
        "task_id": request.task_id,
        "image_path": str(request.image_path),
        "instruction": request.instruction,
        "expected_target": request.expected_target,
        "expected_action": request.expected_action,
        "model_key": request.model_key,
        "model_id": request.model_id,
        "adapter": request.adapter,
        "deployment_mode": request.deployment_mode,
        "device_profile": device_profile,
        "runtime_precision": str(runtime_config.get("runtime_precision", runtime_config.get("dtype", ""))),
        "quantization": str(runtime_config.get("quantization", "")),
        "local_files_only": bool(runtime_config.get("local_files_only", False)),
        "load_success": bool(response.load_success),
        "oom": bool(response.oom or response.failure_type == "oom"),
        "success": bool(response.success and score["auto_score_pass"]),
        "failure_type": response.failure_type,
        "error": response.error,
        "target": response.target,
        "action_text": response.action_text,
        "action_vector": _join_vector(response.action_vector),
        "parsed_actions": _join_actions(score["parsed_actions"]),
        "action_valid": bool(score["action_valid"]),
        "expected_action_found": bool(score["expected_action_found"]),
        "target_mentioned": bool(score["target_mentioned"]),
        "auto_score_pass": bool(score["auto_score_pass"]),
        "confidence": response.confidence,
        "raw_output": response.raw_output,
        "latency_ms": response.latency_ms,
        "model_load_ms": response.model_load_ms,
        "preprocess_ms": response.preprocess_ms,
        "inference_ms": response.inference_ms,
        "postprocess_ms": response.postprocess_ms,
        "end_to_end_ms": response.end_to_end_ms,
        "peak_memory_mb": response.peak_memory_mb,
        "cpu_percent": runtime_config.get("cpu_percent", ""),
        "gpu_percent": runtime_config.get("gpu_percent", ""),
        "power_w": runtime_config.get("power_w", ""),
        "notes": notes,
    }
    return {field_name: record[field_name] for field_name in TRIAL_RECORD_FIELDS}
