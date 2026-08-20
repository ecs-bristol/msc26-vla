"""Evidence helpers for one-observation local-versus-remote policy comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .policies.base import PolicyRequest, PolicyResponse, validate_action


class PolicyParityError(RuntimeError):
    """Raised when a paired policy comparison cannot produce valid evidence."""


def build_policy_parity_summary(
    *,
    request: PolicyRequest,
    local_identity: Mapping[str, object],
    remote_identity: Mapping[str, object],
    local_response: PolicyResponse,
    remote_response: PolicyResponse,
    threshold: float,
) -> dict[str, object]:
    """Return a JSON-safe parity report for exactly one shared policy request."""

    if threshold < 0.0 or not np.isfinite(threshold):
        raise PolicyParityError("threshold must be a finite non-negative value")
    _raise_for_policy_failure("local", local_response)
    _raise_for_policy_failure("remote", remote_response)
    try:
        local_action = validate_action(local_response.action)
        remote_action = validate_action(remote_response.action)
    except ValueError as exc:
        raise PolicyParityError(f"invalid parity action: {exc}") from exc

    delta = remote_action - local_action
    absolute_delta = np.abs(delta)
    max_abs = float(absolute_delta.max())
    return {
        "schema_version": 1,
        "status": "aligned" if max_abs <= threshold else "diverged",
        "threshold": float(threshold),
        "action_valid": True,
        "input": _request_evidence(request),
        "local": _response_evidence(local_identity, local_response, local_action),
        "remote": _response_evidence(remote_identity, remote_response, remote_action),
        "delta": {
            "per_dimension": [float(value) for value in delta],
            "absolute_per_dimension": [float(value) for value in absolute_delta],
            "mae": float(absolute_delta.mean()),
            "max_abs": max_abs,
        },
    }


def build_policy_repeatability_summary(
    *, first_response: PolicyResponse, repeated_response: PolicyResponse
) -> dict[str, object]:
    """Summarize action drift when one policy repeats an identical request."""

    _raise_for_policy_failure("local", first_response)
    _raise_for_policy_failure("local", repeated_response)
    try:
        first_action = validate_action(first_response.action)
        repeated_action = validate_action(repeated_response.action)
    except ValueError as exc:
        raise PolicyParityError(f"invalid repeatability action: {exc}") from exc

    delta = repeated_action - first_action
    absolute_delta = np.abs(delta)
    return {
        "first_action": [float(value) for value in first_action],
        "repeated_action": [float(value) for value in repeated_action],
        "per_dimension": [float(value) for value in delta],
        "absolute_per_dimension": [float(value) for value in absolute_delta],
        "mae": float(absolute_delta.mean()),
        "max_abs": float(absolute_delta.max()),
    }


def write_policy_parity_artifacts(
    summary: Mapping[str, object], output_directory: Path) -> dict[str, Path]:
    """Persist the report and a compact seven-dimension action comparison."""

    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "summary.json"
    actions_path = output_directory / "actions.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    local_action = _action_from_summary(summary, "local")
    remote_action = _action_from_summary(summary, "remote")
    delta = _float_sequence(summary, "delta", "per_dimension")
    absolute_delta = _float_sequence(summary, "delta", "absolute_per_dimension")
    with actions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "dimension",
                "local_action",
                "remote_action",
                "delta",
                "absolute_delta",
            ),
        )
        writer.writeheader()
        for index in range(7):
            writer.writerow(
                {
                    "dimension": f"action_{index}",
                    "local_action": local_action[index],
                    "remote_action": remote_action[index],
                    "delta": delta[index],
                    "absolute_delta": absolute_delta[index],
                }
            )
    return {"summary": summary_path, "actions": actions_path}


def _raise_for_policy_failure(label: str, response: PolicyResponse) -> None:
    if response.failure_type:
        raise PolicyParityError(
            f"{label} policy failed: {response.failure_type}"
            + (f" ({response.error})" if response.error else "")
        )


def _request_evidence(request: PolicyRequest) -> dict[str, object]:
    return {
        "instruction": request.instruction,
        "proprioception": _array_evidence(request.proprioception),
        "images": {
            key: _array_evidence(value)
            for key, value in sorted(request.images.items())
        },
    }


def _response_evidence(
    identity: Mapping[str, object], response: PolicyResponse, action: np.ndarray
) -> dict[str, object]:
    return {
        "identity": _json_identity(identity),
        "action": [float(value) for value in action],
        "inference_ms": float(response.inference_ms),
        "model_key": response.model_key,
        "device": response.device,
        "metadata": _json_identity(response.metadata),
    }


def _array_evidence(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _json_identity(values: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): value
        for key, value in values.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def _action_from_summary(summary: Mapping[str, object], side: str) -> list[float]:
    side_value = summary.get(side)
    if not isinstance(side_value, Mapping):
        raise PolicyParityError(f"summary is missing {side} evidence")
    action = side_value.get("action")
    if not isinstance(action, list) or len(action) != 7:
        raise PolicyParityError(f"summary contains an invalid {side} action")
    return [float(value) for value in action]


def _float_sequence(
    summary: Mapping[str, object], section: str, key: str
) -> list[float]:
    section_value = summary.get(section)
    if not isinstance(section_value, Mapping):
        raise PolicyParityError(f"summary is missing {section} evidence")
    values = section_value.get(key)
    if not isinstance(values, list) or len(values) != 7:
        raise PolicyParityError(f"summary contains invalid {section}.{key}")
    return [float(value) for value in values]
