from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from libero_platform.policies.base import PolicyRequest, PolicyResponse
from libero_platform.policy_parity import (
    PolicyParityError,
    build_policy_parity_summary,
    build_policy_repeatability_summary,
    write_policy_parity_artifacts,
)


def _request() -> PolicyRequest:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[1, 2] = [31, 127, 255]
    return PolicyRequest(
        run_id="policy_parity",
        episode_id=0,
        step_id=0,
        instruction="pick up the black bowl",
        images={"agentview": image},
        proprioception=np.arange(8, dtype=np.float32),
        previous_action=None,
    )


def _response(action: list[float], **kwargs: object) -> PolicyResponse:
    return PolicyResponse(
        action=np.asarray(action, dtype=np.float32),
        inference_ms=kwargs.pop("inference_ms", 12.5),
        model_key="smolvla_libero",
        device="cuda:0",
        **kwargs,
    )


def test_policy_parity_summary_records_input_hashes_and_action_delta() -> None:
    summary = build_policy_parity_summary(
        request=_request(),
        local_identity={"checkpoint": "HuggingFaceVLA/smolvla_libero", "revision": "abc"},
        remote_identity={"checkpoint": "HuggingFaceVLA/smolvla_libero", "revision": "abc"},
        local_response=_response([0.0] * 7),
        remote_response=_response([0.0, 0.0, 0.00001, 0.0, 0.0, 0.0, 0.0]),
        threshold=1e-4,
    )

    assert summary["status"] == "aligned"
    assert summary["action_valid"] is True
    assert summary["input"]["proprioception"]["shape"] == [8]
    assert summary["input"]["images"]["agentview"]["shape"] == [2, 3, 3]
    assert len(summary["input"]["images"]["agentview"]["sha256"]) == 64
    assert summary["delta"]["max_abs"] == pytest.approx(1e-5)
    assert summary["delta"]["mae"] == pytest.approx(1e-5 / 7)


def test_policy_parity_writes_auditable_summary_and_per_dimension_csv(
    tmp_path: Path,
) -> None:
    summary = build_policy_parity_summary(
        request=_request(),
        local_identity={"device": "cuda:0"},
        remote_identity={"device": "jetson"},
        local_response=_response([0.0] * 7),
        remote_response=_response([0.2] * 7),
        threshold=1e-4,
    )

    paths = write_policy_parity_artifacts(summary, tmp_path)

    stored_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert stored_summary["status"] == "diverged"
    with paths["actions"].open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 7
    assert rows[0]["dimension"] == "action_0"
    assert float(rows[0]["absolute_delta"]) == pytest.approx(0.2)


def test_policy_parity_rejects_policy_failures_before_reporting_actions() -> None:
    with pytest.raises(PolicyParityError, match="remote policy failed: remote_unavailable"):
        build_policy_parity_summary(
            request=_request(),
            local_identity={},
            remote_identity={},
            local_response=_response([0.0] * 7),
            remote_response=_response(
                [0.0] * 7,
                failure_type="remote_unavailable",
                error="connection refused",
            ),
            threshold=1e-4,
        )


def test_policy_repeatability_summary_reports_same_device_drift() -> None:
    summary = build_policy_repeatability_summary(
        first_response=_response([0.0] * 7),
        repeated_response=_response([0.0, 0.0, 0.002, 0.0, -0.001, 0.0, 0.0]),
    )

    assert summary["absolute_per_dimension"][2] == pytest.approx(0.002)
    assert summary["absolute_per_dimension"][4] == pytest.approx(0.001)
    assert summary["max_abs"] == pytest.approx(0.002)
    assert summary["mae"] == pytest.approx(0.003 / 7)


def test_policy_repeatability_summary_rejects_policy_failures() -> None:
    with pytest.raises(PolicyParityError, match="local policy failed"):
        build_policy_repeatability_summary(
            first_response=_response(
                [0.0] * 7,
                failure_type="local_failure",
                error="model error",
            ),
            repeated_response=_response([0.0] * 7),
        )
