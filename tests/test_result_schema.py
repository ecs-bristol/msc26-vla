from __future__ import annotations

from libero_platform.result_schema import STEP_RECORD_FIELDS, StepRecord, TrialRecord


def test_step_and_trial_records_serialize_reset_and_action_diagnostics() -> None:
    step = StepRecord(
        run_id="run_1",
        episode_id=0,
        step_id=1,
        policy_latency_ms=1.5,
        service_latency_ms=None,
        transport_latency_ms=0.0,
        end_to_end_ms=1.5,
        action=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        raw_action=[1.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        action_transform="clip[-1,1]",
        action_clipped=True,
        action_valid=True,
        reward=1.0,
        done=True,
        success=True,
    )

    trial = TrialRecord.example(
        reset_seed=42,
        reset_initial_state_source="benchmark",
        reset_settle_steps=10,
        reset_fingerprint="0123456789abcdef",
    )

    assert step.to_dict()["raw_action"] == [1.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert step.to_dict()["action_transform"] == "clip[-1,1]"
    assert step.to_dict()["action_clipped"] is True
    assert trial.to_dict()["reset_seed"] == 42
    assert trial.to_dict()["reset_initial_state_source"] == "benchmark"
    assert trial.to_dict()["reset_settle_steps"] == 10
    assert trial.to_dict()["reset_fingerprint"] == "0123456789abcdef"


def test_step_schema_places_optional_service_latency_after_policy_latency() -> None:
    step = StepRecord(
        run_id="run_1",
        episode_id=0,
        step_id=1,
        policy_latency_ms=1.5,
        service_latency_ms=None,
        transport_latency_ms=None,
        end_to_end_ms=1.5,
        action=None,
        action_valid=False,
        reward=None,
        done=True,
        success=False,
    )

    old_csv_row = {"policy_latency_ms": "1.5", "end_to_end_ms": "1.5"}
    assert STEP_RECORD_FIELDS.index("service_latency_ms") == (
        STEP_RECORD_FIELDS.index("policy_latency_ms") + 1
    )
    assert step.to_dict()["service_latency_ms"] is None
    assert old_csv_row.get("service_latency_ms") is None
