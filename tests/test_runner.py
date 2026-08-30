from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from libero_platform.backends.fake_backend import FakeBackend
from libero_platform.policies.base import (
    EpisodeContext,
    PolicyAdapter,
    PolicyRequest,
    PolicyResponse,
)
from libero_platform.policies.zero_policy import ZeroPolicyAdapter
from libero_platform.recorder import RunRecorder
from libero_platform.runner import RunnerDependencies, run_experiment
from libero_platform.spec import ResolvedExperimentSpec


VALID = {
    "schema_version": 1,
    "name": "fake_smoke",
    "benchmark": {
        "backend": "fake",
        "suite": "libero_spatial",
        "task_ids": [0],
        "initial_state_ids": [0],
        "max_steps": 5,
    },
    "policy": {
        "key": "zero_policy",
        "checkpoint": "none",
        "precision": "none",
        "quantization": "none",
    },
    "deployment": {"mode": "pc_local", "profile": "pc_default"},
    "execution": {
        "episodes_per_initial_state": 1,
        "warmup_episodes": 0,
        "seed": 42,
        "on_episode_failure": "continue",
    },
    "viewer": {"enabled": False},
    "recording": {
        "save_frames": False,
        "save_video": False,
        "frame_stride": 20,
        "save_steps": True,
    },
}


@pytest.fixture
def recorder(tmp_path: Path) -> RunRecorder:
    return RunRecorder(tmp_path / "outputs")


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text("schema_version: 1\nname: fake_smoke\n", encoding="utf-8")
    return path


@pytest.fixture
def resolved_spec(source_path: Path) -> ResolvedExperimentSpec:
    return ResolvedExperimentSpec.model_validate(
        {
            **VALID,
            "source_path": str(source_path),
            "dataset_directory": "datasets/libero",
            "resolved_checkpoint": "none",
            "resolved_revision": "",
            "policy_adapter": "zero",
        }
    )


def test_fake_run_records_two_initial_states(resolved_spec, recorder) -> None:
    spec = resolved_spec.model_copy(
        update={
            "benchmark": resolved_spec.benchmark.model_copy(
                update={"initial_state_ids": [0, 1], "max_steps": 3}
            )
        }
    )

    outcome = run_experiment(
        spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=2),
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(spec.source_path),
        ),
    )

    assert outcome.status == "completed"
    assert outcome.result_integrity == "complete"
    assert len(outcome.trials) == 2
    assert all(trial.success for trial in outcome.trials)

    manifest = recorder.read_manifest(outcome.run_id)
    assert (manifest["status"], manifest["result_integrity"]) == ("completed", "complete")
    assert len((outcome.run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    assert (outcome.run_dir / "metadata.json").is_file()
    assert (outcome.run_dir / "summary.csv").is_file()


def test_stop_failure_policy_skips_remaining_episodes(resolved_spec, recorder) -> None:
    spec = resolved_spec.model_copy(
        update={
            "benchmark": resolved_spec.benchmark.model_copy(
                update={"initial_state_ids": [0, 1], "max_steps": 3}
            ),
            "execution": resolved_spec.execution.model_copy(
                update={"on_episode_failure": "stop"}
            ),
        }
    )

    outcome = run_experiment(
        spec,
        RunnerDependencies(
            backend=FakeBackend(fail_episode=0),
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(spec.source_path),
        ),
    )

    assert outcome.status == "failed"
    assert outcome.result_integrity == "partial"
    assert len(outcome.trials) == 1
    assert outcome.trials[0].success is False


def test_runner_begins_each_episode_after_reset_before_predict(
    resolved_spec: ResolvedExperimentSpec, recorder: RunRecorder
) -> None:
    events: list[str] = []

    class RecordingPolicy(PolicyAdapter):
        def begin_episode(self, context: EpisodeContext) -> None:
            assert context == EpisodeContext(
                suite="libero_spatial",
                task_id=0,
                task_name="fake_task_0",
                initial_state_id=0,
                seed=42,
            )
            events.append("begin")

        def predict(self, request: PolicyRequest):
            assert events == ["reset", "begin"]
            events.append("predict")
            return ZeroPolicyAdapter("recording_policy").predict(request)

    class RecordingBackend(FakeBackend):
        def open_episode(self, *args, **kwargs):
            episode = super().open_episode(*args, **kwargs)
            original_reset = episode.reset

            def reset():
                events.append("reset")
                return original_reset()

            episode.reset = reset
            return episode

    outcome = run_experiment(
        resolved_spec,
        RunnerDependencies(
            backend=RecordingBackend(success_step=1),
            policy=RecordingPolicy(),
            recorder=recorder,
            source_path=Path(resolved_spec.source_path),
        ),
    )

    assert outcome.status == "completed"
    assert events == ["reset", "begin", "predict"]


def test_runner_passes_each_trial_seed_to_backend(resolved_spec, recorder) -> None:
    class RecordingBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.opened_seeds: list[int] = []

        def open_episode(
            self, suite: str, task_id: int, initial_state_id: int, max_steps: int, seed: int
        ):
            self.opened_seeds.append(seed)
            return super().open_episode(suite, task_id, initial_state_id, max_steps, seed)

    backend = RecordingBackend()
    spec = resolved_spec.model_copy(
        update={
            "execution": resolved_spec.execution.model_copy(
                update={"episodes_per_initial_state": 2, "seed": 42}
            )
        }
    )

    run_experiment(
        spec,
        RunnerDependencies(
            backend=backend,
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(spec.source_path),
        ),
    )

    assert backend.opened_seeds == [42, 43]


def test_runner_passes_warmup_seed_to_backend(resolved_spec, recorder) -> None:
    class RecordingBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.opened_seeds: list[int] = []

        def open_episode(
            self, suite: str, task_id: int, initial_state_id: int, max_steps: int, seed: int
        ):
            self.opened_seeds.append(seed)
            return super().open_episode(suite, task_id, initial_state_id, max_steps, seed)

    backend = RecordingBackend()
    spec = resolved_spec.model_copy(
        update={
            "execution": resolved_spec.execution.model_copy(
                update={"warmup_episodes": 2, "seed": 42}
            )
        }
    )

    run_experiment(
        spec,
        RunnerDependencies(
            backend=backend,
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(spec.source_path),
        ),
    )

    assert backend.opened_seeds == [42, 43, 42]


def test_fake_run_evidence_is_deterministic(resolved_spec, recorder) -> None:
    outcome = run_experiment(
        resolved_spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=2),
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(resolved_spec.source_path),
        ),
    )

    step_records = [
        json.loads(line)
        for line in (outcome.run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["success"] for record in step_records] == [False, True]
    assert [record["action"] for record in step_records] == [[0.0] * 6 + [-1.0]] * 2
    assert [record["transport_latency_ms"] for record in step_records] == [None, None]


def test_runner_persists_reset_and_action_diagnostics(
    resolved_spec: ResolvedExperimentSpec, recorder: RunRecorder
) -> None:
    class ClippingPolicy(PolicyAdapter):
        def predict(self, request: PolicyRequest) -> PolicyResponse:
            del request
            return PolicyResponse(
                action=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
                raw_action=[1.2, 0, 0, 0, 0, 0, -1],
                action_transform="clip[-1,1]",
                action_clipped=True,
                inference_ms=2.5,
                model_key="clipping_policy",
                device="cpu",
            )

    outcome = run_experiment(
        resolved_spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=1),
            policy=ClippingPolicy(),
            recorder=recorder,
            source_path=Path(resolved_spec.source_path),
        ),
    )

    trial = outcome.trials[0]
    assert trial.reset_seed == 42
    assert isinstance(trial.reset_fingerprint, str)
    assert len(trial.reset_fingerprint) == 16

    persisted_step = json.loads(
        (outcome.run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert persisted_step["raw_action"] == [1.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert persisted_step["action"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert persisted_step["action_transform"] == "clip[-1,1]"
    assert persisted_step["action_clipped"] is True

    persisted_trial = json.loads(
        (outcome.run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert persisted_trial["reset_seed"] == 42
    assert isinstance(persisted_trial["reset_fingerprint"], str)
    assert len(persisted_trial["reset_fingerprint"]) == 16


def test_runner_saves_stride_and_terminal_frames_when_enabled(
    resolved_spec: ResolvedExperimentSpec, recorder: RunRecorder
) -> None:
    spec = resolved_spec.model_copy(
        update={
            "benchmark": resolved_spec.benchmark.model_copy(update={"max_steps": 3}),
            "recording": resolved_spec.recording.model_copy(
                update={"save_frames": True, "frame_stride": 2}
            ),
        }
    )

    outcome = run_experiment(
        spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=3),
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(spec.source_path),
        ),
    )

    frame_directory = outcome.run_dir / "frames" / "episode_0000"
    assert outcome.trials[0].frame_directory == "frames/episode_0000"
    assert sorted(path.name for path in frame_directory.glob("*.png")) == [
        "step_000002.png",
        "step_000003.png",
    ]


def test_policy_failure_response_does_not_step_and_stops_the_run(
    resolved_spec: ResolvedExperimentSpec, recorder: RunRecorder
) -> None:
    class FailurePolicy(PolicyAdapter):
        def predict(self, request: PolicyRequest) -> PolicyResponse:
            return PolicyResponse(
                action=ZeroPolicyAdapter("zero_policy").predict(request).action,
                inference_ms=3.0,
                model_key="remote_policy",
                device="remote",
                failure_type="remote_unavailable",
                error="connection refused",
                metadata={"service_latency_ms": 12.5, "remote_round_trip_ms": 43.0},
            )

    class StepForbiddenBackend(FakeBackend):
        def open_episode(self, *args, **kwargs):
            episode = super().open_episode(*args, **kwargs)

            def step(_action):
                raise AssertionError("failure responses must not step the environment")

            episode.step = step
            return episode

    spec = resolved_spec.model_copy(
        update={
            "benchmark": resolved_spec.benchmark.model_copy(
                update={"initial_state_ids": [0, 1]}
            ),
            "deployment": resolved_spec.deployment.model_copy(update={"mode": "remote"}),
            "execution": resolved_spec.execution.model_copy(
                update={"on_episode_failure": "stop"}
            ),
        }
    )
    outcome = run_experiment(
        spec,
        RunnerDependencies(
            backend=StepForbiddenBackend(),
            policy=FailurePolicy(),
            recorder=recorder,
            source_path=Path(spec.source_path),
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ("failed", "partial")
    assert len(outcome.trials) == 1
    trial = outcome.trials[0]
    assert (trial.success, trial.steps, trial.action_valid) == (False, 0, False)
    assert trial.termination_reason == "policy_failure"
    assert trial.failure_type == "remote_unavailable"
    assert trial.error_summary == "connection refused"
    persisted_step = json.loads(
        (outcome.run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert persisted_step["service_latency_ms"] == 12.5
    assert persisted_step["transport_latency_ms"] == 30.5
    assert persisted_step["end_to_end_ms"] == 43.0
    events = [
        json.loads(line)
        for line in (outcome.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["event"] == "policy_failure"
        and event["failure_type"] == "remote_unavailable"
        and event["error_summary"] == "connection refused"
        for event in events
    )


def test_runner_records_remote_service_transport_and_round_trip_latency(
    resolved_spec: ResolvedExperimentSpec, recorder: RunRecorder
) -> None:
    class RemoteTimingPolicy(PolicyAdapter):
        def predict(self, request: PolicyRequest) -> PolicyResponse:
            return PolicyResponse(
                action=ZeroPolicyAdapter("zero_policy").predict(request).action,
                inference_ms=99.0,
                model_key="remote_policy",
                device="remote",
                metadata={"service_latency_ms": 12.5, "remote_round_trip_ms": 43.0},
            )

    spec = resolved_spec.model_copy(
        update={"deployment": resolved_spec.deployment.model_copy(update={"mode": "remote"})}
    )
    outcome = run_experiment(
        spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=1),
            policy=RemoteTimingPolicy(),
            recorder=recorder,
            source_path=Path(spec.source_path),
        ),
    )

    step = json.loads((outcome.run_dir / "steps.jsonl").read_text(encoding="utf-8"))
    assert step["policy_latency_ms"] == 12.5
    assert step["service_latency_ms"] == 12.5
    assert step["transport_latency_ms"] == 30.5
    assert step["end_to_end_ms"] == 43.0
    assert outcome.trials[0].policy_latency_mean_ms == 12.5
    assert outcome.trials[0].end_to_end_mean_ms == 43.0


def test_runner_ignores_invalid_remote_timing_metadata_on_policy_failure(
    resolved_spec: ResolvedExperimentSpec, recorder: RunRecorder
) -> None:
    class FailedRemoteTimingPolicy(PolicyAdapter):
        def predict(self, request: PolicyRequest) -> PolicyResponse:
            return PolicyResponse(
                action=ZeroPolicyAdapter("zero_policy").predict(request).action,
                inference_ms=7.0,
                model_key="remote_policy",
                device="remote",
                failure_type="remote_unavailable",
                error="connection refused",
                metadata={"service_latency_ms": float("nan"), "remote_round_trip_ms": "bad"},
            )

    spec = resolved_spec.model_copy(
        update={"deployment": resolved_spec.deployment.model_copy(update={"mode": "remote"})}
    )
    outcome = run_experiment(
        spec,
        RunnerDependencies(
            backend=FakeBackend(),
            policy=FailedRemoteTimingPolicy(),
            recorder=recorder,
            source_path=Path(spec.source_path),
        ),
    )

    step = json.loads((outcome.run_dir / "steps.jsonl").read_text(encoding="utf-8"))
    assert step["policy_latency_ms"] == 7.0
    assert step["service_latency_ms"] is None
    assert step["transport_latency_ms"] is None
    assert step["end_to_end_ms"] == 7.0


def test_continue_run_with_unsuccessful_trials_has_complete_evidence(
    resolved_spec: ResolvedExperimentSpec, recorder: RunRecorder
) -> None:
    spec = resolved_spec.model_copy(
        update={
            "benchmark": resolved_spec.benchmark.model_copy(
                update={"initial_state_ids": [0, 1], "max_steps": 2}
            )
        }
    )
    outcome = run_experiment(
        spec,
        RunnerDependencies(
            backend=FakeBackend(fail_episode=0),
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(spec.source_path),
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ("completed", "complete")
    assert [trial.success for trial in outcome.trials] == [False, True]
    manifest = recorder.read_manifest(outcome.run_id)
    assert (manifest["status"], manifest["result_integrity"]) == ("completed", "complete")


def test_artifact_failure_keeps_trial_evidence_and_final_metadata_consistent(
    resolved_spec: ResolvedExperimentSpec,
    recorder: RunRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from libero_platform import result_writer

    real_write_csv = result_writer.write_csv

    def fail_summary_csv(path, rows, fieldnames) -> None:
        if path.name == "summary.csv":
            raise OSError("summary disk failure")
        real_write_csv(path, rows, fieldnames)

    monkeypatch.setattr(result_writer, "write_csv", fail_summary_csv)
    outcome = run_experiment(
        resolved_spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=1),
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(resolved_spec.source_path),
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ("failed", "unavailable")
    manifest = recorder.read_manifest(outcome.run_id)
    metadata = json.loads((outcome.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert (metadata["status"], metadata["result_integrity"]) == (
        manifest["status"],
        manifest["result_integrity"],
    )
    trials = [
        json.loads(line)
        for line in (outcome.run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(trials) == 1
    assert trials[0]["success"] is True


def test_keyboard_interrupt_stops_run_and_preserves_parseable_evidence(
    resolved_spec: ResolvedExperimentSpec, recorder: RunRecorder
) -> None:
    events: list[dict[str, object]] = []

    class InterruptingPolicy(PolicyAdapter):
        def __init__(self) -> None:
            self._predict_calls = 0

        def predict(self, request: PolicyRequest) -> PolicyResponse:
            self._predict_calls += 1
            if self._predict_calls == 2:
                raise KeyboardInterrupt
            return ZeroPolicyAdapter("interrupting_policy").predict(request)

    spec = resolved_spec.model_copy(
        update={
            "benchmark": resolved_spec.benchmark.model_copy(
                update={"initial_state_ids": [0, 1]}
            )
        }
    )
    outcome = run_experiment(
        spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=1),
            policy=InterruptingPolicy(),
            recorder=recorder,
            source_path=Path(spec.source_path),
            event_handler=events.append,
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ("stopped", "partial")
    manifest = recorder.read_manifest(outcome.run_id)
    assert (manifest["status"], manifest["result_integrity"]) == ("stopped", "partial")
    assert events[-1]["event"] == "run_stopped"
    for name in ("steps.jsonl", "trials.jsonl"):
        lines = (outcome.run_dir / name).read_text(encoding="utf-8").splitlines()
        assert lines
        for line in lines:
            assert isinstance(json.loads(line), dict)


def test_keyboard_interrupt_during_policy_cleanup_stops_and_finalizes_run(
    resolved_spec: ResolvedExperimentSpec, recorder: RunRecorder
) -> None:
    events: list[dict[str, object]] = []

    class ClosingInterruptPolicy(ZeroPolicyAdapter):
        def close(self) -> None:
            raise KeyboardInterrupt

    outcome = run_experiment(
        resolved_spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=1),
            policy=ClosingInterruptPolicy("closing_interrupt"),
            recorder=recorder,
            source_path=Path(resolved_spec.source_path),
            event_handler=events.append,
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ("stopped", "partial")
    manifest = recorder.read_manifest(outcome.run_id)
    assert (manifest["status"], manifest["result_integrity"]) == ("stopped", "partial")
    assert events[-1]["event"] == "run_stopped"


def test_event_write_failure_during_error_handling_still_finalizes_run(
    resolved_spec: ResolvedExperimentSpec, tmp_path: Path
) -> None:
    class EventFailingRecorder(RunRecorder):
        def append_event(self, run_id, event, **fields):
            del run_id, event, fields
            raise OSError("event storage unavailable")

    class LoadingFailurePolicy(PolicyAdapter):
        def load(self) -> None:
            raise RuntimeError("model startup failed")

        def predict(self, request: PolicyRequest) -> PolicyResponse:
            raise AssertionError("predict should not be called")

    recorder = EventFailingRecorder(tmp_path / "outputs")
    outcome = run_experiment(
        resolved_spec,
        RunnerDependencies(
            backend=FakeBackend(),
            policy=LoadingFailurePolicy(),
            recorder=recorder,
            source_path=Path(resolved_spec.source_path),
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ("failed", "partial")
    manifest = recorder.read_manifest(outcome.run_id)
    assert (manifest["status"], manifest["result_integrity"]) == ("failed", "partial")


def test_finalization_failure_is_retried_as_unavailable_terminal_run(
    resolved_spec: ResolvedExperimentSpec, tmp_path: Path
) -> None:
    class FinalizeFailsOnceRecorder(RunRecorder):
        def __init__(self, output_root: Path) -> None:
            super().__init__(output_root)
            self._finalize_calls = 0

        def finalize(self, *args, **kwargs):
            self._finalize_calls += 1
            if self._finalize_calls == 1:
                raise OSError("terminal manifest write failed")
            return super().finalize(*args, **kwargs)

    events: list[dict[str, object]] = []
    recorder = FinalizeFailsOnceRecorder(tmp_path / "outputs")
    outcome = run_experiment(
        resolved_spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=1),
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(resolved_spec.source_path),
            event_handler=events.append,
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ("failed", "unavailable")
    manifest = recorder.read_manifest(outcome.run_id)
    assert (manifest["status"], manifest["result_integrity"]) == ("failed", "unavailable")
    assert events[-1]["event"] == "run_failed"


def test_keyboard_interrupt_during_artifact_persistence_stops_and_finalizes_run(
    resolved_spec: ResolvedExperimentSpec,
    recorder: RunRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from libero_platform import runner

    def interrupt_artifact_persistence(**kwargs) -> None:
        del kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "write_required_artifacts", interrupt_artifact_persistence)
    outcome = run_experiment(
        resolved_spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=1),
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(resolved_spec.source_path),
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ("stopped", "partial")
    manifest = recorder.read_manifest(outcome.run_id)
    assert (manifest["status"], manifest["result_integrity"]) == ("stopped", "partial")
    assert (outcome.run_dir / "trials.jsonl").read_text(encoding="utf-8")


def test_keyboard_interrupt_during_artifact_manifest_update_stops_and_finalizes_run(
    resolved_spec: ResolvedExperimentSpec, tmp_path: Path
) -> None:
    class ArtifactManifestInterruptRecorder(RunRecorder):
        def update_manifest(self, run_id, /, **updates):
            if "artifacts" in updates:
                raise KeyboardInterrupt
            return super().update_manifest(run_id, **updates)

    recorder = ArtifactManifestInterruptRecorder(tmp_path / "outputs")
    outcome = run_experiment(
        resolved_spec,
        RunnerDependencies(
            backend=FakeBackend(success_step=1),
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(resolved_spec.source_path),
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ("stopped", "partial")
    manifest = recorder.read_manifest(outcome.run_id)
    assert (manifest["status"], manifest["result_integrity"]) == ("stopped", "partial")
    assert (outcome.run_dir / "summary.csv").is_file()


def test_running_lifecycle_update_failure_returns_unavailable_terminal_outcome(
    resolved_spec: ResolvedExperimentSpec, tmp_path: Path
) -> None:
    class RunningUpdateFailsOnceRecorder(RunRecorder):
        def __init__(self, output_root: Path) -> None:
            super().__init__(output_root)
            self._running_update_failed = False

        def update_manifest(self, run_id, /, **updates):
            if updates.get("status") == "running" and not self._running_update_failed:
                self._running_update_failed = True
                raise OSError("running lifecycle write failed")
            return super().update_manifest(run_id, **updates)

    recorder = RunningUpdateFailsOnceRecorder(tmp_path / "outputs")
    outcome = run_experiment(
        resolved_spec,
        RunnerDependencies(
            backend=FakeBackend(),
            policy=ZeroPolicyAdapter("zero_policy"),
            recorder=recorder,
            source_path=Path(resolved_spec.source_path),
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ("failed", "unavailable")
    manifest = recorder.read_manifest(outcome.run_id)
    assert (manifest["status"], manifest["result_integrity"]) == ("failed", "unavailable")
