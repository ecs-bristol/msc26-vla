from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from libero_platform.recorder import RunRecorder
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
    path.write_bytes(b"schema_version: 1\nname: exact-source\n")
    return path


@pytest.fixture
def resolved_spec(source_path: Path) -> ResolvedExperimentSpec:
    return ResolvedExperimentSpec.model_validate(
        {
            **VALID,
            "source_path": str(source_path),
            "dataset_directory": "datasets/libero",
            "resolved_checkpoint": "none",
            "policy_adapter": "zero",
        }
    )


@pytest.fixture
def run_context(
    recorder: RunRecorder, source_path: Path, resolved_spec: ResolvedExperimentSpec
):
    return recorder.create_run(source_path, resolved_spec, git_commit="abc123")


def test_create_run_freezes_source_and_resolved_yaml(
    recorder: RunRecorder, source_path: Path, resolved_spec: ResolvedExperimentSpec
) -> None:
    source_bytes = source_path.read_bytes()
    context = recorder.create_run(source_path, resolved_spec, git_commit="abc123")
    source_path.write_bytes(b"schema_version: 1\nname: later-change\n")
    resolved_spec.resolved_checkpoint = "later-checkpoint"

    assert (context.run_dir / "source_spec.yaml").read_bytes() == source_bytes
    assert "resolved_checkpoint:" in (
        context.run_dir / "resolved_spec.yaml"
    ).read_text(encoding="utf-8")
    assert "later-checkpoint" not in (
        context.run_dir / "resolved_spec.yaml"
    ).read_text(encoding="utf-8")
    manifest = recorder.read_manifest(context.run_id)
    assert manifest["status"] == "created"
    assert manifest["result_integrity"] == "pending"
    assert manifest["source_spec_path"] == "source_spec.yaml"
    assert manifest["resolved_spec_path"] == "resolved_spec.yaml"
    assert manifest["git_commit"] == "abc123"
    assert manifest["progress"] == {
        "task_id": None,
        "initial_state_id": None,
        "episode": 0,
        "episode_total": 0,
        "step": 0,
        "max_steps": 0,
    }
    assert manifest["viewer"] == {"enabled": False, "status": "closed"}
    assert all(
        (context.run_dir / name).is_file()
        for name in ("events.jsonl", "steps.jsonl", "trials.jsonl", "run.log")
    )


def test_run_ids_are_unique_and_confined_to_output_root(
    recorder: RunRecorder, source_path: Path, resolved_spec: ResolvedExperimentSpec
) -> None:
    first = recorder.create_run(source_path, resolved_spec, git_commit="abc123")
    second = recorder.create_run(source_path, resolved_spec, git_commit="abc123")

    assert first.run_id != second.run_id
    assert first.run_dir.parent == recorder.output_root
    assert second.run_dir.parent == recorder.output_root
    assert first.run_id.startswith("run_")


def test_update_manifest_replaces_atomically_for_valid_lifecycle_transition(
    recorder: RunRecorder, run_context
) -> None:
    recorder.update_manifest(run_context.run_id, status="validating")
    recorder.update_manifest(run_context.run_id, status="running")

    with patch("libero_platform.recorder.os.replace", wraps=os.replace) as replace:
        updated = recorder.update_manifest(
            run_context.run_id,
            status="failed",
            result_integrity="partial",
            phase="terminal",
            error={"failure_type": "oom"},
        )

    assert replace.called
    assert (updated["status"], updated["result_integrity"]) == ("failed", "partial")
    assert updated["phase"] == "terminal"
    assert updated["error"] == {"failure_type": "oom"}
    assert list(run_context.run_dir.glob(".*.tmp")) == []


def test_update_manifest_rejects_invalid_lifecycle_transition_atomically(
    recorder: RunRecorder, run_context
) -> None:
    manifest_path = run_context.run_dir / "manifest.json"
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        recorder.update_manifest(run_context.run_id, status="running")

    assert manifest_path.read_bytes() == before


def test_lifecycle_transitions_through_validating_and_running(
    recorder: RunRecorder, run_context
) -> None:
    validating = recorder.update_manifest(run_context.run_id, status="validating")
    running = recorder.update_manifest(run_context.run_id, status="running")

    assert (validating["status"], validating["result_integrity"]) == (
        "validating",
        "pending",
    )
    assert (running["status"], running["result_integrity"]) == ("running", "pending")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("run_id", "other_run"),
        ("source_spec_path", "elsewhere/source.yaml"),
        ("resolved_spec_path", "elsewhere/resolved.yaml"),
        ("git_commit", "def456"),
        ("timestamps", {"created_at": "not-the-original-time"}),
    ],
)
def test_update_manifest_rejects_immutable_identity_overrides(
    recorder: RunRecorder, run_context, field: str, value: object
) -> None:
    before = recorder.read_manifest(run_context.run_id)

    with pytest.raises(ValueError, match="immutable"):
        recorder.update_manifest(run_context.run_id, **{field: value})

    assert recorder.read_manifest(run_context.run_id) == before


def test_append_event_is_append_only_and_forces_run_identity(
    recorder: RunRecorder, run_context
) -> None:
    recorder.append_event(
        run_context.run_id,
        {"event": "phase_started", "run_id": "other_run", "timestamp": "caller-time"},
    )
    recorder.append_event(run_context.run_id, "phase_finished", phase="runner")

    events = [
        json.loads(line)
        for line in (run_context.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == ["phase_started", "phase_finished"]
    assert events[0]["run_id"] == run_context.run_id
    assert events[0]["timestamp"] == "caller-time"
    assert events[-1]["phase"] == "runner"


def test_append_step_and_trial_are_jsonl(recorder: RunRecorder, run_context) -> None:
    recorder.append_step(run_context.run_id, {"step_id": 1})
    recorder.append_trial(run_context.run_id, {"episode_id": 0, "success": True})

    assert '"step_id":1' in (run_context.run_dir / "steps.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"success":true' in (run_context.run_dir / "trials.jsonl").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("run_id", ["", ".", "..", "../escape", "bad/id", "bad id"])
def test_rejects_unsafe_run_ids(
    recorder: RunRecorder,
    source_path: Path,
    resolved_spec: ResolvedExperimentSpec,
    run_id: str,
) -> None:
    with pytest.raises(ValueError):
        recorder.create_run(source_path, resolved_spec, run_id=run_id)


def test_read_manifest_rejects_malformed_json(recorder: RunRecorder) -> None:
    run_dir = recorder.output_root / "run_bad"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed"):
        recorder.read_manifest("run_bad")


def test_rejects_symlinked_run_files(
    recorder: RunRecorder, run_context, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    manifest_path = run_context.run_dir / "manifest.json"
    manifest_path.unlink()
    try:
        manifest_path.symlink_to(outside)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip(f"Windows symlink privilege is unavailable: {exc}")
        raise

    with pytest.raises(ValueError, match="escapes"):
        recorder.read_manifest(run_context.run_id)


def test_rejects_symlinked_output_root_without_platform_symlink_privilege(
    tmp_path: Path, source_path: Path, resolved_spec: ResolvedExperimentSpec
) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    recorder = RunRecorder(output_root)

    with patch(
        "libero_platform.recorder.Path.is_symlink",
        autospec=True,
        side_effect=lambda path: path == output_root,
    ):
        with pytest.raises(ValueError, match="output_root must not be a symlink"):
            recorder.create_run(source_path, resolved_spec)


def test_rejects_output_root_with_redirecting_ancestor_without_platform_symlink_privilege(
    tmp_path: Path
) -> None:
    ancestor = tmp_path / "redirected"
    ancestor.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real_resolve = Path.resolve

    def redirected_resolve(path: Path, strict: bool = False) -> Path:
        if path == ancestor:
            return outside
        return real_resolve(path, strict=strict)

    with patch(
        "libero_platform.recorder.Path.resolve",
        autospec=True,
        side_effect=redirected_resolve,
    ):
        with pytest.raises(ValueError, match="output_root must not traverse"):
            RunRecorder(ancestor / "outputs")


def test_rejects_symlinked_run_directory_without_platform_symlink_privilege(
    recorder: RunRecorder, run_context
) -> None:
    with patch(
        "libero_platform.recorder.Path.is_symlink",
        autospec=True,
        side_effect=lambda path: path == run_context.run_dir,
    ):
        with pytest.raises(ValueError, match="run path escapes"):
            recorder.read_manifest(run_context.run_id)


def test_rejects_symlinked_final_file_without_platform_symlink_privilege(
    recorder: RunRecorder, run_context
) -> None:
    manifest_path = run_context.run_dir / "manifest.json"
    with patch(
        "libero_platform.recorder.Path.is_symlink",
        autospec=True,
        side_effect=lambda path: path == manifest_path,
    ):
        with pytest.raises(ValueError, match="escapes"):
            recorder.read_manifest(run_context.run_id)


def test_read_log_tail_is_bounded(recorder: RunRecorder, run_context) -> None:
    log_path = recorder.log_path(run_context.run_id)
    log_path.write_text("\n".join(f"line-{index}" for index in range(20)) + "\n", encoding="utf-8")

    assert recorder.read_log_tail(run_context.run_id, max_lines=3, max_chars=80) == [
        "line-17",
        "line-18",
        "line-19",
    ]


@pytest.mark.parametrize(
    ("status", "result_integrity"),
    [("completed", "complete"), ("failed", "partial"), ("stopped", "partial")],
)
def test_finalize_only_accepts_planned_terminal_statuses(
    recorder: RunRecorder, run_context, status: str, result_integrity: str
) -> None:
    recorder.update_manifest(run_context.run_id, status="validating")
    recorder.update_manifest(run_context.run_id, status="running")
    manifest = recorder.finalize(
        run_context.run_id, status=status, result_integrity=result_integrity
    )

    assert (manifest["status"], manifest["result_integrity"]) == (
        status,
        result_integrity,
    )
    assert manifest["phase"] == "terminal"
    assert "finished_at" in manifest["timestamps"]


def test_finalize_rejects_invalid_terminal_status_combination(
    recorder: RunRecorder, run_context
) -> None:
    recorder.update_manifest(run_context.run_id, status="validating")
    recorder.update_manifest(run_context.run_id, status="running")
    with pytest.raises(ValueError, match="invalid terminal"):
        recorder.finalize(
            run_context.run_id, status="completed", result_integrity="partial"
        )


@pytest.mark.parametrize("status", ["completed", "failed", "stopped"])
def test_finalize_accepts_unavailable_integrity_for_terminal_runs(
    recorder: RunRecorder, run_context, status: str
) -> None:
    recorder.update_manifest(run_context.run_id, status="validating")
    recorder.update_manifest(run_context.run_id, status="running")

    manifest = recorder.finalize(
        run_context.run_id, status=status, result_integrity="unavailable"
    )

    assert (manifest["status"], manifest["result_integrity"]) == (status, "unavailable")


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "running"},
        {"result_integrity": "unavailable"},
        {"status": "completed", "result_integrity": "complete"},
    ],
)
def test_terminal_lifecycle_cannot_be_reopened_or_repaired(
    recorder: RunRecorder, run_context, updates: dict[str, str]
) -> None:
    recorder.update_manifest(run_context.run_id, status="validating")
    recorder.update_manifest(run_context.run_id, status="running")
    recorder.finalize(run_context.run_id, status="failed", result_integrity="partial")
    manifest_path = run_context.run_dir / "manifest.json"
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        recorder.update_manifest(run_context.run_id, **updates)

    assert manifest_path.read_bytes() == before
