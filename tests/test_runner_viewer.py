from __future__ import annotations

from pathlib import Path

from libero_platform.backends.fake_backend import FakeBackend
from libero_platform.policies.zero_policy import ZeroPolicyAdapter
from libero_platform.recorder import RunRecorder
from libero_platform.runner import RunnerDependencies, _run_formal_episode
from libero_platform.spec import ResolvedExperimentSpec


def test_runner_creates_and_closes_a_viewer_for_each_episode(
    tmp_path: Path, monkeypatch
) -> None:
    from libero_platform import runner

    calls: list[tuple[str, object]] = []

    class RecordingBridge:
        def __init__(self, model, data, enabled, *, on_warning) -> None:
            calls.append(("created", (model, data, enabled)))
            self._on_warning = on_warning

        def open(self) -> None:
            calls.append(("open", None))

        def sync(self) -> None:
            calls.append(("sync", None))

        def close(self) -> None:
            calls.append(("close", None))

    monkeypatch.setattr(runner, "PassiveViewerBridge", RecordingBridge)
    source_path = tmp_path / "experiment.yaml"
    source_path.write_text("schema_version: 1\n", encoding="utf-8")
    spec = ResolvedExperimentSpec.model_validate(
        {
            "schema_version": 1,
            "name": "fake_smoke",
            "benchmark": {
                "backend": "fake",
                "suite": "libero_spatial",
                "task_ids": [0],
                "initial_state_ids": [0],
                "max_steps": 1,
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
            "viewer": {"enabled": True},
            "recording": {
                "save_frames": False,
                "save_video": False,
                "frame_stride": 20,
                "save_steps": True,
            },
            "source_path": str(source_path),
            "dataset_directory": "datasets/libero",
            "resolved_checkpoint": "none",
            "policy_adapter": "zero",
        }
    )
    recorder = RunRecorder(tmp_path / "outputs")
    run_context = recorder.create_run(source_path, spec)
    run_id = run_context.run_id
    dependencies = RunnerDependencies(
        backend=FakeBackend(success_step=1),
        policy=ZeroPolicyAdapter("zero_policy"),
        recorder=recorder,
        source_path=source_path,
    )

    for episode_id in range(2):
        _run_formal_episode(
            spec,
            dependencies,
            run_id,
            run_context.run_dir,
            episode_id,
            0,
            "fake_task_0",
            0,
            42 + episode_id,
        )

    assert [call[0] for call in calls] == [
        "created",
        "open",
        "sync",
        "close",
        "created",
        "open",
        "sync",
        "close",
    ]
