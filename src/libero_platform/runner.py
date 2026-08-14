from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Literal, Mapping, Protocol

import numpy as np
from PIL import Image

from .backends.base import BenchmarkBackend, Episode, Observation, StepResult
from .policies.base import EpisodeContext, PolicyAdapter, PolicyRequest, validate_action
from .recorder import RunRecorder
from .result_schema import DeviceProfile, StepRecord, TrialRecord, percentile
from .result_writer import write_metadata, write_required_artifacts
from .viewer_bridge import PassiveViewerBridge
from .spec import ResolvedExperimentSpec


class Closable(Protocol):
    def close(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class RunnerDependencies:
    backend: BenchmarkBackend
    policy: PolicyAdapter
    recorder: RunRecorder
    source_path: Path
    viewer: Closable | None = None
    git_commit: str | None = None
    event_handler: Callable[[dict[str, object]], None] | None = None


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    run_dir: Path
    status: Literal["completed", "failed", "stopped"]
    result_integrity: Literal["complete", "partial", "unavailable"]
    trials: tuple[TrialRecord, ...]


def run_experiment(spec: ResolvedExperimentSpec, dependencies: RunnerDependencies) -> RunOutcome:
    """Run a previously resolved configuration and persist benchmark evidence."""
    if not isinstance(spec, ResolvedExperimentSpec):
        raise ValueError("run_experiment requires a ResolvedExperimentSpec")

    source_path = Path(spec.source_path)
    if Path(dependencies.source_path) != source_path:
        raise ValueError("source_path must match the resolved spec")

    context = dependencies.recorder.create_run(
        source_path,
        spec,
        git_commit=dependencies.git_commit,
    )
    recorder = dependencies.recorder
    trials: list[TrialRecord] = []
    status: Literal["completed", "failed", "stopped"] = "completed"
    integrity: Literal["complete", "partial", "unavailable"] = "complete"
    failure_type = ""

    lifecycle_ready = False
    try:
        recorder.update_manifest(context.run_id, status="validating", phase="validating")
        recorder.update_manifest(context.run_id, status="running", phase="running")
        lifecycle_ready = True
        _emit(
            dependencies.event_handler,
            {
                "event": "run_started",
                "run_id": context.run_id,
                "run_dir": str(context.run_dir),
                "name": spec.name,
                "suite": spec.benchmark.suite,
                "policy_key": spec.policy.key,
                "episode_total": _episode_total(spec),
            },
        )
        task_names = _task_names(dependencies.backend, spec.benchmark.suite)
        dependencies.policy.load()
        _run_warmups(spec, dependencies, task_names, context.run_id)

        episode_id = 0
        stop_requested = False
        for task_id in spec.benchmark.task_ids:
            task_name = task_names[task_id]
            for initial_state_id in spec.benchmark.initial_state_ids:
                for repetition in range(spec.execution.episodes_per_initial_state):
                    del repetition
                    seed = spec.execution.seed + episode_id
                    recorder.update_manifest(
                        context.run_id,
                        progress={
                            "task_id": task_id,
                            "initial_state_id": initial_state_id,
                            "episode": episode_id + 1,
                            "episode_total": _episode_total(spec),
                            "step": 0,
                            "max_steps": spec.benchmark.max_steps,
                        },
                    )
                    _emit(
                        dependencies.event_handler,
                        {
                            "event": "episode_started",
                            "run_id": context.run_id,
                            "episode": episode_id + 1,
                            "episode_total": _episode_total(spec),
                            "task_id": task_id,
                            "task_name": task_name,
                            "initial_state_id": initial_state_id,
                            "max_steps": spec.benchmark.max_steps,
                        },
                    )
                    trial = _run_formal_episode(
                        spec,
                        dependencies,
                        context.run_id,
                        context.run_dir,
                        episode_id,
                        task_id,
                        task_name,
                        initial_state_id,
                        seed,
                    )
                    trials.append(trial)
                    recorder.append_trial(context.run_id, trial.to_dict())
                    _emit(
                        dependencies.event_handler,
                        {
                            "event": "episode_completed",
                            "run_id": context.run_id,
                            "episode": episode_id + 1,
                            "episode_total": _episode_total(spec),
                            "task_id": task_id,
                            "initial_state_id": initial_state_id,
                            "steps": trial.steps,
                            "success": trial.success,
                        },
                    )
                    episode_id += 1
                    if not trial.success and spec.execution.on_episode_failure == "stop":
                        status = "failed"
                        integrity = "partial"
                        failure_type = trial.failure_type or "episode_failure"
                        stop_requested = True
                        break
                if stop_requested:
                    break
            if stop_requested:
                break
    except KeyboardInterrupt as exc:
        status = "stopped"
        integrity = "partial"
        failure_type = _exception_failure_type(exc)
        _append_run_error(recorder, context.run_id, exc)
    except BaseException as exc:
        status = "failed"
        integrity = "partial" if lifecycle_ready else "unavailable"
        failure_type = _exception_failure_type(exc)
        _append_run_error(recorder, context.run_id, exc)
    finally:
        cleanup_errors = [
            _close_quietly(
            dependencies.policy, "policy", dependencies.event_handler, context.run_id
            ),
            _close_quietly(
            dependencies.viewer, "viewer", dependencies.event_handler, context.run_id
            ),
        ]
        for cleanup_error in cleanup_errors:
            if cleanup_error is None:
                continue
            if isinstance(cleanup_error, KeyboardInterrupt):
                status = "stopped"
                integrity = "partial"
            elif status != "stopped":
                status = "failed"
                integrity = "partial"
            failure_type = type(cleanup_error).__name__
            _append_run_error(recorder, context.run_id, cleanup_error)

    try:
        artifacts = write_required_artifacts(
            run_dir=context.run_dir,
            metadata=_run_metadata(context.run_id, spec, source_path, status, integrity),
            device_profile=_device_profile(spec),
            trials=trials,
            preserve_existing_trials_jsonl=True,
        )
        recorder.update_manifest(
            context.run_id,
            artifacts=sorted(path.name for path in artifacts.values()),
        )
    except KeyboardInterrupt as exc:
        status = "stopped"
        integrity = "partial"
        failure_type = type(exc).__name__
        _append_run_error(recorder, context.run_id, exc)
        _write_terminal_metadata(context.run_dir, context.run_id, spec, source_path, status, integrity)
    except BaseException as exc:
        status = "failed"
        integrity = "unavailable"
        failure_type = type(exc).__name__
        _append_run_error(recorder, context.run_id, exc)
        _write_terminal_metadata(context.run_dir, context.run_id, spec, source_path, status, integrity)

    status, integrity, failure_type = _finalize_run(
        recorder,
        context.run_id,
        context.run_dir,
        spec,
        source_path,
        status,
        integrity,
        failure_type,
    )
    _emit(
        dependencies.event_handler,
        {
            "event": f"run_{status}",
            "run_id": context.run_id,
            "status": status,
            "result_integrity": integrity,
            "failure_type": failure_type or status,
            "log_path": _safe_log_path(recorder, context.run_id),
        },
    )
    return RunOutcome(
        run_id=context.run_id,
        run_dir=context.run_dir,
        status=status,
        result_integrity=integrity,
        trials=tuple(trials),
    )


def _run_warmups(
    spec: ResolvedExperimentSpec,
    dependencies: RunnerDependencies,
    task_names: dict[int, str],
    run_id: str,
) -> None:
    if spec.execution.warmup_episodes == 0:
        return
    task_id = spec.benchmark.task_ids[0]
    initial_state_id = spec.benchmark.initial_state_ids[0]
    task_name = task_names[task_id]
    for warmup_index in range(spec.execution.warmup_episodes):
        seed = spec.execution.seed + warmup_index
        episode = dependencies.backend.open_episode(
            spec.benchmark.suite,
            task_id,
            initial_state_id,
            spec.benchmark.max_steps,
            seed,
        )
        viewer = _episode_viewer(spec, dependencies, run_id, episode)
        try:
            observation = episode.reset()
            viewer.open()
            dependencies.policy.begin_episode(
                EpisodeContext(
                    suite=spec.benchmark.suite,
                    task_id=task_id,
                    task_name=task_name,
                    initial_state_id=initial_state_id,
                    seed=seed,
                )
            )
            previous_action = None
            for step_id in range(1, spec.benchmark.max_steps + 1):
                response = dependencies.policy.predict(
                    _policy_request(
                        "warmup",
                        warmup_index,
                        step_id,
                        observation,
                        previous_action,
                    )
                )
                if response.failure_type or response.error:
                    _append_policy_failure(
                        dependencies.recorder,
                        run_id,
                        warmup_index,
                        step_id,
                        response.failure_type,
                        response.error,
                    )
                    raise RuntimeError(_policy_error_summary(response.failure_type, response.error))
                previous_action = validate_action(response.action)
                result = episode.step(previous_action)
                viewer.sync()
                observation = result.observation
                if result.done:
                    break
        finally:
            viewer.close()
            episode.close()


def _run_formal_episode(
    spec: ResolvedExperimentSpec,
    dependencies: RunnerDependencies,
    run_id: str,
    run_dir: Path,
    episode_id: int,
    task_id: int,
    task_name: str,
    initial_state_id: int,
    seed: int,
) -> TrialRecord:
    episode = dependencies.backend.open_episode(
        spec.benchmark.suite,
        task_id,
        initial_state_id,
        spec.benchmark.max_steps,
        seed,
    )
    policy_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    previous_action = None
    steps = 0
    action_valid = True
    success = False
    termination_reason = "max_steps"
    failure_type = ""
    error_summary = ""
    frame_directory: Path | None = None
    viewer = _episode_viewer(spec, dependencies, run_id, episode)
    try:
        observation = episode.reset()
        reset_evidence = episode.reset_evidence
        viewer.open()
        dependencies.policy.begin_episode(
            EpisodeContext(
                suite=spec.benchmark.suite,
                task_id=task_id,
                task_name=task_name,
                initial_state_id=initial_state_id,
                seed=seed,
            )
        )
        for step_id in range(1, spec.benchmark.max_steps + 1):
            try:
                response = dependencies.policy.predict(
                    _policy_request(run_id, episode_id, step_id, observation, previous_action)
                )
                if response.failure_type or response.error:
                    failure_type = response.failure_type or "policy_error"
                    error_summary = _policy_error_summary(response.failure_type, response.error)
                    termination_reason = "policy_failure"
                    action_valid = False
                    timings = _policy_timings(spec, response.inference_ms, response.metadata)
                    if timings.policy_latency_ms is not None:
                        policy_latencies.append(timings.policy_latency_ms)
                    if timings.end_to_end_ms is not None:
                        end_to_end_latencies.append(timings.end_to_end_ms)
                    _append_policy_failure(
                        dependencies.recorder,
                        run_id,
                        episode_id,
                        step_id,
                        failure_type,
                        error_summary,
                    )
                    _append_step(
                        spec,
                        dependencies.recorder,
                        run_id,
                        StepRecord(
                            run_id=run_id,
                            episode_id=episode_id,
                            step_id=step_id,
                            policy_latency_ms=timings.policy_latency_ms,
                            service_latency_ms=timings.service_latency_ms,
                            transport_latency_ms=timings.transport_latency_ms,
                            end_to_end_ms=timings.end_to_end_ms,
                            action=None,
                            action_valid=False,
                            reward=None,
                            done=True,
                            success=False,
                        ),
                    )
                    _emit_step_completed(
                        dependencies,
                        run_id,
                        episode_id,
                        spec,
                        step_id,
                        done=True,
                        success=False,
                        action_valid=False,
                    )
                    break
                action = validate_action(response.action)
                result = episode.step(action)
                viewer.sync()
            except Exception as exc:
                action_valid = False
                termination_reason = "error"
                failure_type = type(exc).__name__
                error_summary = str(exc)
                typed_failure = getattr(exc, "failure_type", "")
                if isinstance(typed_failure, str) and typed_failure:
                    failure_type = typed_failure
                    termination_reason = "policy_failure"
                    _append_policy_failure(
                        dependencies.recorder,
                        run_id,
                        episode_id,
                        step_id,
                        failure_type,
                        error_summary,
                    )
                _append_step(
                    spec,
                    dependencies.recorder,
                    run_id,
                    StepRecord(
                        run_id=run_id,
                        episode_id=episode_id,
                        step_id=step_id,
                        policy_latency_ms=None,
                        service_latency_ms=None,
                        transport_latency_ms=None,
                        end_to_end_ms=None,
                        action=None,
                        action_valid=False,
                        reward=None,
                        done=True,
                        success=False,
                    ),
                )
                _emit_step_completed(
                    dependencies,
                    run_id,
                    episode_id,
                    spec,
                    step_id,
                    done=True,
                    success=False,
                    action_valid=False,
                )
                break

            steps = step_id
            previous_action = action
            timings = _policy_timings(spec, response.inference_ms, response.metadata)
            if timings.policy_latency_ms is not None:
                policy_latencies.append(timings.policy_latency_ms)
            if timings.end_to_end_ms is not None:
                end_to_end_latencies.append(timings.end_to_end_ms)
            if spec.recording.save_frames and (
                step_id % spec.recording.frame_stride == 0 or result.done
            ):
                frame = episode.render_frame()
                if frame is not None:
                    if frame_directory is None:
                        frame_directory = run_dir / "frames" / f"episode_{episode_id:04d}"
                        frame_directory.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame).save(
                        frame_directory / f"step_{step_id:06d}.png", format="PNG"
                    )
            _append_step(
                spec,
                dependencies.recorder,
                run_id,
                StepRecord(
                    run_id=run_id,
                    episode_id=episode_id,
                    step_id=step_id,
                    policy_latency_ms=timings.policy_latency_ms,
                    service_latency_ms=timings.service_latency_ms,
                    transport_latency_ms=timings.transport_latency_ms,
                    end_to_end_ms=timings.end_to_end_ms,
                    action=action.tolist(),
                    raw_action=_raw_action_trace(response, action),
                    action_transform=response.action_transform,
                    action_clipped=bool(response.action_clipped),
                    action_valid=True,
                    reward=float(result.reward),
                    done=bool(result.done),
                    success=bool(result.success),
                ),
            )
            _emit_step_completed(
                dependencies,
                run_id,
                episode_id,
                spec,
                step_id,
                done=bool(result.done),
                success=bool(result.success),
                action_valid=True,
            )
            observation = result.observation
            success = bool(result.success)
            if result.done:
                termination_reason = "success" if success else "done"
                break
    finally:
        viewer.close()
        episode.close()

    if (
        spec.policy_adapter == "demo_replay"
        and not success
        and not failure_type
        and termination_reason in {"done", "max_steps"}
    ):
        failure_type = "reference_replay_failed"

    return TrialRecord(
        run_id=run_id,
        suite=spec.benchmark.suite,
        task_id=task_id,
        task_name=task_name,
        initial_state_id=initial_state_id,
        episode_id=episode_id,
        seed=seed,
        instruction=observation.instruction if "observation" in locals() else "",
        policy_key=spec.policy.key,
        checkpoint=spec.resolved_checkpoint,
        deployment_mode=spec.deployment.mode,
        device_profile=spec.deployment.profile,
        precision=spec.policy.precision,
        quantization=spec.policy.quantization,
        load_success=True,
        success=success,
        steps=steps,
        termination_reason=termination_reason,
        action_valid=action_valid,
        policy_latency_mean_ms=mean(policy_latencies) if policy_latencies else None,
        policy_latency_p95_ms=percentile(policy_latencies, 95),
        end_to_end_mean_ms=mean(end_to_end_latencies) if end_to_end_latencies else None,
        end_to_end_p95_ms=percentile(end_to_end_latencies, 95),
        peak_host_memory_mb=None,
        peak_device_memory_mb=None,
        oom=False,
        failure_type=failure_type,
        error_summary=error_summary,
        video_path=None,
        frame_directory=(
            frame_directory.relative_to(run_dir).as_posix()
            if frame_directory is not None
            else None
        ),
        reset_seed=None if reset_evidence is None else reset_evidence.seed,
        reset_initial_state_source=(
            "" if reset_evidence is None else reset_evidence.initial_state_source
        ),
        reset_settle_steps=0 if reset_evidence is None else reset_evidence.settle_steps,
        reset_fingerprint="" if reset_evidence is None else reset_evidence.fingerprint,
    )


def _policy_request(
    run_id: str,
    episode_id: int,
    step_id: int,
    observation: Observation,
    previous_action,
) -> PolicyRequest:
    return PolicyRequest(
        run_id=run_id,
        episode_id=episode_id,
        step_id=step_id,
        instruction=observation.instruction,
        images=observation.images,
        proprioception=observation.proprioception,
        previous_action=previous_action,
    )


def _append_step(
    spec: ResolvedExperimentSpec,
    recorder: RunRecorder,
    run_id: str,
    record: StepRecord,
) -> None:
    if spec.recording.save_steps:
        recorder.append_step(run_id, record.to_dict())


@dataclass(frozen=True)
class _PolicyTimings:
    policy_latency_ms: float | None
    service_latency_ms: float | None
    transport_latency_ms: float | None
    end_to_end_ms: float | None


def _policy_timings(
    spec: ResolvedExperimentSpec,
    inference_ms: object,
    metadata: Mapping[str, Any],
) -> _PolicyTimings:
    inference_latency = _finite_number(inference_ms)
    if spec.deployment.mode != "remote":
        return _PolicyTimings(
            policy_latency_ms=inference_latency,
            service_latency_ms=None,
            transport_latency_ms=None,
            end_to_end_ms=inference_latency,
        )

    service_latency = _finite_number(metadata.get("service_latency_ms"))
    round_trip_latency = _finite_number(metadata.get("remote_round_trip_ms"))
    policy_latency = service_latency if service_latency is not None else inference_latency
    transport_latency = (
        max(round_trip_latency - service_latency, 0.0)
        if round_trip_latency is not None and service_latency is not None
        else None
    )
    return _PolicyTimings(
        policy_latency_ms=policy_latency,
        service_latency_ms=service_latency,
        transport_latency_ms=transport_latency,
        end_to_end_ms=round_trip_latency if round_trip_latency is not None else policy_latency,
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _raw_action_trace(response, action) -> list[float]:
    raw_action = response.raw_action
    if raw_action is None:
        return action.tolist()
    return np.asarray(raw_action, dtype=np.float64).tolist()


def _emit_step_completed(
    dependencies: RunnerDependencies,
    run_id: str,
    episode_id: int,
    spec: ResolvedExperimentSpec,
    step_id: int,
    *,
    done: bool,
    success: bool,
    action_valid: bool,
) -> None:
    _emit(
        dependencies.event_handler,
        {
            "event": "step_completed",
            "run_id": run_id,
            "episode": episode_id + 1,
            "episode_total": _episode_total(spec),
            "step": step_id,
            "max_steps": spec.benchmark.max_steps,
            "done": done,
            "success": success,
            "action_valid": action_valid,
        },
    )


def _task_names(backend: BenchmarkBackend, suite: str) -> dict[int, str]:
    names: dict[int, str] = {}
    for task in backend.list_tasks(suite):
        task_id = task.get("task_id")
        task_name = task.get("task_name")
        if type(task_id) is not int or not isinstance(task_name, str) or not task_name:
            raise ValueError("backend task catalog contains an invalid task")
        names[task_id] = task_name
    return names


def _episode_total(spec: ResolvedExperimentSpec) -> int:
    return (
        len(spec.benchmark.task_ids)
        * len(spec.benchmark.initial_state_ids)
        * spec.execution.episodes_per_initial_state
    )


def _device_profile(spec: ResolvedExperimentSpec) -> DeviceProfile:
    metadata = spec.device_metadata
    return DeviceProfile(
        device_model=str(metadata.get("device_model") or "unavailable"),
        host_os=_optional_string(metadata.get("host_os")),
        cpu_model=_optional_string(metadata.get("cpu_model")),
        cpu_count=_optional_int(metadata.get("cpu_count")),
        host_memory_total_mb=_optional_float(metadata.get("host_memory_total_mb")),
        device_name=_optional_string(metadata.get("device_name")),
        device_memory_total_mb=_optional_float(metadata.get("device_memory_total_mb")),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if type(value) is int else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _append_run_error(recorder: RunRecorder, run_id: str, exc: BaseException) -> None:
    failure_type = _exception_failure_type(exc)
    try:
        recorder.append_event(
            run_id,
            "run_error",
            failure_type=failure_type,
            error_summary=str(exc),
        )
    except BaseException:
        pass


    try:
        with recorder.log_path(run_id).open("a", encoding="utf-8", newline="") as handle:
            handle.write(f"{type(exc).__name__}: {exc}\n")
    except BaseException:
        pass


def _exception_failure_type(exc: BaseException) -> str:
    typed_failure = getattr(exc, 'failure_type', '')
    if isinstance(typed_failure, str) and typed_failure:
        return typed_failure
    return type(exc).__name__


def _append_policy_failure(
    recorder: RunRecorder,
    run_id: str,
    episode_id: int,
    step_id: int,
    failure_type: str,
    error_summary: str,
) -> None:
    try:
        recorder.append_event(
            run_id,
            "policy_failure",
            episode_id=episode_id,
            step_id=step_id,
            failure_type=failure_type or "policy_error",
            error_summary=error_summary,
        )
    except BaseException:
        return


def _policy_error_summary(failure_type: str, error: str) -> str:
    return error or failure_type or "policy response reported a failure"


def _run_metadata(
    run_id: str,
    spec: ResolvedExperimentSpec,
    source_path: Path,
    status: str,
    integrity: str,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "status": status,
        "result_integrity": integrity,
        "experiment_name": spec.name,
        "source_path": str(source_path),
    }


def _write_terminal_metadata(
    run_dir: Path,
    run_id: str,
    spec: ResolvedExperimentSpec,
    source_path: Path,
    status: str,
    integrity: str,
) -> None:
    try:
        write_metadata(
            run_dir / "metadata.json",
            _run_metadata(run_id, spec, source_path, status, integrity),
        )
    except BaseException:
        return


def _close_quietly(
    resource: Closable | None,
    resource_name: str,
    event_handler: Callable[[dict[str, object]], None] | None,
    run_id: str,
) -> BaseException | None:
    if resource is None:
        return None
    try:
        resource.close()
    except BaseException as exc:
        _emit(
            event_handler,
            {
                "event": "warning",
                "run_id": run_id,
                "message": f"could not close {resource_name}",
                "failure_type": type(exc).__name__,
            },
        )
        return exc
    return None


def _emit(
    event_handler: Callable[[dict[str, object]], None] | None,
    event: dict[str, object],
) -> None:
    if event_handler is None:
        return
    try:
        event_handler(dict(event))
    except BaseException:
        return


def _finalize_run(
    recorder: RunRecorder,
    run_id: str,
    run_dir: Path,
    spec: ResolvedExperimentSpec,
    source_path: Path,
    status: Literal["completed", "failed", "stopped"],
    integrity: Literal["complete", "partial", "unavailable"],
    failure_type: str,
) -> tuple[
    Literal["completed", "failed", "stopped"],
    Literal["complete", "partial", "unavailable"],
    str,
]:
    _prepare_terminal_finalization(recorder, run_id)
    try:
        recorder.finalize(
            run_id,
            status=status,
            result_integrity=integrity,
            error=None if status == "completed" else {"failure_type": failure_type or status},
        )
        return status, integrity, failure_type
    except BaseException as exc:
        failure_type = type(exc).__name__
        status = "failed"
        integrity = "unavailable"
        _append_run_error(recorder, run_id, exc)
        _write_terminal_metadata(
            run_dir,
            run_id,
            spec,
            source_path,
            status,
            integrity,
        )

    terminal_state = _terminal_manifest_state(recorder, run_id)
    if terminal_state is not None:
        return terminal_state[0], terminal_state[1], failure_type

    _prepare_terminal_finalization(recorder, run_id)
    try:
        recorder.finalize(
            run_id,
            status=status,
            result_integrity=integrity,
            error={"failure_type": failure_type},
        )
    except BaseException as retry_error:
        _append_run_error(recorder, run_id, retry_error)
        try:
            recorder.update_manifest(
                run_id,
                status=status,
                result_integrity=integrity,
                phase="terminal",
                error={"failure_type": failure_type},
            )
        except BaseException:
            pass
    return status, integrity, failure_type


def _prepare_terminal_finalization(recorder: RunRecorder, run_id: str) -> None:
    try:
        manifest = recorder.read_manifest(run_id)
        if manifest.get("status") == "created":
            manifest = recorder.update_manifest(
                run_id, status="validating", phase="validating"
            )
        if manifest.get("status") == "validating":
            recorder.update_manifest(run_id, status="running", phase="running")
    except BaseException:
        return


def _safe_log_path(recorder: RunRecorder, run_id: str) -> str:
    try:
        return str(recorder.log_path(run_id))
    except BaseException:
        return "unavailable"


def _terminal_manifest_state(
    recorder: RunRecorder, run_id: str
) -> tuple[
    Literal["completed", "failed", "stopped"],
    Literal["complete", "partial", "unavailable"],
] | None:
    try:
        manifest = recorder.read_manifest(run_id)
    except BaseException:
        return None
    status = manifest.get("status")
    integrity = manifest.get("result_integrity")
    if status in {"completed", "failed", "stopped"} and integrity in {
        "complete",
        "partial",
        "unavailable",
    }:
        return status, integrity
    return None


def _episode_viewer(
    spec: ResolvedExperimentSpec,
    dependencies: RunnerDependencies,
    run_id: str,
    episode: Episode,
) -> PassiveViewerBridge:
    return PassiveViewerBridge(
        episode.model,
        episode.data,
        spec.viewer.enabled,
        on_warning=lambda message: _record_viewer_warning(dependencies, run_id, message),
    )


def _record_viewer_warning(
    dependencies: RunnerDependencies, run_id: str, message: str
) -> None:
    try:
        dependencies.recorder.append_event(run_id, "viewer_warning", message=message)
    except Exception:
        pass
    _emit(
        dependencies.event_handler,
        {"event": "viewer_warning", "run_id": run_id, "message": message},
    )
