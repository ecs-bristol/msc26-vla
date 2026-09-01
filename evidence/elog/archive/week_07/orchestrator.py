from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from .executors import CommandPlan, ConsolePaths, build_command_plan, discover_artifacts
from .recorder import RunRecorder
from .results import inspect_result, normalize_batch
from .schema import expand_model_keys, validate_experiment_spec
from .viewer import ViewerController


PopenFactory = Callable[..., subprocess.Popen]


class ExperimentOrchestrator:
    """Coordinate whitelisted executor processes with persistent run manifests."""

    def __init__(
        self,
        recorder: RunRecorder,
        catalog: dict,
        paths: ConsolePaths,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        self.recorder = recorder
        self.catalog = deepcopy(catalog)
        self.paths = paths
        self.popen_factory = popen_factory
        self.viewer_controller = ViewerController(recorder)
        self._processes: dict[str, subprocess.Popen] = {}
        self._plans: dict[str, CommandPlan] = {}
        self._logs: dict[str, Any] = {}
        self._batches: dict[str, dict[str, Any]] = {}

    def validate(self, payload: dict) -> dict:
        return validate_experiment_spec(payload, self.catalog)

    def preflight(self, validated_spec: dict) -> dict:
        if not isinstance(validated_spec, dict):
            raise ValueError("validated_spec must be an object")

        selection = validated_spec.get("model_selection") or {}
        model_keys = expand_model_keys(validated_spec)
        if selection.get("mode") == "batch":
            plans = []
            for model_key in model_keys:
                child_spec = deepcopy(validated_spec)
                child_spec["model_selection"] = {**selection, "mode": "single", "model_keys": [model_key]}
                plans.append(build_command_plan(child_spec, self.catalog, self.paths, preflight=True))
            executor = "batch"
            viewer_available = bool(plans) and all(plan.viewer_available for plan in plans)
            warnings = ["Each selected model runs separately."]
        else:
            plan = build_command_plan(validated_spec, self.catalog, self.paths, preflight=True)
            executor = plan.executor_key
            viewer_available = plan.viewer_available
            warnings = []

        return {
            "executor": executor,
            "output_root": str(self.recorder.state_root / "runs"),
            "warnings": warnings,
            "model_count": len(model_keys),
            "trial_count": int((validated_spec.get("execution") or {}).get("trials", 0)),
            "viewer_available": viewer_available,
        }

    def start(self, validated_spec: dict) -> dict:
        if not isinstance(validated_spec, dict):
            raise ValueError("validated_spec must be an object")

        initial_plan = build_command_plan(validated_spec, self.catalog, self.paths)
        manifest = self.recorder.create_run(
            spec=validated_spec,
            executor_key=initial_plan.executor_key,
            viewer_available=initial_plan.viewer_available,
        )
        return self._launch_run(manifest["run_id"], validated_spec)

    def start_batch(self, validated_spec: dict) -> dict:
        if not isinstance(validated_spec, dict):
            raise ValueError("validated_spec must be an object")
        selection = validated_spec.get("model_selection") or {}
        if validated_spec.get("mode") != "evaluation" or selection.get("mode") != "batch":
            raise ValueError("batch orchestration requires evaluation batch mode")
        model_keys = expand_model_keys(validated_spec)
        if len(model_keys) < 2:
            raise ValueError("batch orchestration requires at least two models")

        parent_spec = deepcopy(validated_spec)
        shared_snapshot = {
            "environment": deepcopy(parent_spec.get("environment")),
            "task": deepcopy(parent_spec.get("task")),
        }
        shared_spec_hash = _stable_hash(shared_snapshot)
        parent = self.recorder.create_run(spec=parent_spec, executor_key="batch")
        batch_id = parent["run_id"]
        self.recorder.update_manifest(
            batch_id,
            shared_spec_hash=shared_spec_hash,
            child_run_ids=[],
            progress={"completed": 0, "total": len(model_keys)},
        )

        child_specs: list[dict] = []
        preflight_plans: list[CommandPlan | None] = []
        planning_errors: list[Exception | None] = []
        for model_key in model_keys:
            child_spec = deepcopy(parent_spec)
            child_selection = deepcopy(child_spec["model_selection"])
            child_selection["mode"] = "single"
            child_selection["model_keys"] = [model_key]
            child_spec["model_selection"] = child_selection
            child_specs.append(child_spec)
            try:
                plan = build_command_plan(child_spec, self.catalog, self.paths, preflight=True)
            except Exception as exc:
                plan = None
                planning_errors.append(exc)
            else:
                planning_errors.append(None)
            preflight_plans.append(plan)

        shared_planning_error = _shared_preflight_planning_error(planning_errors)
        if shared_planning_error is not None:
            return self._fail_batch_preflight(batch_id, shared_planning_error)

        valid_preflight_plans = [
            (index, plan)
            for index, plan in enumerate(preflight_plans)
            if plan is not None
        ]
        preflight_errors: dict[int, Exception] = {}
        if valid_preflight_plans:
            try:
                self.recorder.update_manifest(
                    batch_id,
                    phase="preflight",
                    log_path=str(self.recorder.log_path(batch_id)),
                )
                preflight_errors = self._run_batch_preflight(batch_id, valid_preflight_plans)
            except Exception as exc:
                return self._fail_batch_preflight(batch_id, exc)

        child_run_ids: list[str] = []
        try:
            for index, (child_spec, plan, planning_error) in enumerate(zip(child_specs, preflight_plans, planning_errors)):
                child = self.recorder.create_run(
                    spec=child_spec,
                    executor_key=plan.executor_key if plan is not None else "unavailable",
                    parent_batch_id=batch_id,
                    viewer_available=plan.viewer_available if plan is not None else False,
                )
                child_run_ids.append(child["run_id"])
                model_preflight_error = planning_error or preflight_errors.get(index)
                if model_preflight_error is not None:
                    self._fail_model_preflight(child["run_id"], model_preflight_error)
            self.recorder.update_manifest(
                batch_id,
                status="running",
                phase="scheduler",
                shared_spec_hash=shared_spec_hash,
                child_run_ids=child_run_ids,
                timestamps={"started_at": _timestamp()},
            )
            self._batches[batch_id] = {
                "child_specs": dict(zip(child_run_ids, child_specs)),
                "shared_spec_hash": shared_spec_hash,
            }
            self.recorder.append_event(batch_id, "batch_started", child_run_ids=child_run_ids)
            self._start_next_child(batch_id, raise_on_failure=True)
            return self._refresh_batch(batch_id)
        except Exception as exc:
            return self._fail_batch_setup(batch_id, child_run_ids, exc)

    def _start_next_child(self, batch_id: str, *, raise_on_failure: bool = False) -> dict:
        batch = self._batches.get(batch_id)
        if batch is None:
            raise ValueError(f"batch is not registered with this orchestrator: {batch_id}")
        parent = self.get_run(batch_id)
        if parent.get("status") in _TERMINAL_STATUSES:
            return parent

        for child_id in parent.get("child_run_ids", []):
            child = self.get_run(child_id)
            if child.get("status") != "queued":
                continue
            child_spec = batch["child_specs"][child_id]
            launched = self._launch_run(child_id, child_spec, raise_on_failure=raise_on_failure)
            if launched.get("status") in _TERMINAL_STATUSES:
                continue
            return launched
        return self.get_run(batch_id)

    def _refresh_batch(self, batch_id: str) -> dict:
        parent = self.get_run(batch_id)
        if parent.get("status") in _TERMINAL_STATUSES:
            return parent

        child_ids = list(parent.get("child_run_ids", []))
        children = [self.get_run(child_id) for child_id in child_ids]
        if not any(child.get("status") == "running" for child in children):
            if any(child.get("status") == "queued" for child in children):
                try:
                    self._start_next_child(batch_id)
                except Exception as exc:
                    return self._fail_batch_setup(batch_id, child_ids, exc)
                children = [self.get_run(child_id) for child_id in child_ids]

        terminal_children = [child for child in children if child.get("status") in _TERMINAL_STATUSES]
        if len(terminal_children) != len(children):
            return self.recorder.update_manifest(
                batch_id,
                status="running",
                phase="scheduler",
                result_integrity="pending",
                progress={"completed": len(terminal_children), "total": len(children)},
            )

        normalized = normalize_batch(parent, children)
        if normalized["result_integrity"] == "unavailable":
            updated = self.recorder.update_manifest(
                batch_id,
                status="failed",
                phase="aggregate",
                result_integrity="unavailable",
                error={
                    "failure_type": "aggregate_unavailable",
                    "message": "batch has no successful child results",
                },
                progress={"completed": len(children), "total": len(children)},
                timestamps={"finished_at": _timestamp()},
            )
            updated = self._persist_terminal_result(updated, children)
            return self._mirror_viewer_status(batch_id, updated)
        updated = self.recorder.update_manifest(
            batch_id,
            status="completed",
            phase="aggregate",
            progress={"completed": len(children), "total": len(children)},
            timestamps={"finished_at": _timestamp()},
        )
        updated = self._persist_terminal_result(updated, children)
        return self._mirror_viewer_status(batch_id, updated)

    def _run_batch_preflight(self, batch_id: str, plans: list[tuple[int, CommandPlan]]) -> dict[int, Exception]:
        log_path = self.recorder.log_path(batch_id)
        results: dict[int, Exception] = {}
        try:
            for index, plan in plans:
                with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                    process = self.popen_factory(
                        list(plan.cmd),
                        cwd=plan.cwd,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    return_code = process.wait()
                if return_code != 0:
                    if plan.preflight_failure_scope == "shared":
                        raise RuntimeError(
                            f"shared preflight exited with return code {return_code}"
                        )
                    results[index] = RuntimeError(
                        f"model preflight exited with return code {return_code}"
                    )
        except Exception as exc:
            raise RuntimeError(f"shared preflight could not execute: {exc}") from exc
        return results

    def _fail_model_preflight(self, run_id: str, exc: Exception) -> dict:
        updated = self.recorder.update_manifest(
            run_id,
            status="failed",
            phase="preflight",
            result_integrity="unavailable",
            error={"failure_type": "model_preflight", "message": str(exc)},
            timestamps={"finished_at": _timestamp()},
        )
        updated = self._persist_terminal_result(updated)
        self.recorder.append_event(run_id, "run_failed", failure_type="model_preflight")
        return self._mirror_viewer_status(run_id, updated)

    def _fail_batch_preflight(self, batch_id: str, exc: Exception) -> dict:
        return self._fail_batch(batch_id, "preflight", "shared_preflight", exc)

    def _fail_batch_setup(self, batch_id: str, child_run_ids: list[str], exc: Exception) -> dict:
        for child_id in child_run_ids:
            try:
                child = self.get_run(child_id)
                status = child.get("status")
                failure_type = (child.get("error") or {}).get("failure_type")
                if child_id in self._processes:
                    try:
                        self._stop_process(child_id, child)
                    except ValueError:
                        pass
                    finally:
                        self._release_process(child_id)
                    self._mark_batch_setup_child(child_id, exc)
                elif status == "running":
                    self._mark_batch_setup_child(child_id, exc)
                elif status == "queued" or (status == "failed" and failure_type == "launch"):
                    self._mark_batch_setup_child(child_id, exc)
            except Exception:
                continue
        return self._fail_batch(batch_id, "scheduler", "batch_setup", exc, child_run_ids=child_run_ids)

    def _mark_batch_setup_child(self, run_id: str, exc: Exception) -> dict:
        updated = self.recorder.update_manifest(
            run_id,
            status="failed",
            phase="scheduler",
            result_integrity="unavailable",
            error={"failure_type": "batch_setup", "message": str(exc)},
            timestamps={"finished_at": _timestamp()},
        )
        updated = self._persist_terminal_result(updated)
        self.recorder.append_event(run_id, "run_failed", failure_type="batch_setup")
        return self._mirror_viewer_status(run_id, updated)

    def _fail_batch(
        self,
        batch_id: str,
        phase: str,
        failure_type: str,
        exc: Exception,
        *,
        child_run_ids: list[str] | None = None,
    ) -> dict:
        updates = {
            "status": "failed",
            "phase": phase,
            "result_integrity": "unavailable",
            "error": {"failure_type": failure_type, "message": str(exc)},
            "timestamps": {"finished_at": _timestamp()},
        }
        if child_run_ids is not None:
            updates["child_run_ids"] = child_run_ids
        updated = self.recorder.update_manifest(batch_id, **updates)
        children = [
            self.get_run(run_id)
            for run_id in child_run_ids or []
            if _manifest_exists(self.recorder, run_id)
        ]
        updated = self._persist_terminal_result(updated, children)
        self.recorder.append_event(batch_id, "batch_failed", failure_type=failure_type)
        return self._mirror_viewer_status(batch_id, updated)

    def _launch_run(
        self,
        run_id: str,
        validated_spec: dict,
        *,
        raise_on_failure: bool = True,
    ) -> dict:
        plan: CommandPlan | None = None
        log_path: Path | None = None
        log_handle: Any = None
        try:
            plan = build_command_plan(validated_spec, self.catalog, self.paths, run_id=run_id)
            log_path = self.recorder.log_path(run_id)
            log_handle = log_path.open("w", encoding="utf-8", buffering=1)
            process = self.popen_factory(
                list(plan.cmd),
                cwd=plan.cwd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            if log_handle is not None:
                try:
                    log_handle.close()
                except BaseException:
                    pass
            failure_updates = {
                "status": "failed",
                "phase": "executor",
                "result_integrity": "unavailable",
                "error": {
                    "failure_type": "launch",
                    "message": (
                        "executor log could not be opened"
                        if log_handle is None
                        else "executor process could not be started"
                    ),
                },
                "timestamps": {"finished_at": _timestamp()},
            }
            if log_path is not None:
                failure_updates["log_path"] = str(log_path)
            try:
                updated = self.recorder.update_manifest(run_id, **failure_updates)
                updated = self._persist_terminal_result(updated)
                self._mirror_viewer_status(run_id, updated)
            except BaseException:
                pass
            try:
                self.recorder.append_event(run_id, "run_failed", failure_type="launch")
            except BaseException:
                pass
            if raise_on_failure:
                raise
            return self.recorder.read_manifest(run_id)

        assert plan is not None
        self._processes[run_id] = process
        self._plans[run_id] = plan
        self._logs[run_id] = log_handle
        try:
            started = self.recorder.update_manifest(
                run_id,
                status="running",
                phase="executor",
                result_integrity="unavailable",
                timestamps={"started_at": _timestamp()},
                command={
                    "executor_key": plan.executor_key,
                    "argv": list(plan.cmd),
                    "cwd": str(plan.cwd),
                },
                process_id=process.pid,
                log_path=str(log_path),
            )
        except BaseException:
            self._terminate_and_release_process(run_id, process)
            raise
        self.recorder.append_event(run_id, "run_started", process_id=process.pid)
        return started

    def poll(self, run_id: str) -> dict:
        manifest = self.get_run(run_id)
        if manifest.get("executor_key") == "batch":
            return self._poll_batch(run_id, manifest)
        return self._poll_process(run_id, manifest)

    def _poll_batch(self, batch_id: str, manifest: dict) -> dict:
        if manifest.get("status") in _TERMINAL_STATUSES:
            return manifest
        if batch_id not in self._batches:
            raise ValueError(f"batch is not registered with this orchestrator: {batch_id}")

        for child_id in manifest.get("child_run_ids", []):
            child = self.get_run(child_id)
            if child.get("status") == "running" and child_id in self._processes:
                self._poll_process(child_id, child)
        return self._refresh_batch(batch_id)

    def _poll_process(self, run_id: str, manifest: dict) -> dict:
        process = self._processes.get(run_id)
        if process is None:
            if manifest.get("status") in _TERMINAL_STATUSES:
                return manifest
            raise ValueError(f"run is not registered with this orchestrator: {run_id}")

        return_code = process.poll()
        if return_code is None:
            return self._mirror_viewer_status(run_id, manifest)

        updated = self._finalize_natural_process(run_id, return_code)
        updated = self._mirror_viewer_status(run_id, updated)
        if updated.get("parent_batch_id"):
            self._refresh_batch(updated["parent_batch_id"])
        return updated

    def open_viewer(self, run_id: str) -> dict:
        return self.viewer_controller.open(run_id)

    def close_viewer(self, run_id: str) -> dict:
        return self.viewer_controller.close(run_id)

    def viewer_status(self, run_id: str) -> dict:
        return self.viewer_controller.status(run_id)

    def _mirror_viewer_status(self, run_id: str, manifest: dict) -> dict:
        viewer = self.viewer_controller.status(run_id)
        if manifest.get("viewer") == viewer:
            return manifest
        return self.recorder.update_manifest(run_id, viewer=viewer)

    def stop(self, run_id: str) -> dict:
        try:
            manifest = self.get_run(run_id)
        except FileNotFoundError as exc:
            raise ValueError(f"unknown run id: {run_id}") from exc
        if manifest.get("status") in _TERMINAL_STATUSES:
            raise ValueError(f"run is already terminal: {run_id}")
        if manifest.get("executor_key") == "batch":
            return self._stop_batch(run_id, manifest)

        updated = self._stop_process(run_id, manifest)
        if updated.get("parent_batch_id"):
            self._refresh_batch(updated["parent_batch_id"])
        return updated

    def _stop_batch(self, batch_id: str, manifest: dict) -> dict:
        child_ids = list(manifest.get("child_run_ids", []))
        for child_id in child_ids:
            child = self.get_run(child_id)
            if child.get("status") == "running":
                try:
                    self._stop_process(child_id, child)
                except ValueError:
                    child = self.get_run(child_id)
                    if child.get("status") not in _TERMINAL_STATUSES:
                        raise
            elif child.get("status") == "queued":
                self._stop_queued_child(child_id)

        terminal_children = [
            self.get_run(child_id)
            for child_id in child_ids
            if self.get_run(child_id).get("status") in _TERMINAL_STATUSES
        ]
        updated = self.recorder.update_manifest(
            batch_id,
            status="stopped",
            phase="scheduler",
            result_integrity="unavailable",
            error={"failure_type": "batch_stopped", "message": "batch stopped by request"},
            progress={"completed": len(terminal_children), "total": len(child_ids)},
            timestamps={"finished_at": _timestamp()},
        )
        updated = self._persist_terminal_result(updated, terminal_children)
        self.recorder.append_event(batch_id, "batch_stopped")
        return self._mirror_viewer_status(batch_id, updated)

    def _stop_queued_child(self, run_id: str) -> dict:
        updated = self.recorder.update_manifest(
            run_id,
            status="stopped",
            phase="scheduler",
            result_integrity="unavailable",
            error={"failure_type": "batch_stopped", "message": "batch stopped by request"},
            timestamps={"finished_at": _timestamp()},
        )
        updated = self._persist_terminal_result(updated)
        self.recorder.append_event(run_id, "run_stopped", failure_type="batch_stopped")
        return self._mirror_viewer_status(run_id, updated)

    def _stop_process(self, run_id: str, manifest: dict) -> dict:
        process = self._processes.get(run_id)
        if process is None:
            raise ValueError(f"run is not registered with this orchestrator: {run_id}")

        return_code = process.poll()
        if return_code is not None:
            self._finalize_natural_process(run_id, return_code)
            raise ValueError(f"run is already terminal: {run_id}")

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        return_code = process.poll()
        # OS APIs cannot reliably distinguish an independent natural exit from
        # one caused by terminate(), so classify the accepted stop request.
        updated = self._finalize_process(run_id, status="stopped", return_code=return_code, event="run_stopped")
        return updated

    def get_run(self, run_id: str) -> dict:
        return self.recorder.read_manifest(run_id)

    def get_run_detail(self, run_id: str) -> dict:
        manifest = self.get_run(run_id)
        return {
            **manifest,
            "spec": self.recorder.read_spec(run_id),
            "events": self.recorder.read_events(run_id, limit=50),
            "log_tail": self.recorder.read_log_tail(run_id, max_lines=80),
        }

    def list_runs(self) -> list[dict]:
        return self.recorder.list_manifests()

    def recover_interrupted_runs(self) -> list[dict]:
        recovered: list[dict] = []
        for manifest in self.recorder.list_manifests():
            run_id = str(manifest["run_id"])
            if manifest.get("status") not in {"queued", "running"} or run_id in self._processes:
                continue
            updated = self.recorder.update_manifest(
                run_id,
                status="failed",
                error={"failure_type": "service_restart"},
                timestamps={"finished_at": _timestamp()},
            )
            child_ids = list(updated.get("child_run_ids", []))
            children = [
                self.get_run(child_id)
                for child_id in child_ids
                if _manifest_exists(self.recorder, child_id)
            ]
            updated = self._persist_terminal_result(updated, children if child_ids else None)
            self.recorder.append_event(run_id, "run_failed", failure_type="service_restart")
            recovered.append(self._mirror_viewer_status(run_id, updated))
        return recovered

    def _release_process(self, run_id: str) -> None:
        self._processes.pop(run_id, None)
        self._plans.pop(run_id, None)
        log_handle = self._logs.pop(run_id, None)
        if log_handle is not None:
            log_handle.close()

    def _terminate_and_release_process(self, run_id: str, process: subprocess.Popen) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            self._release_process(run_id)

    def _finalize_natural_process(self, run_id: str, return_code: int) -> dict:
        status = "completed" if return_code == 0 else "failed"
        return self._finalize_process(run_id, status=status, return_code=return_code, event="run_finished")

    def _finalize_process(self, run_id: str, *, status: str, return_code: int | None, event: str) -> dict:
        log_path = self.recorder.log_path(run_id)
        log_text = _read_log(self._logs.get(run_id), log_path)
        artifacts = discover_artifacts(self._plans[run_id], log_text)
        error = None if status != "failed" else {
            "failure_type": "executor",
            "return_code": return_code,
        }
        updated = self.recorder.update_manifest(
            run_id,
            status=status,
            artifacts=artifacts,
            return_code=return_code,
            log_path=str(log_path),
            error=error,
            timestamps={"finished_at": _timestamp()},
        )
        updated = self._persist_terminal_result(updated)
        updated = self._mirror_viewer_status(run_id, updated)
        if event == "run_finished":
            self.recorder.append_event(run_id, event, status=status, return_code=return_code)
        else:
            self.recorder.append_event(run_id, event, return_code=return_code)
        self._release_process(run_id)
        return updated

    def _persist_terminal_result(self, manifest: dict, children: list[dict] | None = None) -> dict:
        result = normalize_batch(manifest, children) if children is not None else inspect_result(manifest)
        return self.recorder.update_manifest(
            str(manifest["run_id"]),
            result_integrity=result["result_integrity"],
            artifacts=result["artifacts"],
        )


_TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped"})


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shared_preflight_planning_error(errors: list[Exception | None]) -> Exception | None:
    for error in errors:
        if error is not None and getattr(error, "failure_scope", None) == "shared":
            return error
    return None


def _read_log(log_handle: Any, log_path: Path) -> str:
    if log_handle is not None:
        log_handle.flush()
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _manifest_exists(recorder: RunRecorder, run_id: str) -> bool:
    try:
        recorder.read_manifest(run_id)
    except FileNotFoundError:
        return False
    return True
