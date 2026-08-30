from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from threading import RLock
from typing import Any, Literal
import uuid

import yaml

from .spec import ResolvedExperimentSpec


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_IMMUTABLE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "source_spec_path",
        "resolved_spec_path",
        "git_commit",
    }
)
_EVENT_TAIL_MAX_BYTES = 64 * 1024
_TERMINAL_INTEGRITIES = {
    "completed": frozenset({"complete", "unavailable"}),
    "failed": frozenset({"partial", "unavailable"}),
    "stopped": frozenset({"partial", "unavailable"}),
}
_NEXT_STATUSES = {
    "created": frozenset({"validating"}),
    "validating": frozenset({"running"}),
    "running": frozenset(_TERMINAL_INTEGRITIES),
    "completed": frozenset(),
    "failed": frozenset(),
    "stopped": frozenset(),
}
_STATUSES = frozenset(_NEXT_STATUSES)
_RESULT_INTEGRITIES = frozenset({"pending", "complete", "partial", "unavailable"})


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path


class RunRecorder:
    """Persist immutable run inputs and append-only benchmark evidence."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(os.path.abspath(os.fspath(output_root)))
        self._lock = RLock()
        self._run_id_suffix = 0
        self._validate_root_ancestor()

    def create_run(
        self,
        source_path: Path,
        resolved_spec: ResolvedExperimentSpec,
        *,
        git_commit: str | None = None,
        run_id: str | None = None,
    ) -> RunContext:
        if run_id is not None:
            self._validate_run_id(run_id)
        if not isinstance(resolved_spec, ResolvedExperimentSpec):
            raise ValueError("resolved_spec must be a ResolvedExperimentSpec")
        if git_commit is not None and not isinstance(git_commit, str):
            raise ValueError("git_commit must be a string or null")

        source_bytes = Path(source_path).read_bytes()
        resolved_yaml = yaml.safe_dump(
            resolved_spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
        now = _timestamp()

        with self._lock:
            output_root = self._output_root(create=True)
            if run_id is None:
                while True:
                    candidate = self._new_run_id()
                    run_dir = self._run_dir(
                        candidate, output_root=output_root, require_exists=False
                    )
                    if not run_dir.exists() and not run_dir.is_symlink():
                        run_id = candidate
                        break
            else:
                run_dir = self._run_dir(
                    run_id, output_root=output_root, require_exists=False
                )
                if run_dir.exists() or run_dir.is_symlink():
                    raise FileExistsError(f"run already exists: {run_id}")

            assert run_id is not None
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "status": "created",
                "result_integrity": "pending",
                "phase": "created",
                "source_spec_path": "source_spec.yaml",
                "resolved_spec_path": "resolved_spec.yaml",
                "git_commit": git_commit,
                "timestamps": {"created_at": now},
                "progress": {
                    "task_id": None,
                    "initial_state_id": None,
                    "episode": 0,
                    "episode_total": 0,
                    "step": 0,
                    "max_steps": 0,
                },
                "viewer": {"enabled": resolved_spec.viewer.enabled, "status": "closed"},
                "artifacts": [],
                "error": None,
            }
            staging_dir = _unique_staging_dir(output_root, run_id)
            try:
                (staging_dir / "source_spec.yaml").write_bytes(source_bytes)
                with (staging_dir / "resolved_spec.yaml").open(
                    "w", encoding="utf-8", newline=""
                ) as resolved_file:
                    resolved_file.write(resolved_yaml)
                self._atomic_write_json(staging_dir / "manifest.json", manifest)
                for name in ("events.jsonl", "steps.jsonl", "trials.jsonl", "run.log"):
                    (staging_dir / name).touch(exist_ok=False)
                os.replace(staging_dir, run_dir)
            except BaseException:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise
            return RunContext(run_id=run_id, run_dir=run_dir)

    def read_manifest(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run_dir = self._run_dir(run_id)
            path = self._confined_file(run_dir, "manifest.json")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                raise FileNotFoundError(f"manifest not found for run: {run_id}") from None
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"could not read manifest for run: {run_id}") from exc
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed manifest for run: {run_id}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"malformed manifest for run: {run_id}")
            self._validate_manifest_identity(value, run_id, run_dir)
            return deepcopy(value)

    def update_manifest(self, run_id: str, /, **updates: Any) -> dict[str, Any]:
        snapshot = _json_snapshot(updates)
        protected = _IMMUTABLE_MANIFEST_FIELDS.intersection(snapshot)
        if protected:
            names = ", ".join(sorted(protected))
            raise ValueError(f"immutable manifest fields cannot be updated: {names}")
        if "timestamps" in snapshot:
            timestamps = snapshot["timestamps"]
            if not isinstance(timestamps, dict):
                raise ValueError("timestamps update must be an object")
            if "created_at" in timestamps:
                raise ValueError("timestamps.created_at is immutable")
        with self._lock:
            run_dir = self._run_dir(run_id)
            manifest = self.read_manifest(run_id)
            self._validate_lifecycle_transition(manifest, snapshot)
            if "timestamps" in snapshot:
                merged_timestamps = deepcopy(manifest["timestamps"])
                merged_timestamps.update(snapshot["timestamps"])
                snapshot["timestamps"] = merged_timestamps
            manifest.update(snapshot)
            self._atomic_write_json(self._confined_file(run_dir, "manifest.json"), manifest)
            return deepcopy(manifest)

    def finalize(
        self,
        run_id: str,
        *,
        status: Literal["completed", "failed", "stopped"],
        result_integrity: Literal["complete", "partial", "unavailable"],
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if result_integrity not in _TERMINAL_INTEGRITIES.get(status, frozenset()):
            raise ValueError("invalid terminal status and result_integrity combination")
        return self.update_manifest(
            run_id,
            status=status,
            result_integrity=result_integrity,
            phase="terminal",
            timestamps={"finished_at": _timestamp()},
            error=error,
        )

    def append_event(self, run_id: str, event: dict[str, Any] | str, **fields: Any) -> dict[str, Any]:
        if isinstance(event, str):
            record: dict[str, Any] = {"event": event}
        elif isinstance(event, dict):
            record = dict(event)
        else:
            raise ValueError("event must be a string or object")
        record.update(fields)
        record["run_id"] = run_id
        record.setdefault("timestamp", _timestamp())
        return self._append_record(run_id, "events.jsonl", record)

    def append_step(self, run_id: str, step: dict[str, Any]) -> dict[str, Any]:
        return self._append_record(run_id, "steps.jsonl", step)

    def append_trial(self, run_id: str, trial: dict[str, Any]) -> dict[str, Any]:
        return self._append_record(run_id, "trials.jsonl", trial)

    def log_path(self, run_id: str) -> Path:
        with self._lock:
            return self._run_file(run_id, "run.log")

    def read_log_tail(
        self, run_id: str, *, max_lines: int = 80, max_chars: int = 12_000
    ) -> list[str]:
        max_lines = max(0, min(int(max_lines), 500))
        max_chars = max(0, min(int(max_chars), 100_000))
        if max_lines == 0 or max_chars == 0:
            return []
        with self._lock:
            try:
                return _tail_lines(
                    self._run_file(run_id, "run.log"), max_lines, max_bytes=max_chars
                )
            except FileNotFoundError:
                return []

    def _append_record(
        self, run_id: str, name: str, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        snapshot = _json_snapshot(record)
        with self._lock:
            self._append_jsonl(self._run_file(run_id, name), snapshot)
        return deepcopy(snapshot)

    def _run_file(self, run_id: str, name: str) -> Path:
        return self._confined_file(self._run_dir(run_id), name)

    def _run_dir(
        self,
        run_id: str,
        *,
        output_root: Path | None = None,
        require_exists: bool = True,
    ) -> Path:
        self._validate_run_id(run_id)
        if output_root is None:
            output_root = self._output_root(create=False)
            if output_root is None:
                raise FileNotFoundError(f"run not found: {run_id}")
        candidate = output_root / run_id
        if not _is_within(candidate, output_root) or candidate.is_symlink():
            raise ValueError("run path escapes the output root")
        if require_exists and not candidate.is_dir():
            raise FileNotFoundError(f"run not found: {run_id}")
        if candidate.exists():
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise ValueError("run path escapes the output root") from exc
            if not _is_within(resolved, output_root):
                raise ValueError("run path escapes the output root")
        return candidate

    def _output_root(self, *, create: bool) -> Path | None:
        self._validate_root_ancestor()
        if self.output_root.is_symlink():
            raise ValueError("output_root must not be a symlink")
        if create:
            self.output_root.mkdir(parents=True, exist_ok=True)
        elif not self.output_root.is_dir():
            return None
        try:
            resolved = self.output_root.resolve(strict=True)
        except OSError as exc:
            raise ValueError("output_root escapes its configured path") from exc
        if _path_key(resolved) != _path_key(self.output_root):
            raise ValueError("output_root escapes its configured path")
        return resolved

    @staticmethod
    def _confined_file(run_dir: Path, name: str) -> Path:
        path = run_dir / name
        if path.is_symlink():
            raise ValueError("run file escapes the run directory")
        if path.exists():
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise ValueError("run file escapes the run directory") from exc
            if not _is_within(resolved, run_dir):
                raise ValueError("run file escapes the run directory")
        return path

    def _validate_root_ancestor(self) -> None:
        ancestor = self.output_root
        while not ancestor.exists() and not ancestor.is_symlink():
            parent = ancestor.parent
            if parent == ancestor:
                return
            ancestor = parent
        try:
            resolved = ancestor.resolve(strict=True)
        except OSError as exc:
            raise ValueError("output_root must not traverse a symlink or junction") from exc
        if _path_key(resolved) != _path_key(ancestor):
            raise ValueError("output_root must not traverse a symlink or junction")

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not isinstance(run_id, str) or _SAFE_RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must contain only letters, numbers, underscores, and hyphens")

    @staticmethod
    def _validate_lifecycle_transition(
        manifest: dict[str, Any], updates: dict[str, Any]
    ) -> None:
        current_status = manifest.get("status")
        current_integrity = manifest.get("result_integrity")
        if current_status not in _STATUSES or current_integrity not in _RESULT_INTEGRITIES:
            raise ValueError("manifest lifecycle is invalid")

        next_status = updates.get("status", current_status)
        next_integrity = updates.get("result_integrity", current_integrity)
        if next_status not in _STATUSES:
            raise ValueError("status is not supported by the run lifecycle")
        if next_integrity not in _RESULT_INTEGRITIES:
            raise ValueError("result_integrity is not supported by the run lifecycle")

        if next_status == current_status and next_integrity == current_integrity:
            return
        if next_status not in _NEXT_STATUSES[current_status]:
            raise ValueError("invalid lifecycle transition")
        if next_status in {"created", "validating", "running"}:
            if next_integrity != "pending":
                raise ValueError("invalid lifecycle transition")
            return
        if next_integrity not in _TERMINAL_INTEGRITIES[next_status]:
            raise ValueError("invalid lifecycle transition")

    @staticmethod
    def _validate_manifest_identity(
        manifest: dict[str, Any], run_id: str, run_dir: Path
    ) -> None:
        timestamps = manifest.get("timestamps")
        if (
            type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest.get("run_id") != run_id
            or manifest.get("source_spec_path") != "source_spec.yaml"
            or manifest.get("resolved_spec_path") != "resolved_spec.yaml"
            or not isinstance(manifest.get("git_commit"), (str, type(None)))
            or not isinstance(timestamps, dict)
            or not isinstance(timestamps.get("created_at"), str)
            or not timestamps["created_at"]
        ):
            raise ValueError("manifest identity does not match its containing run directory")
        for name in ("source_spec.yaml", "resolved_spec.yaml"):
            if not RunRecorder._confined_file(run_dir, name).is_file():
                raise ValueError("manifest identity does not match its containing run directory")

    def _new_run_id(self) -> str:
        self._run_id_suffix += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"run_{timestamp}_{self._run_id_suffix:04d}"

    @staticmethod
    def _atomic_write_json(target: Path, value: Any) -> None:
        text = _json_text(value)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _append_jsonl(target: Path, value: dict[str, Any]) -> None:
        with target.open("a", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_staging_dir(output_root: Path, run_id: str) -> Path:
    for _attempt in range(100):
        candidate = output_root / f"stg-{uuid.uuid4().hex[:8]}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"could not create a unique staging directory for run: {run_id}")


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must contain only JSON-serializable data") from exc


def _json_snapshot(value: Any) -> Any:
    return json.loads(_json_text(value))


def _tail_lines(
    path: Path,
    limit: int,
    *,
    block_size: int = 8192,
    max_bytes: int | None = None,
) -> list[str]:
    if limit <= 0 or (max_bytes is not None and max_bytes <= 0):
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        remaining = min(position, max_bytes) if max_bytes is not None else position
        chunks: list[bytes] = []
        newline_count = 0
        while position > 0 and remaining > 0 and newline_count <= limit:
            read_size = min(block_size, position, remaining)
            position -= read_size
            remaining -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    return b"".join(reversed(chunks)).decode("utf-8", errors="replace").splitlines()[-limit:]


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([_path_key(path), _path_key(root)]) == _path_key(root)
    except ValueError:
        return False


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))
