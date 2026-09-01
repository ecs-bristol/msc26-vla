from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


_CORE_FILENAMES = ("metadata.json", "summary.csv", "trials.csv")
_OPTIONAL_FILENAMES = ("failures.csv",)
_VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".webm"})
_FRAME_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp"})


def inspect_result(manifest: dict) -> dict:
    """Inspect a terminal run's result files without changing executor data."""
    return normalize_run(manifest)


def normalize_run(manifest: dict) -> dict:
    """Return a stable console result object for one executor manifest."""
    artifacts, artifact_errors = _artifact_index(manifest)
    by_label = {item["label"]: item for item in artifacts}
    integrity: dict[str, list[Any]] = {
        "required": list(_CORE_FILENAMES),
        "optional": list(_OPTIONAL_FILENAMES),
        "errors": list(artifact_errors),
    }

    metadata: dict[str, Any] = {}
    summary: list[dict[str, str]] = []
    trials: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    valid_required: set[str] = set()

    metadata_artifact = by_label.get("metadata.json")
    if metadata_artifact is not None and metadata_artifact["exists"]:
        value, error = _read_json(Path(metadata_artifact["path"]))
        if error is not None:
            integrity["errors"].append(_artifact_error(metadata_artifact, error))
        elif isinstance(value, dict):
            metadata = value
            valid_required.add("metadata.json")
        else:
            integrity["errors"].append(_artifact_error(metadata_artifact, "metadata must be an object"))

    for label, destination in (("summary.csv", "summary"), ("trials.csv", "trials"), ("failures.csv", "failures")):
        artifact = by_label.get(label)
        if artifact is None or not artifact["exists"]:
            continue
        rows, error = _read_csv(Path(artifact["path"]))
        if error is not None:
            integrity["errors"].append(_artifact_error(artifact, error))
            continue
        if label in _CORE_FILENAMES:
            valid_required.add(label)
        if destination == "summary":
            summary = rows
        elif destination == "trials":
            trials = rows
        else:
            failures = rows

    if not failures and trials:
        failures = [
            dict(row)
            for row in trials
            if _is_false(row.get("success")) or (row.get("failure_type") not in (None, "", "none"))
        ]

    required_present = all(
        by_label.get(label) is not None and by_label[label]["exists"]
        for label in _CORE_FILENAMES
    )
    required_errors = any(
        error["label"] in _CORE_FILENAMES
        for error in integrity["errors"]
    )
    usable = bool(valid_required)
    if not usable:
        result_integrity = "unavailable"
    elif required_present and not required_errors and len(valid_required) == len(_CORE_FILENAMES):
        result_integrity = "complete"
    else:
        result_integrity = "partial"

    training_metrics = _training_metrics(metadata)
    return {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "result_integrity": result_integrity,
        "integrity": integrity,
        "summary": summary,
        "trials": trials,
        "failures": failures,
        "training_metrics": training_metrics,
        "artifacts": artifacts,
    }


def normalize_batch(parent: dict, children: list[dict]) -> dict:
    """Normalize child runs into comparison rows under their shared experiment identity."""
    normalized_children = [normalize_run(child) for child in children]
    shared_task, shared_environment = _shared_identity(parent)
    comparison_rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    artifacts: list[dict] = []
    errors: list[dict] = []
    training_metrics: dict[str, Any] = {}
    expected_child_ids = [
        str(run_id)
        for run_id in parent.get("child_run_ids", [])
        if isinstance(run_id, (str, int))
    ]
    present_child_ids = {str(child.get("run_id")) for child in children}
    missing_child_ids = [
        run_id
        for run_id in expected_child_ids
        if run_id not in present_child_ids
    ]

    for child, normalized in zip(children, normalized_children):
        task, environment = _shared_identity(child)
        task = shared_task or task
        environment = shared_environment or environment
        child_rows = normalized["summary"] or [{}]
        for source in child_rows:
            row = dict(source)
            row.update({
                "run_id": normalized["run_id"],
                "status": normalized["status"],
                "result_integrity": normalized["result_integrity"],
                "task_id": task.get("task_id", row.get("task_id", "")),
                "task_version": task.get("version", ""),
                "environment_key": environment.get("key", ""),
            })
            comparison_rows.append(row)
        trials.extend(normalized["trials"])
        failures.extend(normalized["failures"])
        artifacts.extend(normalized["artifacts"])
        errors.extend({"run_id": normalized["run_id"], **error} for error in normalized["integrity"]["errors"])
        if normalized["training_metrics"]:
            training_metrics[str(normalized["run_id"])] = normalized["training_metrics"]

    for missing_child_id in missing_child_ids:
        errors.append({
            "kind": "batch_child",
            "label": missing_child_id,
            "message": "declared child manifest is missing",
        })

    usable_children = [item for item in normalized_children if item["result_integrity"] != "unavailable"]
    if not usable_children:
        result_integrity = "unavailable"
    elif not missing_child_ids and len(usable_children) == len(normalized_children) and all(
        child.get("status") == "completed" and item["result_integrity"] == "complete"
        for child, item in zip(children, normalized_children)
    ):
        result_integrity = "complete"
    else:
        result_integrity = "partial"

    return {
        "run_id": parent.get("run_id"),
        "status": parent.get("status"),
        "result_integrity": result_integrity,
        "integrity": {"required": list(_CORE_FILENAMES), "optional": list(_OPTIONAL_FILENAMES), "errors": errors},
        "summary": comparison_rows,
        "trials": trials,
        "failures": failures,
        "training_metrics": training_metrics,
        "artifacts": _deduplicate_artifacts(artifacts),
    }


def _artifact_index(manifest: dict) -> tuple[list[dict], list[dict]]:
    manifest_paths: list[Path] = []
    manifest_artifacts = manifest.get("artifacts")
    if isinstance(manifest_artifacts, list):
        for item in manifest_artifacts:
            if isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"]:
                manifest_paths.append(Path(item["path"]))

    errors: list[dict] = []
    base_roots = _base_trusted_roots(manifest)
    result_dirs = _trusted_result_directories(manifest_paths, base_roots)
    trusted_roots = _trusted_roots(base_roots, result_dirs)
    paths: list[Path] = []
    for result_dir in result_dirs:
        for filename in (*_CORE_FILENAMES, *_OPTIONAL_FILENAMES):
            paths.append(result_dir / filename)
        if result_dir.is_dir():
            try:
                paths.extend(path for path in result_dir.rglob("*") if path.is_file())
            except OSError:
                continue

    for path in manifest_paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if any(_is_within(resolved, root) for root in trusted_roots):
            paths.append(path)

    trusted_paths = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        trusted_paths.add(resolved)

    for path in manifest_paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in trusted_paths:
            continue
        artifact = _normalize_artifact(path)
        if not any(_is_within(resolved, root) for root in trusted_roots):
            errors.append(_artifact_error(artifact, "artifact path is outside the trusted result root"))
    return _deduplicate_artifacts(_normalize_artifact(path) for path in paths), errors


def _base_trusted_roots(manifest: dict) -> list[Path]:
    roots: list[Path] = []
    command = manifest.get("command")
    cwd = command.get("cwd") if isinstance(command, dict) else None
    if isinstance(cwd, str) and cwd:
        try:
            resolved_cwd = Path(cwd).resolve()
        except OSError:
            resolved_cwd = None
        if resolved_cwd is not None and resolved_cwd.is_dir():
            roots.append(resolved_cwd)
    return roots


def _trusted_roots(base_roots: list[Path], result_dirs: list[Path]) -> list[Path]:
    roots = list(base_roots)
    for result_dir in result_dirs:
        try:
            resolved = result_dir.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in roots:
            roots.append(resolved)
    return roots


def _trusted_result_directories(paths: list[Path], base_roots: list[Path]) -> list[Path]:
    directories: list[Path] = []
    for path in paths:
        if path.name not in _CORE_FILENAMES:
            continue
        candidate = path.parent
        try:
            resolved_candidate = candidate.resolve()
        except OSError:
            continue
        if not any(_is_within(resolved_candidate, root) for root in base_roots):
            continue
        if not any((resolved_candidate / filename).exists() for filename in _CORE_FILENAMES):
            continue
        if candidate not in directories:
            directories.append(candidate)
    return directories


def _normalize_artifact(path: Path) -> dict:
    resolved = path.resolve()
    label = resolved.name
    return {
        "artifact_id": hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:24],
        "kind": _artifact_kind(resolved),
        "label": label,
        "exists": resolved.is_file(),
        "path": str(resolved),
    }


def _deduplicate_artifacts(artifacts: Any) -> list[dict]:
    result: dict[str, dict] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        result[artifact["path"]] = artifact
    return sorted(result.values(), key=lambda item: (item["label"].casefold(), item["path"].casefold()))


def _artifact_kind(path: Path) -> str:
    if path.name == "metadata.json":
        return "metadata"
    if path.name == "summary.csv":
        return "summary"
    if path.name == "trials.csv":
        return "trials"
    if path.name == "failures.csv":
        return "failures"
    if path.suffix.lower() in _VIDEO_SUFFIXES:
        return "video"
    if path.suffix.lower() in _FRAME_SUFFIXES:
        return "frame"
    return path.suffix.lower().lstrip(".") or "artifact"


def _read_json(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"could not read JSON: {exc}"


def _read_csv(path: Path) -> tuple[list[dict[str, str]], str | None]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            if not reader.fieldnames:
                return [], "CSV has no header"
            return [dict(row) for row in reader], None
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], f"could not read CSV: {exc}"


def _artifact_error(artifact: dict, message: str) -> dict:
    return {
        "artifact_id": artifact["artifact_id"],
        "kind": artifact["kind"],
        "label": artifact["label"],
        "path": artifact["path"],
        "message": message,
    }


def _training_metrics(metadata: dict[str, Any]) -> dict:
    for key in ("training_metrics", "train_metrics"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _is_false(value: Any) -> bool:
    return str(value).strip().casefold() in {"0", "false", "no"}


def _shared_identity(manifest: dict) -> tuple[dict, dict]:
    spec = manifest.get("spec") if isinstance(manifest.get("spec"), dict) else manifest
    task = spec.get("task") if isinstance(spec.get("task"), dict) else {}
    environment = spec.get("environment") if isinstance(spec.get("environment"), dict) else {}
    return task, environment


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents
