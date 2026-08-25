"""Materialize an offline, paired LIBERO Spatial pilot plan.

This command is intentionally dry-run only.  It never imports LeRobot, opens
LIBERO, or instantiates a model; its output is the immutable input and result
layout that a later, separately-authorized rollout runner must consume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

import yaml


SMOLVLA_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SMOLVLM2_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_HORIZONS = frozenset({1, 5, 10, 20, 50})
SUMMARY_FIELDS = (
    "condition",
    "task_id",
    "seed",
    "initial_state_id",
    "environment_seed",
    "inference_seed",
    "success_at_280",
    "success_step",
    "executed_env_steps",
    "wall_time_to_terminal_s",
    "model_invocations",
    "model_inference_time_s",
    "range_violations",
    "range_clips",
    "buffer_discards",
    "mean_actual_horizon",
    "action_trace_sha256",
    "termination_reason",
    "git_sha",
    "resolved_config_path",
)
_TERMINAL_EPISODE_STATUSES = frozenset({"completed", "failed"})
_INFERENCE_SEED_DERIVATION = "sha256(libero_spatial|task_id|seed|initial_state_id)[:8] & ((1<<63)-1)"


class PilotPolicy(Protocol):
    @property
    def telemetry(self) -> tuple[dict[str, object], ...]: ...

    @property
    def model_inference_time_s(self) -> float: ...

    def reset(self) -> None: ...

    def select_action(self, observation: object) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EpisodeTelemetrySnapshot:
    telemetry_start: int
    model_invocations_start: int
    model_inference_time_s_start: float


def _git_sha(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("pilot config must be a YAML mapping")
    return value


def _snapshot_path(value: str, revision: str, field: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{field} must be an existing absolute local snapshot directory")
    if path.name != revision or path.parent.name != "snapshots":
        raise ValueError(f"{field} must name the frozen snapshot revision {revision}")
    return str(path.resolve())


def _validate_config(config: dict[str, Any]) -> None:
    model = config.get("model")
    conditions = config.get("conditions")
    if config.get("suite") != "libero_spatial" or config.get("task_ids") != list(range(10)):
        raise ValueError("pilot config must cover the ten libero_spatial tasks")
    if config.get("episodes_per_task") != 5:
        raise ValueError("pilot config requires five episodes per task")
    if config.get("episode_cap") != 280 or config.get("batch_size") != 1:
        raise ValueError("pilot config requires episode_cap=280 and batch_size=1")
    if not isinstance(model, dict):
        raise ValueError("pilot config must contain a model mapping")
    if (
        model.get("base_revision") != SMOLVLA_REVISION
        or model.get("vlm_revision") != SMOLVLM2_REVISION
        or model.get("local_files_only") is not True
        or model.get("num_steps") != 2
        or model.get("chunk_size") != 50
    ):
        raise ValueError("pilot config must retain the frozen local SmolVLA protocol")
    if not isinstance(conditions, list) or len(conditions) != 6:
        raise ValueError("pilot config must declare the six paired conditions")
    names = [condition.get("name") for condition in conditions if isinstance(condition, dict)]
    if names != ["Static-H1", "Static-H5", "Static-H10", "Static-H20", "Static-H50", "Adaptive-H20→H1"]:
        raise ValueError("pilot conditions must be the five static horizons and Adaptive-H20→H1")
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("each condition must be a mapping")
        if type(condition.get("fixed_h")) is not int or condition["fixed_h"] not in ALLOWED_HORIZONS:
            raise ValueError("condition fixed_h must be one of 1, 5, 10, 20, 50")
        if condition.get("safety_enabled") is not True:
            raise ValueError("all paired-pilot conditions require safety_enabled=True")
    adaptive = conditions[-1]
    if adaptive["fixed_h"] != 20 or adaptive.get("replan_after_safety_violation") is not True:
        raise ValueError("Adaptive-H20→H1 must replan after a safety violation")
    if any(condition.get("replan_after_safety_violation") is not False for condition in conditions[:-1]):
        raise ValueError("static paired-pilot conditions must not safety-trigger replanning")


def _trials(config: dict[str, Any]) -> list[dict[str, int]]:
    seed = config["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    trials = [
        {
            "task_id": task_id,
            "seed": seed + episode_index,
            "initial_state_id": episode_index,
            "episode_index": episode_index,
        }
        for task_id in config["task_ids"]
        for episode_index in range(config["episodes_per_task"])
    ]
    if len({(trial["task_id"], trial["seed"], trial["initial_state_id"]) for trial in trials}) != len(trials):
        raise ValueError("paired trial keys must be unique")
    return trials


def _inference_seed(trial: dict[str, int]) -> int:
    """Derive a condition-independent reproducible inference RNG seed."""

    pairing_key = "|".join(
        str(value)
        for value in (
            "libero_spatial",
            trial["task_id"],
            trial["seed"],
            trial["initial_state_id"],
        )
    )
    digest = hashlib.sha256(pairing_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def _set_inference_seed(seed: int) -> None:
    """Set all RNGs used by the frozen policy before one episode begins."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_provenance(trials: list[dict[str, int]]) -> list[dict[str, int]]:
    return [
        {
            "task_id": trial["task_id"],
            "initial_state_id": trial["initial_state_id"],
            "environment_seed": trial["seed"],
            "inference_seed": _inference_seed(trial),
        }
        for trial in trials
    ]


def _action_trace_sha256(actions: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(b"libero-paired-pilot-action-trace-v1\0")
    digest.update(len(actions).to_bytes(8, byteorder="big"))
    for action in actions:
        value = np.ascontiguousarray(np.asarray(action, dtype=np.float32))
        if value.shape != (7,):
            raise ValueError("action trace accepts only LIBERO (7,) actions")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _condition_slug(name: str) -> str:
    return name.replace("→", "-to-").lower()


def _write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _episode_path(output_dir: Path, condition_name: str, trial: dict[str, int]) -> Path:
    return (
        output_dir
        / "episodes"
        / _condition_slug(condition_name)
        / f"task_{trial['task_id']:02d}_seed_{trial['seed']}_state_{trial['initial_state_id']}.json"
    )


def materialize_dry_run(
    *, config_path: Path, output_dir: Path, base_snapshot_path: str, vlm_snapshot_path: str
) -> dict[str, Any]:
    """Write immutable paired inputs and planned per-episode result files."""

    config_path = config_path.resolve(strict=True)
    config = _read_config(config_path)
    _validate_config(config)
    config["model"]["base_snapshot_path"] = _snapshot_path(
        base_snapshot_path, SMOLVLA_REVISION, "base_snapshot_path"
    )
    config["model"]["vlm_snapshot_path"] = _snapshot_path(
        vlm_snapshot_path, SMOLVLM2_REVISION, "vlm_snapshot_path"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_config.json"
    if resolved_path.exists() and _read_json(resolved_path) != config:
        raise ValueError("existing resolved_config.json does not match the requested dry run")
    if not resolved_path.exists():
        _write_json(resolved_path, config)
    sha = _git_sha(PROJECT_ROOT)
    trials = _trials(config)
    manifest = {
        "schema_version": 1,
        "suite": config["suite"],
        "episodes_per_task": config["episodes_per_task"],
        "pairing_key": ["task_id", "seed", "initial_state_id"],
        "seed_strategy": "seed = configured seed + initial_state_id; every condition reuses this exact manifest",
        "trials": trials,
    }
    manifest_path = output_dir / "paired_manifest.json"
    if manifest_path.exists() and _read_json(manifest_path) != manifest:
        raise ValueError("existing paired_manifest.json does not match the requested dry run")
    if not manifest_path.exists():
        _write_json(manifest_path, manifest)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    provenance = {
        "schema_version": 1,
        "mode": "dry_run",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": sha,
        "config_source_path": str(config_path),
        "resolved_config_path": str(resolved_path.resolve()),
        "paired_manifest_path": str(manifest_path.resolve()),
        "paired_manifest_sha256": manifest_sha256,
        "local_files_only": True,
        "base_snapshot_path": config["model"]["base_snapshot_path"],
        "vlm_snapshot_path": config["model"]["vlm_snapshot_path"],
        "base_revision": SMOLVLA_REVISION,
        "vlm_revision": SMOLVLM2_REVISION,
        "num_steps": 2,
        "chunk_size": 50,
        "batch_size": 1,
        "episode_cap": 280,
        "environment_seed_source": "paired_manifest.trials[].seed",
        "inference_seed_derivation": _INFERENCE_SEED_DERIVATION,
        "pairing_seeds": _seed_provenance(trials),
    }
    _write_json(output_dir / "provenance.json", provenance)

    rows: list[dict[str, object]] = []
    for condition in config["conditions"]:
        for trial in trials:
            episode_path = _episode_path(output_dir, condition["name"], trial)
            result = {
                "schema_version": 1,
                "status": "planned_dry_run",
                "condition": condition["name"],
                "condition_config": condition,
                **trial,
                "environment_seed": trial["seed"],
                "inference_seed": _inference_seed(trial),
                "success_at_280": None,
                "success_step": None,
                "executed_env_steps": None,
                "wall_time_to_terminal_s": None,
                "model_invocations": None,
                "model_inference_time_s": None,
                "range_violations": None,
                "range_clips": None,
                "buffer_discards": None,
                "mean_actual_horizon": None,
                "action_trace_sha256": None,
                "termination_reason": "not_started_dry_run",
                "git_sha": sha,
                "resolved_config_path": str(resolved_path.resolve()),
            }
            if not episode_path.exists():
                _write_json(episode_path, result)
            persisted = _read_json(episode_path)
            rows.append({field: persisted.get(field) for field in SUMMARY_FIELDS})
    _write_summary(output_dir / "summary.csv", rows)
    command = shlex.join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dry-run",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--base-snapshot-path",
            config["model"]["base_snapshot_path"],
            "--vlm-snapshot-path",
            config["model"]["vlm_snapshot_path"],
        ]
    )
    _atomic_write_text(output_dir / "dry_run_command.txt", command + "\n")
    return {"output_dir": str(output_dir), "planned_episodes": len(rows), "command": command}


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_execution_inputs(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, int]]]:
    """Read, rather than regenerate, the immutable dry-run configuration and pairs."""

    resolved_path = output_dir / "resolved_config.json"
    manifest_path = output_dir / "paired_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"paired_manifest.json is required before --execute: {manifest_path}"
        )
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"resolved_config.json is required before --execute: {resolved_path}"
        )
    config = _read_json(resolved_path)
    _validate_config(config)
    model = config["model"]
    _snapshot_path(model["base_snapshot_path"], SMOLVLA_REVISION, "base_snapshot_path")
    _snapshot_path(model["vlm_snapshot_path"], SMOLVLM2_REVISION, "vlm_snapshot_path")
    manifest = _read_json(manifest_path)
    if manifest.get("suite") != config["suite"]:
        raise ValueError("paired manifest suite does not match resolved config")
    if manifest.get("pairing_key") != ["task_id", "seed", "initial_state_id"]:
        raise ValueError("paired manifest must use task_id, seed, initial_state_id pairing")
    trials = manifest.get("trials")
    if not isinstance(trials, list) or any(not isinstance(trial, dict) for trial in trials):
        raise ValueError("paired manifest trials must be a list of objects")
    required = {"task_id", "seed", "initial_state_id", "episode_index"}
    if any(set(trial) != required for trial in trials):
        raise ValueError("paired manifest trials have an invalid schema")
    typed_trials: list[dict[str, int]] = []
    for trial in trials:
        if any(type(trial[field]) is not int for field in required):
            raise ValueError("paired manifest trial fields must be integers")
        typed_trials.append({field: trial[field] for field in required})
    if typed_trials != _trials(config):
        raise ValueError("paired manifest does not match the existing dry-run trial plan")
    return config, typed_trials


def _selected_trials(
    trials: list[dict[str, int]],
    task_ids: set[int] | None,
    episodes_per_task: int | None,
) -> list[dict[str, int]]:
    selected_tasks = task_ids if task_ids is not None else {trial["task_id"] for trial in trials}
    if not selected_tasks or any(type(task_id) is not int or task_id < 0 for task_id in selected_tasks):
        raise ValueError("task_ids must be non-empty non-negative integers")
    if episodes_per_task is not None and episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    selected: list[dict[str, int]] = []
    for task_id in sorted(selected_tasks):
        task_trials = [trial for trial in trials if trial["task_id"] == task_id]
        if not task_trials:
            raise ValueError(f"task_id {task_id} is absent from paired_manifest.json")
        task_trials.sort(key=lambda trial: trial["episode_index"])
        if episodes_per_task is not None:
            task_trials = task_trials[:episodes_per_task]
        selected.extend(task_trials)
    return selected


def _episode_summary_row(result: dict[str, object]) -> dict[str, object]:
    missing = [field for field in SUMMARY_FIELDS if field not in result]
    if missing:
        raise ValueError(f"episode result is missing summary fields: {missing}")
    return {field: result[field] for field in SUMMARY_FIELDS}


def _refresh_summary(
    output_dir: Path, config: dict[str, Any], trials: list[dict[str, int]]
) -> None:
    rows = [
        _episode_summary_row(_read_json(_episode_path(output_dir, condition["name"], trial)))
        for condition in config["conditions"]
        for trial in trials
    ]
    _write_summary(output_dir / "summary.csv", rows)


def _telemetry_metrics(records: tuple[dict[str, object], ...]) -> dict[str, object]:
    releases = [record for record in records if record.get("event") == "action_release"]
    horizons = [float(record["actual_horizon"]) for record in releases]
    return {
        "model_invocations": sum(bool(record.get("model_invoked")) for record in releases),
        "range_violations": sum(bool(record.get("range_violation")) for record in releases),
        "range_clips": sum(bool(record.get("range_clipped")) for record in releases),
        "buffer_discards": sum(bool(record.get("buffer_discarded")) for record in releases),
        "mean_actual_horizon": sum(horizons) / len(horizons) if horizons else None,
    }


def _model_invocation_count(records: tuple[dict[str, object], ...]) -> int:
    return sum(
        record.get("event") == "action_release" and record.get("model_invoked") is True
        for record in records
    )


def _episode_telemetry_snapshot(policy: PilotPolicy) -> EpisodeTelemetrySnapshot:
    records = tuple(policy.telemetry)
    inference_time = float(policy.model_inference_time_s)
    if inference_time < 0.0:
        raise ValueError("policy model_inference_time_s must be non-negative")
    return EpisodeTelemetrySnapshot(
        telemetry_start=len(records),
        model_invocations_start=_model_invocation_count(records),
        model_inference_time_s_start=inference_time,
    )


def _run_episode(
    *,
    output_dir: Path,
    config: dict[str, Any],
    condition: dict[str, Any],
    trial: dict[str, int],
    backend: object,
    policy: PilotPolicy,
    inference_seed_setter: Callable[[int], None] = _set_inference_seed,
) -> dict[str, object]:
    """Run one manifest-selected environment episode and return a terminal record."""

    inference_seed = _inference_seed(trial)
    inference_seed_setter(inference_seed)
    episode = backend.open_episode(
        config["suite"],
        trial["task_id"],
        trial["initial_state_id"],
        config["episode_cap"],
        trial["seed"],
    )
    started_at = time.perf_counter()
    steps = 0
    success = False
    success_step: int | None = None
    telemetry_snapshot: EpisodeTelemetrySnapshot | None = None
    action_trace: list[np.ndarray] = []
    termination_reason = "max_steps"
    status = "completed"
    try:
        observation = episode.reset()
        policy.reset()
        telemetry_snapshot = _episode_telemetry_snapshot(policy)
        for step_id in range(1, config["episode_cap"] + 1):
            action = np.asarray(policy.select_action(observation), dtype=np.float32)
            if action.shape != (7,):
                raise ValueError(f"policy action must have shape (7,), got {action.shape}")
            action_trace.append(action.copy())
            step = episode.step(action)
            steps = step_id
            observation = step.observation
            if step.success:
                success = True
                success_step = step_id
                termination_reason = "success"
                break
            if step.done:
                termination_reason = "done"
                break
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        status = "failed"
        termination_reason = f"error:{type(exc).__name__}"
    finally:
        episode.close()

    if telemetry_snapshot is None:
        # reset failed before a policy snapshot could be taken; report no
        # synthetic historical telemetry from an earlier episode.
        episode_records: tuple[dict[str, object], ...] = ()
        model_invocations = 0
        inference_seconds = 0.0
    else:
        all_records = tuple(policy.telemetry)
        episode_records = all_records[telemetry_snapshot.telemetry_start :]
        model_invocations = (
            _model_invocation_count(all_records)
            - telemetry_snapshot.model_invocations_start
        )
        inference_seconds = (
            float(policy.model_inference_time_s)
            - telemetry_snapshot.model_inference_time_s_start
        )
        if model_invocations < 0 or inference_seconds < 0.0:
            raise RuntimeError("policy telemetry counters must be monotonic within a condition")
    metrics = _telemetry_metrics(episode_records)
    metrics["model_invocations"] = model_invocations
    return {
        "schema_version": 1,
        "status": status,
        "condition": condition["name"],
        "condition_config": condition,
        **trial,
        "environment_seed": trial["seed"],
        "inference_seed": inference_seed,
        "success_at_280": success,
        "success_step": success_step,
        "executed_env_steps": steps,
        "wall_time_to_terminal_s": time.perf_counter() - started_at,
        "model_inference_time_s": inference_seconds,
        **metrics,
        "action_trace_sha256": _action_trace_sha256(action_trace),
        "termination_reason": termination_reason,
        "git_sha": _git_sha(PROJECT_ROOT),
        "resolved_config_path": str((output_dir / "resolved_config.json").resolve()),
    }


def execute_pilot(
    *,
    output_dir: Path,
    backend_factory: Callable[[], object],
    policy_factory: Callable[[dict[str, Any], dict[str, Any]], PilotPolicy],
    task_ids: set[int] | None = None,
    episodes_per_task: int | None = None,
    inference_seed_setter: Callable[[int], None] = _set_inference_seed,
) -> dict[str, int]:
    """Execute only planned pairs, atomically persisting each terminal result.

    Existing terminal episode files are never overwritten, so a resumed process
    only opens missing or non-terminal planned episodes.
    """

    output_dir = output_dir.resolve()
    config, trials = _read_execution_inputs(output_dir)
    selected_trials = _selected_trials(trials, task_ids, episodes_per_task)
    manifest_path = output_dir / "paired_manifest.json"
    _write_json(
        output_dir / "execution_provenance.json",
        {
            "schema_version": 1,
            "mode": "execute",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(PROJECT_ROOT),
            "resolved_config_path": str((output_dir / "resolved_config.json").resolve()),
            "paired_manifest_path": str(manifest_path.resolve()),
            "paired_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "selected_task_ids": sorted({trial["task_id"] for trial in selected_trials}),
            "selected_episodes_per_task": episodes_per_task,
            "local_files_only": config["model"]["local_files_only"],
            "base_snapshot_path": config["model"]["base_snapshot_path"],
            "vlm_snapshot_path": config["model"]["vlm_snapshot_path"],
            "environment_seed_source": "paired_manifest.trials[].seed",
            "inference_seed_derivation": _INFERENCE_SEED_DERIVATION,
            "pairing_seeds": _seed_provenance(selected_trials),
        },
    )
    backend = backend_factory()
    executed = 0
    skipped = 0
    try:
        for condition in config["conditions"]:
            pending = [
                trial
                for trial in selected_trials
                if _read_json(_episode_path(output_dir, condition["name"], trial)).get("status")
                not in _TERMINAL_EPISODE_STATUSES
            ]
            if not pending:
                skipped += len(selected_trials)
                continue
            policy = policy_factory(condition, config)
            try:
                for trial in selected_trials:
                    path = _episode_path(output_dir, condition["name"], trial)
                    if _read_json(path).get("status") in _TERMINAL_EPISODE_STATUSES:
                        skipped += 1
                        continue
                    result = _run_episode(
                        output_dir=output_dir,
                        config=config,
                        condition=condition,
                        trial=trial,
                        backend=backend,
                        policy=policy,
                        inference_seed_setter=inference_seed_setter,
                    )
                    # Re-read just before atomic replacement: completed results
                    # from a prior resume are an immutable boundary.
                    if _read_json(path).get("status") in _TERMINAL_EPISODE_STATUSES:
                        skipped += 1
                        continue
                    _write_json(path, result)
                    _refresh_summary(output_dir, config, trials)
                    executed += 1
            finally:
                policy.close()
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    return {"executed_episodes": executed, "skipped_episodes": skipped}


class _LocalSmolVLAPilotPolicy:
    """Minimal direct adapter from a LIBERO Observation to the frozen plugin."""

    def __init__(self, policy: object, torch: object) -> None:
        self._policy = policy
        self._torch = torch
        self._model_inference_time_s = 0.0

    @property
    def telemetry(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(record) for record in self._policy.action_telemetry)

    @property
    def model_inference_time_s(self) -> float:
        """Cumulative time spent in calls that actually invoked the model."""

        return self._model_inference_time_s

    def reset(self) -> None:
        self._policy.reset()

    def select_action(self, observation: object) -> np.ndarray:
        images = observation.images
        agentview = np.asarray(images["agentview"], dtype=np.uint8)
        wrist = np.asarray(images["wrist"], dtype=np.uint8)
        if agentview.ndim != 3 or wrist.ndim != 3:
            raise ValueError("LIBERO images must be HWC RGB arrays")
        torch = self._torch
        batch = {
            "observation.images.image": torch.from_numpy(
                np.ascontiguousarray(agentview.transpose(2, 0, 1))
            ).unsqueeze(0).to(dtype=torch.float32).div(255.0),
            "observation.images.image2": torch.from_numpy(
                np.ascontiguousarray(wrist.transpose(2, 0, 1))
            ).unsqueeze(0).to(dtype=torch.float32).div(255.0),
            "observation.state": torch.from_numpy(
                np.ascontiguousarray(np.asarray(observation.proprioception, dtype=np.float32))
            ).unsqueeze(0),
            "task": [observation.instruction],
        }
        telemetry_start = len(self._policy.action_telemetry)
        started_at = time.perf_counter()
        action = self._policy.select_action(batch)
        elapsed = time.perf_counter() - started_at
        releases = self._policy.action_telemetry[telemetry_start:]
        if any(record.get("model_invoked") is True for record in releases):
            self._model_inference_time_s += elapsed
        return np.asarray(action.detach().cpu().numpy()[0], dtype=np.float32)

    def close(self) -> None:
        del self._policy


def _local_policy_factory(
    *, device: str
) -> Callable[[dict[str, Any], dict[str, Any]], PilotPolicy]:
    """Build policies lazily, only after the user has explicitly chosen --execute."""

    def factory(condition: dict[str, Any], config: dict[str, Any]) -> PilotPolicy:
        import torch

        from lerobot_policy_smolvla_adaptive.configuration_smolvla_adaptive import (
            SmolVLAAdaptiveConfig,
        )
        from lerobot_policy_smolvla_adaptive.modeling_smolvla_adaptive import (
            SmolVLAAdaptivePolicy,
        )

        model = config["model"]
        policy_config = SmolVLAAdaptiveConfig(
            base_checkpoint=model["checkpoint"],
            base_revision=model["base_revision"],
            base_snapshot_path=model["base_snapshot_path"],
            vlm_checkpoint=model["vlm_checkpoint"],
            vlm_revision=model["vlm_revision"],
            vlm_snapshot_path=model["vlm_snapshot_path"],
            local_files_only=True,
            fixed_h=condition["fixed_h"],
            safety_enabled=True,
            replan_after_safety_violation=condition["replan_after_safety_violation"],
            num_steps=2,
            chunk_size=50,
            precision="fp16",
            device=device,
        )
        policy = SmolVLAAdaptivePolicy(policy_config)
        policy.to(device)
        policy.eval()
        return _LocalSmolVLAPilotPolicy(policy, torch)

    return factory


def _local_backend_factory(
    *, dataset_directory: Path, initial_state_source: str
) -> Callable[[], object]:
    def factory() -> object:
        from libero_platform.backends.libero_backend import LiberoBackend

        return LiberoBackend(
            dataset_directory=dataset_directory,
            initial_state_source=initial_state_source,
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or materialize a paired LIBERO Spatial pilot")
    parser.add_argument("--dry-run", action="store_true", help="explicitly materialize only; this is the default")
    parser.add_argument("--execute", action="store_true", help="required before the executor may call LIBERO env.step")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "evaluation" / "libero_spatial_paired_pilot.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-snapshot-path", required=True)
    parser.add_argument("--vlm-snapshot-path", required=True)
    parser.add_argument("--dataset-directory", type=Path, default=Path("."))
    parser.add_argument("--initial-state-source", choices=("benchmark", "demonstration"), default="benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-id", type=int, action="append")
    parser.add_argument("--episodes-per-task", type=int)
    args = parser.parse_args()
    if args.dry_run and args.execute:
        raise SystemExit("--dry-run and --execute are mutually exclusive")
    if not args.execute:
        result = materialize_dry_run(
            config_path=args.config,
            output_dir=args.output_dir,
            base_snapshot_path=args.base_snapshot_path,
            vlm_snapshot_path=args.vlm_snapshot_path,
        )
    else:
        # Validate the immutable plan before checking execution-only settings:
        # a missing manifest must never progress as far as LIBERO construction.
        resolved, _ = _read_execution_inputs(args.output_dir.resolve())
        if _snapshot_path(
            args.base_snapshot_path, SMOLVLA_REVISION, "base_snapshot_path"
        ) != resolved["model"]["base_snapshot_path"]:
            raise SystemExit("--base-snapshot-path must match resolved_config.json")
        if _snapshot_path(
            args.vlm_snapshot_path, SMOLVLM2_REVISION, "vlm_snapshot_path"
        ) != resolved["model"]["vlm_snapshot_path"]:
            raise SystemExit("--vlm-snapshot-path must match resolved_config.json")
        # Do not re-materialize here. --execute has one permitted source of
        # trial identity: the manifest already present in output_dir.
        result = execute_pilot(
            output_dir=args.output_dir,
            backend_factory=_local_backend_factory(
                dataset_directory=args.dataset_directory,
                initial_state_source=args.initial_state_source,
            ),
            policy_factory=_local_policy_factory(device=args.device),
            task_ids=set(args.task_id) if args.task_id else None,
            episodes_per_task=args.episodes_per_task,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
