"""Materialize an offline, paired LIBERO Spatial pilot plan.

This command is intentionally dry-run only.  It never imports LeRobot, opens
LIBERO, or instantiates a model; its output is the immutable input and result
layout that a later, separately-authorized rollout runner must consume.
"""

from __future__ import annotations

import argparse
import csv
import gc
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
ALLOWED_HORIZONS = frozenset({1, 5, 10, 20, 25, 30, 50})
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
    "realized_actions_per_call",
    "generated_actions",
    "unused_actions",
    "chunk_utilization",
    "horizon_tail_discarded_actions",
    "trigger_tail_discarded_actions",
    "terminal_tail_unused_actions",
    "range_violations",
    "range_violation_dimension_counts",
    "range_violation_max_excess_by_dimension",
    "trigger_range_violations",
    "gripper_only_range_violations",
    "range_clips",
    "buffer_discards",
    "mean_actual_horizon",
    "action_trace_sha256",
    "termination_reason",
    "git_sha",
    "resolved_config_path",
)
_TERMINAL_EPISODE_STATUSES = frozenset({"completed", "failed"})
_LEGACY_INFERENCE_SEED_NAMESPACE = "libero_spatial"
_ADAPTIVE_V2_HELD_OUT_NAMESPACE = "adaptive-v2-heldout-v1|libero_spatial"
_ADAPTIVE_V2_CONFIRMATORY_NAMESPACE = "adaptive-v2-confirmatory-v1|libero_spatial"


class PilotPolicy(Protocol):
    @property
    def telemetry(self) -> tuple[dict[str, object], ...]: ...

    @property
    def model_inference_time_s(self) -> float: ...

    def reset(self) -> None: ...

    def finalize_episode(self) -> None: ...

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
    formal_confirmatory = config.get("formal_adaptive_v2_confirmatory", False)
    if type(formal_confirmatory) is not bool:
        raise ValueError("formal_adaptive_v2_confirmatory must be a boolean")
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
    if not isinstance(conditions, list):
        raise ValueError("pilot config conditions must be a list")
    if formal_confirmatory:
        expected_conditions = [
            {
                "name": "Static-H20",
                "fixed_h": 20,
                "safety_enabled": True,
                "replan_after_safety_violation": False,
                "adaptive_v2_trigger": False,
                "clip_actions": False,
            },
            {
                "name": "Adaptive-v2a-H20→H1",
                "fixed_h": 20,
                "safety_enabled": True,
                "replan_after_safety_violation": False,
                "adaptive_v2_trigger": True,
                "clip_actions": False,
            },
        ]
        if conditions != expected_conditions:
            raise ValueError(
                "formal Adaptive-v2a evaluation must contain only matched "
                "Static-H20 and Adaptive-v2a conditions"
            )
        if config.get("development_trigger_coverage", False) is not False:
            raise ValueError("formal evaluation cannot be marked development coverage")
        if config.get("inference_seed_namespace") != _ADAPTIVE_V2_CONFIRMATORY_NAMESPACE:
            raise ValueError("formal Adaptive-v2a evaluation requires confirmatory seeds")
        return
    if len(conditions) != 6:
        raise ValueError("pilot config must declare the six paired conditions")
    names = [condition.get("name") for condition in conditions if isinstance(condition, dict)]
    static_names = [
        "Static-H1-original",
        "Static-H5",
        "Static-H10",
        "Static-H20",
        "Static-H50",
    ]
    legacy_names = [*static_names, "Adaptive-H20→H1"]
    v2_names = [*static_names, "Adaptive-v2a-H20→H1"]
    if names not in (legacy_names, v2_names):
        raise ValueError("pilot conditions must include native-equivalent Static-H1-original")
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("each condition must be a mapping")
        if type(condition.get("fixed_h")) is not int or condition["fixed_h"] not in ALLOWED_HORIZONS:
            raise ValueError("condition fixed_h must be an allowed execution horizon")
        if condition.get("clip_actions") is not False:
            raise ValueError("paired-pilot conditions use native actions without clipping")
    if any(condition.get("safety_enabled") is not False for condition in conditions[:-1]):
        raise ValueError("static paired-pilot conditions must be detection-only telemetry")
    adaptive = conditions[-1]
    if any(
        condition.get("replan_after_safety_violation") is not False
        for condition in conditions[:-1]
    ):
        raise ValueError("static paired-pilot conditions must not safety-trigger replanning")
    if any(
        condition.get("adaptive_v2_trigger", False) is not False
        for condition in conditions[:-1]
    ):
        raise ValueError("static paired-pilot conditions must not use Adaptive-v2a")
    if names == legacy_names:
        if adaptive["fixed_h"] != 20 or adaptive.get("replan_after_safety_violation") is not True:
            raise ValueError("Adaptive-H20→H1 must replan after a safety violation")
        if adaptive.get("safety_enabled") is not True:
            raise ValueError("Adaptive-H20→H1 requires range-violation detection")
        if adaptive.get("adaptive_v2_trigger", False) is not False:
            raise ValueError("the frozen v1 condition cannot enable Adaptive-v2a")
    else:
        development_coverage = config.get("development_trigger_coverage", False)
        if type(development_coverage) is not bool:
            raise ValueError("development_trigger_coverage must be a boolean")
        expected_namespace = (
            _LEGACY_INFERENCE_SEED_NAMESPACE
            if development_coverage
            else _ADAPTIVE_V2_HELD_OUT_NAMESPACE
        )
        if config.get("inference_seed_namespace") != expected_namespace:
            raise ValueError(
                "Adaptive-v2a inference seed namespace does not match its "
                "held-out/development role"
            )
        if adaptive != {
            "name": "Adaptive-v2a-H20→H1",
            "fixed_h": 20,
            "safety_enabled": True,
            "replan_after_safety_violation": False,
            "adaptive_v2_trigger": True,
            "clip_actions": False,
        }:
            raise ValueError("Adaptive-v2a must change only the preregistered trigger logic")


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


def _inference_seed(
    trial: dict[str, int], namespace: str = _LEGACY_INFERENCE_SEED_NAMESPACE
) -> int:
    """Derive a condition-independent reproducible inference RNG seed."""

    pairing_key = "|".join(
        str(value)
        for value in (
            namespace,
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


def _seed_provenance(
    trials: list[dict[str, int]], namespace: str = _LEGACY_INFERENCE_SEED_NAMESPACE
) -> list[dict[str, int]]:
    return [
        {
            "task_id": trial["task_id"],
            "initial_state_id": trial["initial_state_id"],
            "environment_seed": trial["seed"],
            "inference_seed": _inference_seed(trial, namespace),
        }
        for trial in trials
    ]


def _inference_seed_derivation(namespace: str) -> str:
    return (
        f"sha256({namespace}|task_id|seed|initial_state_id)[:8] "
        "& ((1<<63)-1)"
    )


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
    inference_seed_namespace = config.get(
        "inference_seed_namespace", _LEGACY_INFERENCE_SEED_NAMESPACE
    )
    manifest = {
        "schema_version": 1,
        "suite": config["suite"],
        "episodes_per_task": config["episodes_per_task"],
        "pairing_key": ["task_id", "seed", "initial_state_id"],
        "seed_strategy": "seed = configured seed + initial_state_id; every condition reuses this exact manifest",
        "trials": trials,
    }
    if "inference_seed_namespace" in config:
        manifest["inference_seed_namespace"] = inference_seed_namespace
        manifest["inference_seed_records"] = _seed_provenance(
            trials, inference_seed_namespace
        )
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
        "inference_seed_namespace": inference_seed_namespace,
        "inference_seed_derivation": _inference_seed_derivation(
            inference_seed_namespace
        ),
        "pairing_seeds": _seed_provenance(trials, inference_seed_namespace),
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
                "inference_seed": _inference_seed(trial, inference_seed_namespace),
                "success_at_280": None,
                "success_step": None,
                "executed_env_steps": None,
                "wall_time_to_terminal_s": None,
                "model_invocations": None,
                "model_inference_time_s": None,
                "realized_actions_per_call": None,
                "generated_actions": None,
                "unused_actions": None,
                "chunk_utilization": None,
                "horizon_tail_discarded_actions": None,
                "trigger_tail_discarded_actions": None,
                "terminal_tail_unused_actions": None,
                "range_violations": None,
                "range_violation_dimension_counts": None,
                "range_violation_max_excess_by_dimension": None,
                "trigger_range_violations": None,
                "gripper_only_range_violations": None,
                "range_clips": None,
                "buffer_discards": None,
                "mean_actual_horizon": None,
                "adaptive_v2_trigger_events": None,
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
        writer.writerows(
            {
                field: (
                    json.dumps(value, separators=(",", ":"), sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for field, value in row.items()
            }
            for row in rows
        )
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
    inference_seed_namespace = config.get(
        "inference_seed_namespace", _LEGACY_INFERENCE_SEED_NAMESPACE
    )
    if "inference_seed_namespace" in config:
        if manifest.get("inference_seed_namespace") != inference_seed_namespace:
            raise ValueError("paired manifest inference seed namespace mismatch")
        if manifest.get("inference_seed_records") != _seed_provenance(
            _trials(config), inference_seed_namespace
        ):
            raise ValueError("paired manifest held-out inference seeds do not match config")
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


def _load_pairing_key_filter(path: Path) -> set[tuple[int, int, int]]:
    """Read an explicit, outcome-free execution subset from a committed artifact."""

    payload = _read_json(path.resolve())
    if payload.get("selection_role") not in {
        "trigger_coverage_development",
        "formal_heldout_block",
    }:
        raise ValueError("pairing-key filter has an unsupported selection role")
    raw_keys = payload.get("pairing_keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ValueError("pairing-key filter must contain a non-empty pairing_keys list")
    required = {"task_id", "seed", "initial_state_id"}
    forbidden = {"success_at_280", "success_step", "termination_reason"}
    selected: set[tuple[int, int, int]] = set()
    for item in raw_keys:
        if not isinstance(item, dict) or set(item) != required:
            extras = set(item) - required if isinstance(item, dict) else set()
            if extras & forbidden:
                raise ValueError("outcome fields are forbidden in a pairing-key filter")
            raise ValueError("each pairing-key filter item must contain exactly the key fields")
        values = tuple(item[field] for field in ("task_id", "seed", "initial_state_id"))
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("pairing-key filter values must be non-negative integers")
        selected.add(values)
    if len(selected) != len(raw_keys):
        raise ValueError("pairing-key filter contains duplicate keys")
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


def _telemetry_metrics(
    records: tuple[dict[str, object], ...], *, executed_env_steps: int
) -> dict[str, object]:
    refills = [record for record in records if record.get("event") == "refill"]
    releases = [record for record in records if record.get("event") == "action_release"]
    finalizations = {
        int(record["model_invocation"]): record
        for record in records
        if record.get("event") == "call_finalized"
    }
    if len(releases) != executed_env_steps:
        raise RuntimeError(
            "released action count must equal executed_env_steps for utilization accounting"
        )

    releases_by_call: dict[int, list[dict[str, object]]] = {}
    for record in releases:
        releases_by_call.setdefault(int(record["model_invocation"]), []).append(record)

    realized_actions_per_call: list[int] = []
    horizon_tail = 0
    trigger_tail = 0
    terminal_tail = 0
    for refill in refills:
        invocation = int(refill["model_invocation"])
        planned_horizon = int(refill["planned_horizon"])
        realized = len(releases_by_call.pop(invocation, []))
        if realized > planned_horizon:
            raise RuntimeError("realized actions cannot exceed planned_horizon")
        finalization = finalizations.get(invocation)
        if finalization is not None:
            if int(finalization["actual_horizon"]) != realized:
                raise RuntimeError("call finalization actual_horizon does not match releases")
            reason = str(finalization["finalization_reason"])
        else:
            reason = "terminal"
        realized_actions_per_call.append(realized)
        horizon_tail += 50 - planned_horizon
        planned_remainder = planned_horizon - realized
        if reason == "trigger":
            trigger_tail += planned_remainder
        elif reason == "terminal":
            terminal_tail += planned_remainder
        elif reason != "horizon" or planned_remainder != 0:
            raise RuntimeError("invalid action-call finalization telemetry")
    if releases_by_call:
        raise RuntimeError("action release exists without a matching refill")

    model_invocations = len(refills)
    generated_actions = 50 * model_invocations
    unused_actions = generated_actions - executed_env_steps
    if unused_actions < 0:
        raise RuntimeError("executed actions cannot exceed generated actions")
    if horizon_tail + trigger_tail + terminal_tail != unused_actions:
        raise RuntimeError("unused action categories do not sum to generated minus executed")

    dimension_counts = {str(index): 0 for index in range(7)}
    max_excess = {str(index): 0.0 for index in range(7)}
    for release in releases:
        dimensions = list(release.get("range_violation_dimensions", []))
        excesses = list(release.get("range_violation_excess", []))
        if len(dimensions) != len(excesses):
            raise RuntimeError("range violation dimensions and excess telemetry must align")
        for dimension, excess in zip(dimensions, excesses, strict=True):
            key = str(int(dimension))
            if key not in dimension_counts:
                raise RuntimeError("range violation dimension is outside the LIBERO action")
            dimension_counts[key] += 1
            max_excess[key] = max(max_excess[key], float(excess))

    return {
        "model_invocations": model_invocations,
        "realized_actions_per_call": realized_actions_per_call,
        "generated_actions": generated_actions,
        "unused_actions": unused_actions,
        "chunk_utilization": (
            executed_env_steps / generated_actions if generated_actions else None
        ),
        "horizon_tail_discarded_actions": horizon_tail,
        "trigger_tail_discarded_actions": trigger_tail,
        "terminal_tail_unused_actions": terminal_tail,
        "range_violations": sum(bool(record.get("range_violation")) for record in releases),
        "range_violation_dimension_counts": dimension_counts,
        "range_violation_max_excess_by_dimension": max_excess,
        "trigger_range_violations": sum(
            bool(record.get("trigger_range_violation")) for record in releases
        ),
        "gripper_only_range_violations": sum(
            bool(record.get("gripper_only_range_violation")) for record in releases
        ),
        "range_clips": sum(bool(record.get("range_clipped")) for record in releases),
        "buffer_discards": sum(bool(record.get("buffer_discarded")) for record in releases),
        "mean_actual_horizon": (
            sum(realized_actions_per_call) / len(realized_actions_per_call)
            if realized_actions_per_call
            else None
        ),
    }


def _model_invocation_count(records: tuple[dict[str, object], ...]) -> int:
    return sum(record.get("event") == "refill" for record in records)


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


def _adaptive_v2_trigger_event(
    release: dict[str, object],
    *,
    trial: dict[str, int],
    inference_seed: int,
    env_step: int,
) -> dict[str, object]:
    """Build a complete, per-event preregistered trigger artifact."""

    if release.get("adaptive_v2_triggered") is not True:
        raise ValueError("Adaptive-v2a event builder requires a triggered release")
    dimensions = [int(value) for value in release["trigger_violation_dimensions"]]
    raw_values = [float(value) for value in release["adaptive_v2_trigger_raw_values"]]
    bounds = [float(value) for value in release["adaptive_v2_trigger_bounds"]]
    excesses = [float(value) for value in release["adaptive_v2_trigger_excess"]]
    severities = [float(value) for value in release["adaptive_v2_trigger_severity"]]
    persistence = [
        int(value) for value in release["adaptive_v2_trigger_persistence_counts"]
    ]
    if not dimensions or any(dimension == 6 for dimension in dimensions):
        raise RuntimeError("Adaptive-v2a trigger dimensions must be non-gripper")
    lengths = {
        len(dimensions),
        len(raw_values),
        len(bounds),
        len(excesses),
        len(severities),
        len(persistence),
    }
    if len(lengths) != 1:
        raise RuntimeError("Adaptive-v2a per-dimension trigger telemetry is incomplete")
    violations = [
        {
            "dimension": dimension,
            "raw_value": raw_value,
            "bound": bound,
            "excess": excess,
            "severity": severity,
            "persistence_count": count,
        }
        for dimension, raw_value, bound, excess, severity, count in zip(
            dimensions,
            raw_values,
            bounds,
            excesses,
            severities,
            persistence,
            strict=True,
        )
    ]
    event = {
        "schema_version": 1,
        "task_id": trial["task_id"],
        "initial_state_id": trial["initial_state_id"],
        "seed": trial["seed"],
        "environment_seed": trial["seed"],
        "inference_seed": inference_seed,
        "model_call_index": int(release["model_invocation"]),
        "env_step": env_step,
        "evaluation_order": "trigger_evaluated_before_env_step",
        "dimensions": dimensions,
        "raw_values": raw_values,
        "bounds": bounds,
        "excess": excesses,
        "severity": severities,
        "persistence_counts": persistence,
        "violations": violations,
        "discarded_actions": int(
            release["adaptive_v2_trigger_tail_discarded_actions"]
        ),
        "buffer_entries_cleared": int(
            release["adaptive_v2_buffer_entries_cleared"]
        ),
        "horizon_before": int(release["adaptive_v2_horizon_before"]),
        "horizon_after": int(release["adaptive_v2_horizon_after"]),
        "recovery_horizon": int(release["adaptive_v2_recovery_horizon"]),
        "state_before": str(release["adaptive_v2_state_before"]),
        "state_after": str(release["adaptive_v2_state_after"]),
        "immediate_dimensions": [
            int(value) for value in release["adaptive_v2_immediate_dimensions"]
        ],
        "persistent_dimensions": [
            int(value) for value in release["adaptive_v2_persistent_dimensions"]
        ],
        "cooldown_h20_actions": int(release["adaptive_v2_cooldown_h20_actions"]),
    }
    required = {
        "task_id", "initial_state_id", "seed", "model_call_index", "env_step",
        "dimensions", "raw_values", "bounds", "excess", "severity",
        "discarded_actions", "horizon_before", "horizon_after", "recovery_horizon",
    }
    if not required <= event.keys():
        raise RuntimeError("Adaptive-v2a trigger event schema is incomplete")
    return event


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

    inference_seed_namespace = config.get(
        "inference_seed_namespace", _LEGACY_INFERENCE_SEED_NAMESPACE
    )
    inference_seed = _inference_seed(trial, inference_seed_namespace)
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
    adaptive_v2_trigger_events: list[dict[str, object]] = []
    termination_reason = "max_steps"
    status = "completed"
    try:
        observation = episode.reset()
        policy.reset()
        telemetry_snapshot = _episode_telemetry_snapshot(policy)
        for step_id in range(1, config["episode_cap"] + 1):
            telemetry_before_action = len(policy.telemetry)
            action = np.asarray(policy.select_action(observation), dtype=np.float32)
            if action.shape != (7,):
                raise ValueError(f"policy action must have shape (7,), got {action.shape}")
            action_records = tuple(policy.telemetry)[telemetry_before_action:]
            if condition.get("adaptive_v2_trigger", False):
                releases = [
                    record for record in action_records
                    if record.get("event") == "action_release"
                ]
                if (
                    len(releases) != 1
                    or releases[0].get("adaptive_v2_enabled") is not True
                ):
                    raise RuntimeError(
                        "Adaptive-v2a must persist exactly one pre-step release record"
                    )
                if releases[0].get("adaptive_v2_triggered") is True:
                    adaptive_v2_trigger_events.append(
                        _adaptive_v2_trigger_event(
                            releases[0],
                            trial=trial,
                            inference_seed=inference_seed,
                            env_step=step_id,
                        )
                    )
            action_trace.append(action.copy())
            steps = step_id
            # Adaptive-v2a trigger evaluation and event construction above are
            # deliberately complete before this environment mutation boundary.
            step = episode.step(action)
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
        try:
            finalize_episode = getattr(policy, "finalize_episode", None)
            if callable(finalize_episode):
                finalize_episode()
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
    metrics = _telemetry_metrics(episode_records, executed_env_steps=steps)
    if metrics["model_invocations"] != model_invocations:
        raise RuntimeError("model invocation delta does not match episode refill telemetry")
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
        "adaptive_v2_trigger_events": adaptive_v2_trigger_events,
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
    condition_names: set[str] | None = None,
    pairing_keys: set[tuple[int, int, int]] | None = None,
    pairing_key_filter_path: Path | None = None,
    inference_seed_setter: Callable[[int], None] = _set_inference_seed,
) -> dict[str, int]:
    """Execute only planned pairs, atomically persisting each terminal result.

    Existing terminal episode files are never overwritten, so a resumed process
    only opens missing or non-terminal planned episodes.
    """

    output_dir = output_dir.resolve()
    config, trials = _read_execution_inputs(output_dir)
    if (pairing_keys is None) != (pairing_key_filter_path is None):
        raise ValueError("pairing keys and their immutable filter path must be provided together")
    filter_role = None
    if pairing_key_filter_path is not None:
        filter_role = _read_json(pairing_key_filter_path.resolve()).get("selection_role")
        if filter_role == "trigger_coverage_development":
            if config.get("development_trigger_coverage") is not True:
                raise ValueError("development filter requires development trigger coverage")
        elif filter_role == "formal_heldout_block":
            if config.get("formal_adaptive_v2_confirmatory") is not True:
                raise ValueError("formal block filter requires formal confirmatory config")
        else:
            raise ValueError("pairing-key filter role is invalid")
    selected_trials = _selected_trials(trials, task_ids, episodes_per_task)
    if pairing_keys is not None:
        selected_trials = [
            trial
            for trial in selected_trials
            if (trial["task_id"], trial["seed"], trial["initial_state_id"])
            in pairing_keys
        ]
        observed = {
            (trial["task_id"], trial["seed"], trial["initial_state_id"])
            for trial in selected_trials
        }
        if observed != pairing_keys:
            missing = sorted(pairing_keys - observed)
            raise ValueError(f"pairing-key filter is not a subset of the manifest: {missing}")
    available_conditions = {condition["name"] for condition in config["conditions"]}
    selected_condition_names = (
        available_conditions if condition_names is None else condition_names
    )
    if not selected_condition_names or not selected_condition_names <= available_conditions:
        unknown = sorted(selected_condition_names - available_conditions)
        raise ValueError(f"unknown or empty condition selection: {unknown}")
    if filter_role == "trigger_coverage_development" and selected_condition_names != {"Adaptive-v2a-H20→H1"}:
        raise ValueError("development pairing-key filters may execute only Adaptive-v2a")
    if filter_role == "formal_heldout_block" and len(selected_condition_names) != 1:
        raise ValueError("each formal held-out phase must execute exactly one condition")
    selected_conditions = [
        condition for condition in config["conditions"]
        if condition["name"] in selected_condition_names
    ]
    manifest_path = output_dir / "paired_manifest.json"
    inference_seed_namespace = config.get(
        "inference_seed_namespace", _LEGACY_INFERENCE_SEED_NAMESPACE
    )
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
            "selected_conditions": [condition["name"] for condition in selected_conditions],
            "pairing_key_filter_path": (
                str(pairing_key_filter_path.resolve())
                if pairing_key_filter_path is not None
                else None
            ),
            "pairing_key_filter_sha256": (
                hashlib.sha256(pairing_key_filter_path.read_bytes()).hexdigest()
                if pairing_key_filter_path is not None
                else None
            ),
            "pairing_key_filter_role": filter_role,
            "local_files_only": config["model"]["local_files_only"],
            "base_snapshot_path": config["model"]["base_snapshot_path"],
            "vlm_snapshot_path": config["model"]["vlm_snapshot_path"],
            "environment_seed_source": "paired_manifest.trials[].seed",
            "inference_seed_namespace": inference_seed_namespace,
            "inference_seed_derivation": _inference_seed_derivation(
                inference_seed_namespace
            ),
            "pairing_seeds": _seed_provenance(
                selected_trials, inference_seed_namespace
            ),
        },
    )
    backend = backend_factory()
    executed = 0
    skipped = 0
    try:
        for condition in selected_conditions:
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
        policy = self._policy
        if policy is None:
            return
        try:
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
        finally:
            self._policy = None
            del policy
            gc.collect()
            cuda = getattr(self._torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                cuda.synchronize()
                cuda.empty_cache()


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
            safety_enabled=condition["safety_enabled"],
            replan_after_safety_violation=condition["replan_after_safety_violation"],
            adaptive_v2_trigger=condition.get("adaptive_v2_trigger", False),
            clip_actions=condition["clip_actions"],
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


def _local_backend_factory() -> Callable[[], object]:
    def factory() -> object:
        from libero_platform.backends.libero_backend import OfficialLeRobotLiberoBackend

        return OfficialLeRobotLiberoBackend()

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
    parser.add_argument("--condition", action="append")
    parser.add_argument(
        "--pairing-key-file",
        type=Path,
        help="outcome-free development subset; execution only",
    )
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
        if args.initial_state_source != "benchmark":
            raise SystemExit("official paired-pilot backend requires benchmark initial states")
        pairing_keys = (
            _load_pairing_key_filter(args.pairing_key_file)
            if args.pairing_key_file is not None
            else None
        )
        # Do not re-materialize here. --execute has one permitted source of
        # trial identity: the manifest already present in output_dir.
        result = execute_pilot(
            output_dir=args.output_dir,
            backend_factory=_local_backend_factory(),
            policy_factory=_local_policy_factory(device=args.device),
            task_ids=set(args.task_id) if args.task_id else None,
            episodes_per_task=args.episodes_per_task,
            condition_names=set(args.condition) if args.condition else None,
            pairing_keys=pairing_keys,
            pairing_key_filter_path=args.pairing_key_file,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
