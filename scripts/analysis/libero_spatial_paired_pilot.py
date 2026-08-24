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
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    "termination_reason",
    "git_sha",
    "resolved_config_path",
)


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


def _condition_slug(name: str) -> str:
    return name.replace("→", "-to-").lower()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    }
    _write_json(output_dir / "provenance.json", provenance)

    rows: list[dict[str, object]] = []
    for condition in config["conditions"]:
        for trial in trials:
            episode_path = (
                output_dir
                / "episodes"
                / _condition_slug(condition["name"])
                / f"task_{trial['task_id']:02d}_seed_{trial['seed']}_state_{trial['initial_state_id']}.json"
            )
            result = {
                "schema_version": 1,
                "status": "planned_dry_run",
                "condition": condition["name"],
                "condition_config": condition,
                **trial,
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
                "termination_reason": "not_started_dry_run",
                "git_sha": sha,
                "resolved_config_path": str(resolved_path.resolve()),
            }
            _write_json(episode_path, result)
            rows.append({field: result[field] for field in SUMMARY_FIELDS})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
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
    (output_dir / "dry_run_command.txt").write_text(command + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir), "planned_episodes": len(rows), "command": command}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a paired LIBERO Spatial pilot dry run")
    parser.add_argument("--dry-run", action="store_true", help="required: rollout execution is not implemented")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "evaluation" / "libero_spatial_paired_pilot.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-snapshot-path", required=True)
    parser.add_argument("--vlm-snapshot-path", required=True)
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("Refusing rollout: this pilot tool currently supports --dry-run only")
    result = materialize_dry_run(
        config_path=args.config,
        output_dir=args.output_dir,
        base_snapshot_path=args.base_snapshot_path,
        vlm_snapshot_path=args.vlm_snapshot_path,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
