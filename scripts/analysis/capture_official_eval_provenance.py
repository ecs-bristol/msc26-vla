"""Write reproducibility artifacts for an official LeRobot evaluation invocation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _command_output(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _build_manifest(args: argparse.Namespace) -> dict[str, object]:
    if args.suite != "libero_spatial":
        raise ValueError("paired manifest generation currently supports libero_spatial")
    trials = [
        {
            "task_id": task_id,
            "episode_index": episode_index,
            "seed": args.seed + episode_index,
            "initial_state_id": None,
            "initial_state_provenance": "LeRobot LIBERO reset(seed); state ID is not exposed by v0.6.1",
        }
        for task_id in range(10)
        for episode_index in range(args.episodes_per_task)
    ]
    return {
        "schema_version": 1,
        "suite": args.suite,
        "episodes_per_task": args.episodes_per_task,
        "seed_strategy": "Per task, LeRobot v0.6.1 resets episode_index i with seed + i.",
        "trials": trials,
    }


def _materialize_manifest(args: argparse.Namespace) -> tuple[Path, Path, str]:
    """Create or verify the pre-evaluation paired trial manifest.

    A caller may supply a stable path shared by Fixed-H and Adaptive runs.  We
    always copy the verified bytes into the current output directory so each
    result remains self-contained and auditable.
    """
    expected = _build_manifest(args)
    manifest_path = args.manifest_path or args.output_dir / "paired_seed_manifest.json"
    manifest_path = manifest_path.expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                "existing paired manifest does not match suite, seed, or episodes-per-task"
            )
    else:
        manifest_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")

    output_manifest_path = args.output_dir / "paired_seed_manifest.json"
    if manifest_path.resolve() != output_manifest_path.resolve():
        shutil.copyfile(manifest_path, output_manifest_path)

    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return manifest_path, output_manifest_path, digest


def _write_launcher_resolved_config(
    args: argparse.Namespace, manifest_path: Path, manifest_sha256: str
) -> None:
    """Record launcher-resolved inputs when the evaluator process is exec'd."""
    if not args.write_launcher_resolved_config:
        return
    payload = {
        "source": "launcher-resolved inputs before lerobot-eval exec",
        "suite": args.suite,
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.checkpoint_revision,
        "num_steps": args.num_steps,
        "episode_length": args.episode_length,
        "paired_seed_manifest_path": str(manifest_path.resolve()),
        "paired_seed_manifest_sha256": manifest_sha256,
    }
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_summary_if_available(output_dir: Path) -> None:
    """Produce a derived CSV only from the evaluator's existing eval_info.json."""
    info_path = output_dir / "eval_info.json"
    if not info_path.is_file():
        return
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for task in payload.get("per_task", []):
        metrics = task.get("metrics", {})
        successes = metrics.get("successes", [])
        rows.append(
            {
                "task_group": task.get("task_group"),
                "task_id": task.get("task_id"),
                "episodes": len(successes),
                "successes": sum(bool(value) for value in successes),
                "success_rate": (
                    sum(bool(value) for value in successes) / len(successes)
                    if successes
                    else None
                ),
            }
        )
    if not rows:
        return
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--episode-length", type=int, required=True)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--write-launcher-resolved-config", action="store_true")
    args = parser.parse_args()

    if args.episodes_per_task < 1 or args.seed < 0:
        raise ValueError("episodes and seed must be positive/non-negative")
    if args.num_steps is not None and args.num_steps < 1:
        raise ValueError("num_steps must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, output_manifest_path, manifest_sha256 = _materialize_manifest(args)
    _write_launcher_resolved_config(args, manifest_path, manifest_sha256)
    metadata = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "project_git_commit": _command_output(["git", "rev-parse", "HEAD"], args.project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "lerobot_version": _package_version("lerobot"),
        "libero_version": _package_version("libero"),
        "torch_version": _package_version("torch"),
        "cuda_driver": _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            args.project_root,
        ),
        "gpu": _command_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            args.project_root,
        ),
        "mujoco_gl": __import__("os").environ.get("MUJOCO_GL"),
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.checkpoint_revision,
        "suite": args.suite,
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "episode_length": args.episode_length,
        "exit_code": args.exit_code,
        "paired_seed_manifest_path": str(manifest_path.resolve()),
        "paired_seed_manifest_output_path": str(output_manifest_path.resolve()),
        "paired_seed_manifest_sha256": manifest_sha256,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary_if_available(args.output_dir)


if __name__ == "__main__":
    main()
