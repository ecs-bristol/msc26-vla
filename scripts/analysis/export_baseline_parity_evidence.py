"""Export a small, checksum-addressed baseline-parity evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


EPISODE_FIELDS = (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_h1_episodes(episodes_dir: Path) -> list[tuple[Path, dict[str, object]]]:
    episodes: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(episodes_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("condition") != "Static-H1-original":
            continue
        if payload.get("status") != "completed":
            continue
        episodes.append((path, payload))
    episodes.sort(key=lambda item: int(item[1]["task_id"]))
    if [int(payload["task_id"]) for _, payload in episodes] != list(range(10)):
        raise ValueError("expected exactly one completed Static-H1-original episode for tasks 0..9")
    return episodes


def export_bundle(
    *,
    official_eval_info: Path,
    paired_summary: Path,
    paired_episodes_dir: Path,
    parity_report: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_output = output_dir / "episodes"
    episode_output.mkdir()

    shutil.copyfile(official_eval_info, output_dir / "official_eval_info.json")
    shutil.copyfile(paired_summary, output_dir / "paired_summary.csv")
    shutil.copyfile(parity_report, output_dir / "parity_report.json")

    source_hashes: dict[str, str] = {}
    for source, payload in _completed_h1_episodes(paired_episodes_dir):
        selected = {field: payload.get(field) for field in EPISODE_FIELDS}
        destination = episode_output / f"task_{int(payload['task_id']):02d}.json"
        destination.write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        source_hashes[str(source.resolve())] = _sha256(source)

    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    checksums = {
        path.relative_to(output_dir).as_posix(): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    }
    manifest = {
        "schema_version": 1,
        "sources": {
            "official_eval_info": str(official_eval_info.resolve()),
            "paired_summary": str(paired_summary.resolve()),
            "paired_episode_source_sha256": source_hashes,
            "parity_report": str(parity_report.resolve()),
        },
        "files": checksums,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_lines = [
        f"{metadata['sha256']}  {relative_path}"
        for relative_path, metadata in sorted(checksums.items())
    ]
    (output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-eval-info", type=Path, required=True)
    parser.add_argument("--paired-summary", type=Path, required=True)
    parser.add_argument("--paired-episodes-dir", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_bundle(
        official_eval_info=args.official_eval_info,
        paired_summary=args.paired_summary,
        paired_episodes_dir=args.paired_episodes_dir,
        parity_report=args.parity_report,
        output_dir=args.output_dir,
    )
    print(json.dumps({"files": len(manifest["files"]), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
