"""Export a small, checksum-addressed baseline-parity evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text_bytes(path: Path) -> bytes:
    text = path.read_bytes().decode("utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _export_text(source: Path, destination: Path) -> dict[str, object]:
    normalized = _normalized_text_bytes(source)
    destination.write_bytes(normalized)
    return {
        "source_path": str(source.resolve()),
        "source_sha256": _sha256(source),
        "committed_export_sha256": hashlib.sha256(normalized).hexdigest(),
        "size_bytes": len(normalized),
    }


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

    file_evidence = {
        "official_eval_info.json": _export_text(
            official_eval_info, output_dir / "official_eval_info.json"
        ),
        "paired_summary.csv": _export_text(
            paired_summary, output_dir / "paired_summary.csv"
        ),
        "parity_report.json": _export_text(
            parity_report, output_dir / "parity_report.json"
        ),
    }

    for source, payload in _completed_h1_episodes(paired_episodes_dir):
        selected = {field: payload.get(field) for field in EPISODE_FIELDS}
        relative_path = f"episodes/task_{int(payload['task_id']):02d}.json"
        destination = output_dir / relative_path
        normalized = (json.dumps(selected, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        destination.write_bytes(normalized)
        file_evidence[relative_path] = {
            "source_path": str(source.resolve()),
            "source_sha256": _sha256(source),
            "committed_export_sha256": hashlib.sha256(normalized).hexdigest(),
            "size_bytes": len(normalized),
        }

    manifest = {
        "schema_version": 2,
        "text_normalization": "UTF-8 with CRLF and CR normalized to LF before export hashing",
        "files": dict(sorted(file_evidence.items())),
    }
    (output_dir / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    checksum_lines = [
        f"{metadata['committed_export_sha256']}  {relative_path}"
        for relative_path, metadata in sorted(file_evidence.items())
    ]
    (output_dir / "SHA256SUMS").write_bytes(
        ("\n".join(checksum_lines) + "\n").encode("utf-8")
    )
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
