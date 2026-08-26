from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = PROJECT_ROOT / "scripts" / "analysis" / "export_baseline_parity_evidence.py"
    spec = importlib.util.spec_from_file_location("export_parity_evidence_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_bundle_selects_only_ten_completed_h1_episodes(tmp_path: Path) -> None:
    official = tmp_path / "eval_info.json"
    summary = tmp_path / "summary.csv"
    parity = tmp_path / "parity_report.json"
    episodes = tmp_path / "episodes"
    output = tmp_path / "export"
    episodes.mkdir()
    official.write_text('{"success": 10}\n', encoding="utf-8")
    summary.write_text("condition,task_id\nStatic-H1-original,0\n", encoding="utf-8")
    parity.write_text('{"action_parity": "PASS"}\n', encoding="utf-8")
    for task_id in range(10):
        payload = {
            "condition": "Static-H1-original",
            "status": "completed",
            "task_id": task_id,
            "action_trace_sha256": f"hash-{task_id}",
        }
        (episodes / f"task_{task_id:02d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    (episodes / "dry_run.json").write_text(
        json.dumps({
            "condition": "Static-H1-original",
            "status": "planned",
            "task_id": 0,
        }),
        encoding="utf-8",
    )

    manifest = _module().export_bundle(
        official_eval_info=official,
        paired_summary=summary,
        paired_episodes_dir=episodes,
        parity_report=parity,
        output_dir=output,
    )

    assert len(list((output / "episodes").glob("*.json"))) == 10
    assert (output / "paired_summary.csv").read_bytes() == summary.read_bytes()
    assert len(manifest["files"]) == 13
    assert (output / "SHA256SUMS").is_file()


def test_export_bundle_refuses_to_overwrite_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "export"
    output.mkdir()
    (output / "existing").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _module().export_bundle(
            official_eval_info=tmp_path / "missing-official",
            paired_summary=tmp_path / "missing-summary",
            paired_episodes_dir=tmp_path / "missing-episodes",
            parity_report=tmp_path / "missing-parity",
            output_dir=output,
        )
