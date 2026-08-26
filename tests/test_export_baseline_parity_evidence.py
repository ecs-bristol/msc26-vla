from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess

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
    summary.write_bytes(b"condition,task_id\r\nStatic-H1-original,0\r\n")
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
    assert (output / "paired_summary.csv").read_bytes() == (
        b"condition,task_id\nStatic-H1-original,0\n"
    )
    assert len(manifest["files"]) == 13
    summary_evidence = manifest["files"]["paired_summary.csv"]
    assert summary_evidence["source_sha256"] == hashlib.sha256(summary.read_bytes()).hexdigest()
    assert summary_evidence["committed_export_sha256"] == hashlib.sha256(
        (output / "paired_summary.csv").read_bytes()
    ).hexdigest()
    assert summary_evidence["source_sha256"] != summary_evidence["committed_export_sha256"]
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


def test_committed_evidence_sha256s_pass_in_clean_checkout_bytes() -> None:
    bundle = PROJECT_ROOT / "evidence" / "parity_hardening" / "baseline_parity_export"
    checksum_lines = (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert checksum_lines
    for line in checksum_lines:
        expected, relative_path = line.split("  ", 1)
        data = (bundle / relative_path).read_bytes()
        assert b"\r\n" not in data
        assert hashlib.sha256(data).hexdigest() == expected

    summary_hash = next(
        line.split("  ", 1)[0]
        for line in checksum_lines
        if line.endswith("  paired_summary.csv")
    )
    assert summary_hash == "5372dab2cbf5538035632a536dbf76674c47c572ec565b0e41e11a790e78a791"

    attribute = subprocess.check_output(
        [
            "git",
            "check-attr",
            "eol",
            "--",
            "evidence/parity_hardening/baseline_parity_export/paired_summary.csv",
        ],
        cwd=PROJECT_ROOT,
        text=True,
    )
    assert attribute.rstrip().endswith("eol: lf")
