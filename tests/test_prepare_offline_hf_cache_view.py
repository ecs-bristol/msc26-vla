from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analysis.prepare_offline_hf_cache_view import prepare_cache_view


def _snapshot(tmp_path: Path, repo_name: str, revision: str) -> Path:
    repo = tmp_path / "source" / repo_name
    (repo / "blobs").mkdir(parents=True)
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    return snapshot


def test_cache_view_pins_main_to_existing_snapshot_without_copying_blobs(tmp_path: Path) -> None:
    repo_name = "models--Example--FrozenModel"
    revision = "0123456789abcdef"
    snapshot = _snapshot(tmp_path, repo_name, revision)

    manifest_path = prepare_cache_view(
        cache_root=tmp_path / "view",
        repo_cache_name=repo_name,
        snapshot=snapshot,
    )

    target_repo = tmp_path / "view" / repo_name
    assert (target_repo / "refs" / "main").read_text() == revision
    assert (target_repo / "blobs").is_symlink()
    assert (target_repo / "snapshots" / revision).is_symlink()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_snapshot"] == str(snapshot.resolve())
    assert manifest["network_access"] is False


def test_cache_view_refuses_a_snapshot_from_another_repo(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "models--Example--One", "revision")

    with pytest.raises(ValueError, match="snapshot belongs"):
        prepare_cache_view(
            cache_root=tmp_path / "view",
            repo_cache_name="models--Example--Two",
            snapshot=snapshot,
        )
