"""Create an isolated Hugging Face cache ref for an existing frozen snapshot.

The snapshot and blob store remain read-only and are linked into a new cache
view.  This is needed for processor configs that contain a Hub repo ID without
an explicit revision even though the model itself is loaded from a local path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def prepare_cache_view(*, cache_root: Path, repo_cache_name: str, snapshot: Path) -> Path:
    snapshot = snapshot.expanduser().resolve()
    revision = snapshot.name
    source_repo = snapshot.parent.parent
    if not repo_cache_name.startswith("models--") or "--" not in repo_cache_name[8:]:
        raise ValueError("repo_cache_name must use the Hugging Face models--ORG--NAME form")
    if source_repo.name != repo_cache_name:
        raise ValueError(
            f"snapshot belongs to {source_repo.name}, not requested {repo_cache_name}"
        )
    if not (snapshot / "config.json").is_file() or not (source_repo / "blobs").is_dir():
        raise ValueError("snapshot must be a complete local Hugging Face cache snapshot")

    target_repo = cache_root.expanduser().resolve() / repo_cache_name
    target_snapshot = target_repo / "snapshots" / revision
    target_ref = target_repo / "refs" / "main"
    target_snapshot.parent.mkdir(parents=True, exist_ok=True)
    target_ref.parent.mkdir(parents=True, exist_ok=True)

    links = {
        target_repo / "blobs": source_repo / "blobs",
        target_snapshot: snapshot,
    }
    for target, source in links.items():
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"existing cache-view link points elsewhere: {target}")
        elif target.exists():
            raise ValueError(f"cache-view path already exists and is not a symlink: {target}")
        else:
            target.symlink_to(source, target_is_directory=True)

    if target_ref.exists():
        current_ref = target_ref.read_text(encoding="utf-8")
        if current_ref.strip() != revision:
            raise ValueError(f"existing main ref does not match frozen revision: {target_ref}")
        # huggingface_hub uses the ref contents verbatim as a directory name.
        # A trailing newline therefore makes an otherwise valid snapshot miss.
        if current_ref != revision:
            target_ref.write_text(revision, encoding="utf-8")
    else:
        target_ref.write_text(revision, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "repo_cache_name": repo_cache_name,
        "revision": revision,
        "source_snapshot": str(snapshot),
        "source_blobs": str((source_repo / "blobs").resolve()),
        "cache_view": str(target_repo),
        "network_access": False,
    }
    manifest_path = target_repo / "offline_cache_view.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repo-cache-name", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_cache_view(
        cache_root=args.cache_root,
        repo_cache_name=args.repo_cache_name,
        snapshot=args.snapshot,
    )
    print(manifest)


if __name__ == "__main__":
    main()
