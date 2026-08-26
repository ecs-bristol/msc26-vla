from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "wsl" / "check_paired_runtime_shutdown.py"


def test_lifecycle_smoke_source_has_one_action_selection_and_one_explicit_step() -> None:
    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        node.func.attr
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]

    assert calls.count("select_action") == 1
    assert calls.count("step") == 1
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(main))


def test_lifecycle_smoke_refuses_nonoffline_execution(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("HF_HUB_OFFLINE", None)
    environment.pop("TRANSFORMERS_OFFLINE", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-snapshot-path",
            str(tmp_path / "base"),
            "--vlm-snapshot-path",
            str(tmp_path / "vlm"),
            "--device",
            "cpu",
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1" in completed.stderr
