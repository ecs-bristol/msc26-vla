from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = PROJECT_ROOT / "scripts" / "analysis" / "compare_pytest_junit.py"
    spec = importlib.util.spec_from_file_location("compare_pytest_junit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_reports_exact_failure_node_id_sets(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.xml"
    current = tmp_path / "current.xml"
    baseline.write_text(
        '<testsuite><testcase classname="tests.test_a" name="stable">'
        '<failure message="x"/></testcase><testcase classname="tests.test_a" name="gone">'
        '<error message="x"/></testcase><testcase classname="tests.test_a" name="pass"/>'
        '</testsuite>',
        encoding="utf-8",
    )
    current.write_text(
        '<testsuite><testcase classname="tests.test_a" name="stable">'
        '<failure message="x"/></testcase><testcase classname="tests.test_b" name="new">'
        '<error message="x"/></testcase></testsuite>',
        encoding="utf-8",
    )

    result = _module().compare(baseline, current)

    assert result["new_failure_node_ids"] == ["tests/test_b.py::new"]
    assert result["disappeared_failure_node_ids"] == ["tests/test_a.py::gone"]
    assert result["unchanged_failure_node_ids"] == ["tests/test_a.py::stable"]
    assert result["baseline"]["counts"] == {
        "passed": 1,
        "skipped": 0,
        "failed": 1,
        "errors": 1,
    }
