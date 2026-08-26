"""Compare pytest failure node IDs from two JUnit XML reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_id(testcase: ET.Element) -> str:
    file_name = testcase.get("file")
    if file_name:
        prefix = file_name.replace("\\", "/")
    else:
        class_name = testcase.get("classname", "")
        prefix = class_name.replace(".", "/") + ".py"
    return f"{prefix}::{testcase.get('name', '<unnamed>')}"


def _read_report(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    statuses: dict[str, str] = {}
    counts = {"passed": 0, "skipped": 0, "failed": 0, "errors": 0}
    for testcase in root.iter("testcase"):
        node_id = _node_id(testcase)
        if testcase.find("failure") is not None:
            status = "failed"
        elif testcase.find("error") is not None:
            status = "error"
        elif testcase.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        counts["errors" if status == "error" else status] += 1
        if status in {"failed", "error"}:
            statuses[node_id] = status
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "counts": counts,
        "failing_node_ids": statuses,
    }


def compare(baseline: Path, current: Path) -> dict[str, object]:
    baseline_report = _read_report(baseline)
    current_report = _read_report(current)
    baseline_failures = set(baseline_report["failing_node_ids"])
    current_failures = set(current_report["failing_node_ids"])
    return {
        "schema_version": 1,
        "baseline": baseline_report,
        "current": current_report,
        "new_failure_node_ids": sorted(current_failures - baseline_failures),
        "disappeared_failure_node_ids": sorted(baseline_failures - current_failures),
        "unchanged_failure_node_ids": sorted(baseline_failures & current_failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare(args.baseline, args.current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "new": len(result["new_failure_node_ids"]),
        "disappeared": len(result["disappeared_failure_node_ids"]),
        "unchanged": len(result["unchanged_failure_node_ids"]),
        "output": str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
