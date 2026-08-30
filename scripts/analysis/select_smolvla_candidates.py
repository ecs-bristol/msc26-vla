from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid success value: {value!r}")
    return normalized == "true"


def select_task_ids(rows: list[dict[str, str]], minimum_tasks: int) -> list[int]:
    if minimum_tasks < 1:
        raise ValueError("minimum_tasks must be at least one")
    all_ids = sorted({int(row["task_id"]) for row in rows})
    if len(all_ids) < minimum_tasks:
        raise ValueError(
            f"cannot select minimum {minimum_tasks} task IDs; "
            f"trials CSV contains {len(all_ids)} unique task IDs"
        )
    successes = sorted({int(row["task_id"]) for row in rows if _as_bool(row["success"])})
    return (successes + [task_id for task_id in all_ids if task_id not in successes])[
        : max(minimum_tasks, len(successes))
    ]


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-tasks", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        selected = select_task_ids(_load_rows(args.trials), args.minimum_tasks)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
