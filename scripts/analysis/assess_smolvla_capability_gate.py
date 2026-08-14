from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid boolean value: {value!r}")
    return normalized == "true"


def _selected_five_protocol(path: Path) -> tuple[list[int], list[int], int, int]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selected-five config must contain a YAML mapping")
    benchmark = payload.get("benchmark")
    execution = payload.get("execution")
    if not isinstance(benchmark, dict) or not isinstance(execution, dict):
        raise ValueError("selected-five config requires benchmark and execution mappings")
    task_ids = benchmark.get("task_ids")
    initial_state_ids = benchmark.get("initial_state_ids")
    repetitions = execution.get("episodes_per_initial_state")
    seed = execution.get("seed")
    if (
        not isinstance(task_ids, list)
        or len(task_ids) != 5
        or len(set(task_ids)) != 5
        or any(not isinstance(value, int) or isinstance(value, bool) for value in task_ids)
    ):
        raise ValueError("selected-five config must define five unique integer task IDs")
    if (
        not isinstance(initial_state_ids, list)
        or len(initial_state_ids) != 1
        or not isinstance(initial_state_ids[0], int)
        or isinstance(initial_state_ids[0], bool)
    ):
        raise ValueError("selected-five config must define exactly one initial state ID")
    if repetitions != 5:
        raise ValueError("selected-five config must define five repetitions")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("selected-five config seed must be a non-negative integer")
    return task_ids, initial_state_ids, repetitions, seed


def _validate_selected_five_rows(
    rows: list[dict[str, str]],
    task_ids: list[int],
    initial_state_ids: list[int],
    repetitions: int,
    base_seed: int,
) -> None:
    expected_count = len(task_ids) * len(initial_state_ids) * repetitions
    if len(rows) != expected_count:
        raise ValueError(
            f"selected-five protocol requires {expected_count} trials; found {len(rows)}"
        )

    by_episode: dict[int, dict[str, str]] = {}
    for row in rows:
        episode_id = int(row["episode_id"])
        if episode_id in by_episode:
            raise ValueError(f"selected-five task/seed structure has duplicate episode {episode_id}")
        by_episode[episode_id] = row

    if set(by_episode) != set(range(expected_count)):
        raise ValueError("selected-five task/seed structure requires episode IDs 0 through 24")

    episode_id = 0
    for task_id in task_ids:
        for initial_state_id in initial_state_ids:
            for _ in range(repetitions):
                row = by_episode[episode_id]
                expected_seed = base_seed + episode_id
                actual = (
                    int(row["task_id"]),
                    int(row["initial_state_id"]),
                    int(row["seed"]),
                    int(row["reset_seed"]),
                )
                expected = (task_id, initial_state_id, expected_seed, expected_seed)
                if actual != expected:
                    raise ValueError(
                        "selected-five task/seed structure mismatch at "
                        f"episode {episode_id}: expected {expected}, found {actual}"
                    )
                if not _as_bool(row["action_valid"]):
                    raise ValueError(
                        f"selected-five protocol requires action_valid=True at episode {episode_id}"
                    )
                _as_bool(row["success"])
                episode_id += 1


def _task_rates(rows: list[dict[str, str]]) -> dict[int, float]:
    grouped: dict[int, list[bool]] = {}
    for row in rows:
        grouped.setdefault(int(row["task_id"]), []).append(_as_bool(row["success"]))
    return {
        task_id: sum(results) / len(results)
        for task_id, results in sorted(grouped.items())
    }


def evaluate(rows: list[dict[str, str]]) -> dict[str, object]:
    rates = _task_rates(rows)
    mean_rate = sum(rates.values()) / len(rates)
    successful_task_count = sum(rate > 0.0 for rate in rates.values())
    passed = successful_task_count >= 3 and mean_rate > 0.0
    return {
        "successful_task_count": successful_task_count,
        "mean_success_rate": mean_rate,
        "passed": passed,
        "fine_tuning_required": not passed,
    }


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        rows = _load_rows(args.trials)
        protocol = _selected_five_protocol(args.config)
        _validate_selected_five_rows(rows, *protocol)
        result = evaluate(rows)
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
