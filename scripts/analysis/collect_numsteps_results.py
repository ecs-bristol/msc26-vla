#!/usr/bin/env python
"""Regenerate the num_steps evidence CSVs from official eval_info.json outputs.

Run in WSL after new runs land. Reads every
`~/vla/results/libero_spatial_pc_local_*ns*/eval_info.json` and writes
`evidence/latest/pc_local/num_steps/num_steps_summary.csv` and
`num_steps_per_task.csv`.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=os.path.expanduser("~/vla/results"),
        help="directory containing libero_*_pc_local_*ns*/eval_info.json runs",
    )
    parser.add_argument(
        "--suite",
        default="libero_spatial",
        help="suite prefix used in result directory names",
    )
    parser.add_argument(
        "--outdir",
        default="evidence/latest/pc_local/num_steps",
        help="where to write the CSV records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = os.path.join(args.results_dir, f"{args.suite}_pc_local_*ns*/eval_info.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no eval_info.json matched: {pattern}")

    summary_rows: list[dict] = []
    per_task_rows: list[dict] = []
    for path in paths:
        name = os.path.basename(os.path.dirname(path))
        match = re.search(r"ns(\d+)", name)
        if match is None:
            continue
        num_steps = int(match.group(1))
        with open(path, encoding="utf-8") as handle:
            info = json.load(handle)
        per_task = info["per_task"]
        n_episodes = len(per_task[0]["metrics"]["successes"])
        run_type = "confirmation" if n_episodes >= 3 else "screening"
        summary_rows.append(
            {
                "run_type": run_type,
                "num_steps": num_steps,
                "n_episodes_per_task": n_episodes,
                "overall_success_pct": info["overall"]["pc_success"],
                "eval_ep_s": info["overall"]["eval_ep_s"],
            }
        )
        for task in per_task:
            per_task_rows.append(
                {
                    "run_type": run_type,
                    "num_steps": num_steps,
                    "task_id": task["task_id"],
                    "successes": sum(task["metrics"]["successes"]),
                    "n_episodes": n_episodes,
                }
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "num_steps_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (outdir / "num_steps_per_task.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_task_rows[0]))
        writer.writeheader()
        writer.writerows(per_task_rows)
    print(f"wrote {len(summary_rows)} summary rows and {len(per_task_rows)} per-task rows to {outdir}")


if __name__ == "__main__":
    main()
