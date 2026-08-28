#!/usr/bin/env python3
"""Render three paper-ready figures from frozen final VLA CSV tables only.

This script never imports model or LIBERO packages. Development and confirmatory
cohorts are separated by a visible gap and labeled independently; no statistic is
pooled across cohorts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


COHORT_ORDER = ["parity-corrected development", "untouched held-out confirmatory"]
COLORS = {
    "Static-H1": "#8C8C8C",
    "Static-H10": "#4C78A8",
    "Static-H20": "#2A9D8F",
    "Adaptive-v1 H20→H1": "#E76F51",
    "Adaptive-v2a H20→H1": "#F4A261",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _layout(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], list[float]]:
    ordered: list[dict[str, str]] = []
    positions: list[float] = []
    cursor = 0.0
    for cohort in COHORT_ORDER:
        cohort_rows = sorted(
            (row for row in rows if row["cohort"] == cohort),
            key=lambda row: int(row["display_order"]),
        )
        for row in cohort_rows:
            ordered.append(row)
            positions.append(cursor)
            cursor += 1.0
        cursor += 1.2
    return ordered, positions


def _annotate_cohorts(ax: plt.Axes, rows: list[dict[str, str]], positions: list[float]) -> None:
    for cohort in COHORT_ORDER:
        xs = [x for row, x in zip(rows, positions) if row["cohort"] == cohort]
        ax.text(
            sum(xs) / len(xs),
            1.03,
            "Development cohort" if cohort.startswith("parity") else "Confirmatory cohort",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def _finish(ax: plt.Axes, rows: list[dict[str, str]], positions: list[float], ylabel: str) -> None:
    ax.set_xticks(positions, [row["condition"] for row in rows], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    _annotate_cohorts(ax, rows, positions)


def plot_success(data_dir: Path, output_dir: Path) -> None:
    rows, xs = _layout(_read(data_dir / "final_vla_success.csv"))
    rates = [float(row["success_rate"]) for row in rows]
    lower = [rate - float(row["wilson95_low"]) for row, rate in zip(rows, rates)]
    upper = [float(row["wilson95_high"]) - rate for row, rate in zip(rows, rates)]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.bar(xs, rates, color=[COLORS[row["condition"]] for row in rows], width=0.72)
    ax.errorbar(xs, rates, yerr=[lower, upper], fmt="none", ecolor="#222222", capsize=4)
    for x, rate, row in zip(xs, rates, rows):
        ax.text(x, rate + 0.025, f"{row['successes']}/{row['episodes']}", ha="center", fontsize=8)
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    _finish(ax, rows, xs, "Success@280")
    fig.savefig(output_dir / "final_vla_success.svg")
    fig.savefig(output_dir / "final_vla_success.png", dpi=300)
    plt.close(fig)


def plot_calls(data_dir: Path, output_dir: Path) -> None:
    rows, xs = _layout(_read(data_dir / "final_vla_model_calls.csv"))
    means = [float(row["mean_model_calls"]) for row in rows]
    medians = [float(row["median_model_calls"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.bar(xs, means, color=[COLORS[row["condition"]] for row in rows], width=0.72, label="Mean")
    ax.scatter(xs, medians, color="#111111", marker="D", s=22, zorder=3, label="Median")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper right")
    _finish(ax, rows, xs, "Model invocations per episode (log scale)")
    fig.savefig(output_dir / "final_vla_model_calls.svg")
    fig.savefig(output_dir / "final_vla_model_calls.png", dpi=300)
    plt.close(fig)


def plot_wall_time(data_dir: Path, output_dir: Path) -> None:
    rows, xs = _layout(_read(data_dir / "final_vla_wall_time.csv"))
    means = [float(row["mean_wall_time_s"]) for row in rows]
    medians = [float(row["median_wall_time_s"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.bar(xs, means, color=[COLORS[row["condition"]] for row in rows], width=0.72, label="Mean")
    ax.scatter(xs, medians, color="#111111", marker="D", s=22, zorder=3, label="Median")
    ax.legend(frameon=False, loc="upper right")
    _finish(ax, rows, xs, "Wall time per episode (s)")
    fig.savefig(output_dir / "final_vla_wall_time.svg")
    fig.savefig(output_dir / "final_vla_wall_time.png", dpi=300)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "analysis" / "figures")
    parser.add_argument("--output-dir", type=Path, default=project_root / "analysis" / "figures" / "generated")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_success(args.data_dir, args.output_dir)
    plot_calls(args.data_dir, args.output_dir)
    plot_wall_time(args.data_dir, args.output_dir)
    print("FINAL_VLA_FIGURES_RENDERED=3 formats=svg,png")


if __name__ == "__main__":
    main()
