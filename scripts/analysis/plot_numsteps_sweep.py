#!/usr/bin/env python
"""Generate report figures for the num_steps sweep from the evidence CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="evidence/latest/pc_local/num_steps",
        help="directory containing num_steps_summary.csv and num_steps_per_task.csv",
    )
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )


def _fig_success(summary: pd.DataFrame, per_task: pd.DataFrame, outdir: Path) -> None:
    confirm = summary[summary["run_type"] == "confirmation"].sort_values("num_steps")
    screen = summary[summary["run_type"] == "screening"].sort_values("num_steps")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(screen["num_steps"], screen["overall_success_pct"], "o", ms=6, c="gray",
            label="screening (1 ep/task)")
    xs, ys, errs = [], [], []
    for _, row in confirm.iterrows():
        rows = per_task[per_task["num_steps"] == row["num_steps"]]
        rates = rows["successes"] / rows["n_episodes"]
        xs.append(row["num_steps"])
        ys.append(row["overall_success_pct"])
        errs.append(rates.std() * 100)
    ax.errorbar(xs, ys, yerr=errs, fmt="-o", ms=7, capsize=4, lw=2, c="#d62728",
                label="confirmation (5 ep/task)")
    ax.set_xlabel(r"Flow-matching denoising steps ($num\_steps$)")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(sorted(confirm["num_steps"].tolist() + screen["num_steps"].tolist()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig_success_vs_numsteps.png")
    plt.close(fig)


def _fig_time(summary: pd.DataFrame, outdir: Path) -> None:
    confirm = summary[summary["run_type"] == "confirmation"].sort_values("num_steps")
    base = confirm[confirm["num_steps"] == confirm["num_steps"].max()]["eval_ep_s"].iloc[0]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(confirm["num_steps"].astype(str), confirm["eval_ep_s"],
                  color=["#4c72b0"] * len(confirm))
    for i, (_, row) in enumerate(confirm.iterrows()):
        saved = (1 - row["eval_ep_s"] / base) * 100
        ax.text(i, row["eval_ep_s"] + 1.5, f"{row['eval_ep_s']:.0f}s\n({saved:.0f}%)",
                ha="center", fontsize=9)
    ax.set_xlabel("num_steps")
    ax.set_ylabel("Mean episode time (s)")
    ax.set_title(f"Episode time vs denoising steps (baseline = {base:.0f}s at "
                 f"num_steps={confirm['num_steps'].max()})")
    fig.tight_layout()
    fig.savefig(outdir / "fig_episode_time_vs_numsteps.png")
    plt.close(fig)


def _fig_tradeoff(summary: pd.DataFrame, outdir: Path) -> None:
    confirm = summary[summary["run_type"] == "confirmation"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(confirm["eval_ep_s"], confirm["overall_success_pct"], s=90,
               c="#2ca02c")
    for _, row in confirm.iterrows():
        ax.annotate(f"ns={int(row['num_steps'])}", (row["eval_ep_s"], row["overall_success_pct"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Mean episode time (s)")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlim(40, 100)
    ax.set_ylim(0, 110)
    ax.set_title("Success-time trade-off (5 ep/task)")
    fig.tight_layout()
    fig.savefig(outdir / "fig_tradeoff.png")
    plt.close(fig)


def _fig_heatmap(per_task: pd.DataFrame, outdir: Path) -> None:
    confirm = per_task[per_task["run_type"] == "confirmation"]
    pivot = confirm.pivot_table(index="task_id", columns="num_steps",
                                values="successes", aggfunc="sum")
    counts = confirm.pivot_table(index="task_id", columns="num_steps",
                                 values="n_episodes", aggfunc="max")
    rates = pivot / counts
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(rates.values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(rates.shape[1]), rates.columns)
    ax.set_yticks(range(rates.shape[0]), rates.index)
    ax.set_xlabel("num_steps")
    ax.set_ylabel("Task ID")
    ax.set_title("Per-task success rate (5 ep/task)")
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            ax.text(j, i, f"{int(pivot.values[i, j])}/{int(counts.values[i, j])}",
                    ha="center", va="center", fontsize=9,
                    color="black" if rates.values[i, j] < 0.75 else "white")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(outdir / "fig_per_task_heatmap.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    outdir = data_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    _style()
    summary = pd.read_csv(data_dir / "num_steps_summary.csv")
    per_task = pd.read_csv(data_dir / "num_steps_per_task.csv")
    _fig_success(summary, per_task, outdir)
    _fig_time(summary, outdir)
    _fig_tradeoff(summary, outdir)
    _fig_heatmap(per_task, outdir)
    print(f"figures written to {outdir}")


if __name__ == "__main__":
    main()
