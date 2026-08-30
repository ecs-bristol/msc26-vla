#!/usr/bin/env python
"""Generate report figures for experiment B (quantization) from evidence CSVs."""
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
        default="evidence/latest/pc_local/int4",
        help="directory containing quant_summary.csv and quant_bench.csv",
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


def _success_series(summary: pd.DataFrame) -> list[tuple[str, str, dict[int, float]]]:
    main = summary[summary["n_episodes_per_task"] >= 5]
    series = []
    for quant_method, scope, label, color in (
        ("none", "none", "fp16", "#4c72b0"),
        ("int8_groupwise", "language", "int8 language-only", "#2ca02c"),
        ("int8_groupwise", "backbone", "int8 backbone", "#d62728"),
    ):
        rows = main[
            (main["quant_method"] == quant_method) & (main["scope"] == scope)
        ]
        values = {int(row["num_steps"]): row["overall_success_pct"] for _, row in rows.iterrows()}
        series.append((label, color, values))
    return series


def _fig_success(summary: pd.DataFrame, outdir: Path) -> None:
    series = _success_series(summary)
    labels = ["ns10", "ns2"]
    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(6, 4))
    for offset, (label, color, values) in zip((-width, 0, width), series):
        heights = [values.get(10, 0), values.get(2, 0)]
        ax.bar(x + offset, heights, width, label=label, color=color)
        for i, height in enumerate(heights):
            ax.text(x[i] + offset, height + 1, f"{height:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    ax.set_xlabel("num_steps")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Quantization success rate (5 ep/task, 50 episodes)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_quant_success.png")
    plt.close(fig)


def _fig_memory(bench: pd.DataFrame, outdir: Path) -> None:
    fp16 = bench[bench["quant_method"] == "none"].iloc[0]
    language = bench[(bench["quant_method"] == "int8_groupwise") & (bench["scope"] == "language")].iloc[0]
    backbone = bench[(bench["quant_method"] == "int8_groupwise") & (bench["scope"] == "backbone")].iloc[0]
    mixed = bench[bench["quant_method"] == "mixed"].iloc[0]
    labels = ["Weights (MB)", "Peak VRAM (MB)"]
    groups = [
        ("fp16", "#4c72b0", [fp16["param_bytes_mb"], fp16["peak_allocated_mb"]]),
        ("int8 language-only", "#2ca02c", [language["param_bytes_mb"], language["peak_allocated_mb"]]),
        ("int8 backbone", "#d62728", [backbone["param_bytes_mb"], backbone["peak_allocated_mb"]]),
        ("mixed 4/8", "#9467bd", [mixed["param_bytes_mb"], mixed["peak_allocated_mb"]]),
    ]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(6, 4))
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * width
    for offset, (label, color, values) in zip(offsets, groups):
        ax.bar(x + offset, values, width, label=label, color=color)
        for i, value in enumerate(values):
            ax.text(x[i] + offset, value + 15, f"{value:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MB")
    ax.set_title("Memory: fp16 vs int8 (language-only vs backbone)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_quant_memory.png")
    plt.close(fig)


def _fig_latency(bench: pd.DataFrame, outdir: Path) -> None:
    fp16 = bench[bench["quant_method"] == "none"].set_index("num_steps")["mean_ms"]
    language = bench[(bench["quant_method"] == "int8_groupwise") & (bench["scope"] == "language")].set_index("num_steps")["mean_ms"]
    backbone = bench[(bench["quant_method"] == "int8_groupwise") & (bench["scope"] == "backbone")].set_index("num_steps")["mean_ms"]
    mixed = bench[bench["quant_method"] == "mixed"].set_index("num_steps")["mean_ms"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([2, 5, 10], [fp16[2], fp16[5], fp16[10]], "-o", label="fp16", color="#4c72b0")
    ax.plot([2, 5, 10], [language[2], language[5], language[10]], "-o", label="int8 language-only", color="#2ca02c")
    ax.plot([2, 10], [backbone[2], backbone[10]], "-o", label="int8 backbone", color="#d62728")
    ax.plot([2], [mixed[2]], "o", label="mixed 4/8", color="#9467bd", ms=8)
    ax.set_xlabel("num_steps")
    ax.set_ylabel("Mean inference (ms)")
    ax.set_title("Per-step inference latency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_quant_latency.png")
    plt.close(fig)


def _fig_inference_latency(bench: pd.DataFrame, outdir: Path) -> None:
    fp16 = bench[bench["quant_method"] == "none"].set_index("num_steps")
    language = bench[(bench["quant_method"] == "int8_groupwise") & (bench["scope"] == "language")].set_index("num_steps")
    backbone = bench[(bench["quant_method"] == "int8_groupwise") & (bench["scope"] == "backbone")].set_index("num_steps")

    fig, ax = plt.subplots(figsize=(6, 4))
    for frame, label, color in (
        (fp16, "fp16", "#4c72b0"),
        (language, "int8 language-only", "#2ca02c"),
        (backbone, "int8 backbone", "#d62728"),
    ):
        steps = sorted(frame.index)
        mean = [frame.loc[step, "mean_ms"] for step in steps]
        p95 = [frame.loc[step, "p95_ms"] for step in steps]
        ax.plot(steps, mean, "-o", label=label, color=color)
        ax.fill_between(steps, mean, p95, alpha=0.15, color=color)

    ax.annotate(
        "~32 ms per denoising step",
        xy=(10, fp16.loc[10, "mean_ms"]),
        xytext=(5, 430),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    ax.set_xticks([2, 5, 10])
    ax.set_xlabel("num_steps")
    ax.set_ylabel("Inference latency (ms)")
    ax.set_title("Policy-only inference latency (line = mean, shaded = p95)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_inference_latency_vs_numsteps.png")
    plt.close(fig)


def _fig_ablation_overview(summary: pd.DataFrame, bench: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    # Panel 1: success rate.
    ax = axes[0, 0]
    series = _success_series(summary)
    labels = ["ns10", "ns2"]
    x = np.arange(len(labels))
    width = 0.22
    for offset, (label, color, values) in zip((-width, 0, width), series):
        heights = [values.get(10, 0), values.get(2, 0)]
        ax.bar(x + offset, heights, width, label=label, color=color)
        for i, height in enumerate(heights):
            ax.text(x[i] + offset, height + 1, f"{height:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    ax.set_xlabel("num_steps")
    ax.set_ylabel("Success (%)")
    ax.set_title("Success rate (5 ep/task)")
    ax.text(0.02, 0.02, "int4 PTQ smoke: 30% (backbone), 10% (language-only)",
            transform=ax.transAxes, fontsize=8, color="#d62728")
    ax.legend(fontsize=7)

    # Panel 2: memory.
    ax = axes[0, 1]
    fp16 = bench[bench["quant_method"] == "none"].iloc[0]
    language = bench[(bench["quant_method"] == "int8_groupwise") & (bench["scope"] == "language")].iloc[0]
    backbone = bench[(bench["quant_method"] == "int8_groupwise") & (bench["scope"] == "backbone")].iloc[0]
    mem_labels = ["Weights", "Peak VRAM"]
    groups = [
        ("fp16", "#4c72b0", [fp16["param_bytes_mb"], fp16["peak_allocated_mb"]]),
        ("int8 language", "#2ca02c", [language["param_bytes_mb"], language["peak_allocated_mb"]]),
        ("int8 backbone", "#d62728", [backbone["param_bytes_mb"], backbone["peak_allocated_mb"]]),
    ]
    x = np.arange(len(mem_labels))
    for offset, (label, color, values) in zip((-width, 0, width), groups):
        ax.bar(x + offset, values, width, label=label, color=color)
        for i, value in enumerate(values):
            ax.text(x[i] + offset, value + 15, f"{value:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, mem_labels)
    ax.set_ylabel("MB")
    ax.set_title("Memory")
    ax.legend(fontsize=7)

    # Panel 3: latency at ns10 and ns2.
    ax = axes[1, 0]
    lat_labels = ["ns10", "ns2"]
    lat_groups = [
        ("fp16", "#4c72b0", [_mean_at(bench, "none", "all", 10), _mean_at(bench, "none", "all", 2)]),
        ("int8 language", "#2ca02c", [_mean_at(bench, "int8_groupwise", "language", 10), _mean_at(bench, "int8_groupwise", "language", 2)]),
        ("int8 backbone", "#d62728", [_mean_at(bench, "int8_groupwise", "backbone", 10), _mean_at(bench, "int8_groupwise", "backbone", 2)]),
    ]
    for offset, (label, color, values) in zip((-width, 0, width), lat_groups):
        ax.bar(x + offset, values, width, label=label, color=color)
        for i, value in enumerate(values):
            ax.text(x[i] + offset, value + 6, f"{value:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, lat_labels)
    ax.set_ylabel("Mean inference (ms)")
    ax.set_title("Latency")
    ax.legend(fontsize=7)

    # Panel 4: best-combined summary.
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = (
        "Best combined config\n\n"
        "int8 language-only + ns2\n"
        "success: 80%\n"
        "inference: 133 ms/step\n"
        "weights: 929 MB (-24%)\n\n"
        "int8 backbone + ns2\n"
        "success: 78%\n"
        "inference: 143 ms/step\n"
        "weights: 836 MB (-31%)"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle("SmolVLA software-optimization ablation (LIBERO spatial, PC local)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "fig_ablation_overview.png")
    plt.close(fig)


def _fig_incremental_ablation(summary: pd.DataFrame, bench: pd.DataFrame, outdir: Path) -> None:
    configs = [
        ("A", "none", "none", "all", 10, "#4c72b0", "baseline"),
        ("A+C", "int8_groupwise", "language", "language", 10, "#2ca02c", "+ int8"),
        ("A+B", "none", "none", "all", 2, "#4c72b0", "+ steps"),
        ("A+B+C", "int8_groupwise", "language", "language", 2, "#2ca02c", "+ steps + int8"),
    ]
    labels = [row[0] for row in configs]
    descriptions = [row[6] for row in configs]
    colors = [row[5] for row in configs]
    success, latency, memory = [], [], []
    for _, quant_method, summary_scope, bench_scope, num_steps, _, _ in configs:
        success_row = summary[
            (summary["quant_method"] == quant_method)
            & (summary["scope"] == summary_scope)
            & (summary["num_steps"] == num_steps)
            & (summary["n_episodes_per_task"] >= 5)
        ]
        bench_row = bench[
            (bench["quant_method"] == quant_method)
            & (bench["scope"] == bench_scope)
            & (bench["num_steps"] == num_steps)
        ]
        success.append(success_row["overall_success_pct"].iloc[0])
        latency.append(bench_row["mean_ms"].iloc[0])
        memory.append(bench_row["param_bytes_mb"].iloc[0])

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    bars = ax1.bar(x, success, 0.52, color=colors, alpha=0.85)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Success rate (%)")
    ax1.set_xticks(x, [f"{label}\n{desc}" for label, desc in zip(labels, descriptions)], fontsize=9)
    for i, (value, mem) in enumerate(zip(success, memory)):
        ax1.text(i, value + 3, f"{value:.0f}%", ha="center", fontsize=10)
        ax1.text(i, 4, f"{mem:.0f}MB", ha="center", fontsize=8, color="white")

    ax2 = ax1.twinx()
    ax2.plot(x, latency, "o-", color="#d62728", lw=2, label="inference latency")
    ax2.set_ylim(0, 520)
    ax2.set_ylabel("Mean inference (ms)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    for i, value in enumerate(latency):
        ax2.text(i, value + 18, f"{value:.0f}ms", ha="center", fontsize=9, color="#d62728")

    ax2.legend(loc="lower right", fontsize=8)
    fig.suptitle("Incremental ablation: step reduction + int8 quantization", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "fig_incremental_ablation.png")
    plt.close(fig)


def _mean_at(bench: pd.DataFrame, quant_method: str, scope: str, num_steps: int) -> float:
    rows = bench[(bench["quant_method"] == quant_method) & (bench["scope"] == scope) & (bench["num_steps"] == num_steps)]
    return rows["mean_ms"].iloc[0]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    outdir = data_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    _style()
    summary = pd.read_csv(data_dir / "quant_summary.csv")
    bench = pd.read_csv(data_dir / "quant_bench.csv")
    _fig_success(summary, outdir)
    _fig_memory(bench, outdir)
    _fig_latency(bench, outdir)
    _fig_inference_latency(bench, outdir)
    _fig_ablation_overview(summary, bench, outdir)
    _fig_incremental_ablation(summary, bench, outdir)
    print(f"figures written to {outdir}")


if __name__ == "__main__":
    main()
