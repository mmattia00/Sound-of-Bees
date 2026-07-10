#!/usr/bin/env python3
"""
plot_daily_validation_comparison.py

DESCRIPTION:
    Generate bar-chart figures comparing daily manually-validated candidate
    counts (true positives vs false positives) for the DSP and Animal2vec
    whoop-detection pipelines, with precision percentage annotated per day.

USAGE:
    python plot_daily_validation_comparison.py <output_dir> [--filename <name>]

ARGUMENTS:
    output_dir    Folder where the output plot will be saved (created if missing)
    --filename    Output file name, e.g. daily_validation_comparison.pdf or .svg
                  (default: daily_validation_comparison.pdf)

EXAMPLES:
    python plot_daily_validation_comparison.py Z:\plots
    python plot_daily_validation_comparison.py Z:\plots --filename daily_validation_a2v.svg

NOTE:
    Daily TP/FP counts for both pipelines (DATA_DSP, DATA_A2V) are hardcoded
    below — update these lists to plot new data.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# ── daily manually-validated candidate counts (date, TP, FP) ────────────────
# DSP pipeline
DATA_DSP = [
    ("2026-05-27", 53, 16),
    ("2026-05-28", 58, 14),
    ("2026-05-29", 70, 16),
    ("2026-05-30", 41, 9),
    ("2026-05-31", 48, 19),
    ("2026-06-01", 54, 22),
    ("2026-06-02", 48, 28),
    ("2026-06-03", 64, 30),
    ("2026-06-04", 46, 31),
    ("2026-06-05", 31, 15),
    ("2026-06-06", 51, 17),
    ("2026-06-07", 73, 20),
    ("2026-06-08", 80, 25),
    ("2026-06-09", 88, 11),
]

# Animal2vec pipeline
DATA_A2V = [
    ("2026-05-27", 24, 46),
    ("2026-05-28", 37, 62),
    ("2026-05-29", 95, 211),
    ("2026-05-30", 43, 110),
    ("2026-05-31", 20, 43),
]


def plot_pipeline(ax, data, title):
    """
    Draw a stacked bar chart (TP + FP) on `ax` for one pipeline's daily
    data, annotating each bar with its precision percentage.
    """
    dates = [d for d, _, _ in data]
    tp = np.array([t for _, t, _ in data])
    fp = np.array([f for _, _, f in data])
    total = tp + fp
    precision = tp / total * 100

    x = np.arange(len(dates))

    ax.bar(x, tp, label="Validated TP", color="#2E8B57")
    ax.bar(x, fp, bottom=tp, label="Validated FP", color="#C44E52")

    # annotate precision percentage above each stacked bar
    for i, p in enumerate(precision):
        ax.text(i, total[i] + max(total) * 0.03, f"{p:.1f}%", ha="center", va="bottom", fontsize=15)

    # ax.set_title(title)
    ax.set_ylabel("Number of candidates", fontsize=25)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=25)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_ylim(0, max(total) * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def build_subplotplot(output_dir: Path, filename: str = "daily_validation_comparison.pdf") -> Path:
    """Build a two-panel figure comparing DSP (top) and Animal2vec (bottom) daily validation results, and save it to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 12),
        constrained_layout=True
    )

    plot_pipeline(
        axes[0],
        DATA_DSP,
        "DSP pipeline: daily manually validated candidates"
    )

    plot_pipeline(
        axes[1],
        DATA_A2V,
        "Animal2vec pipeline: daily manually validated candidates"
    )

    axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=15)
    axes[1].set_xlabel("Recording day", fontsize=25)

    out_path = output_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_plot(output_dir: Path, filename: str = "daily_validation_a2v.pdf") -> Path:
    """Build a single-panel figure with only the Animal2vec daily validation results, and save it to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)

    plot_pipeline(
        ax,
        DATA_A2V,
        "Animal2vec pipeline: daily manually validated candidates"
    )

    ax.set_xlabel("Recording day", fontsize=25)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=15)

    out_path = output_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a vector figure with DSP and Animal2vec manual-validation results."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where the output plot will be saved.",
    )
    parser.add_argument(
        "--filename",
        default="daily_validation_comparison.pdf",
        help="Output file name, e.g. daily_validation_comparison.pdf or .svg",
    )
    args = parser.parse_args()

    # NOTE: calls build_plot() (Animal2vec-only panel). Use build_subplotplot()
    # instead if you want the two-panel DSP + Animal2vec comparison figure.
    saved = build_plot(args.output_dir, args.filename)
    print(saved)