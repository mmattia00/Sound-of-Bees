#!/usr/bin/env python3
"""
a2v_estimate_single_manifest.py

DESCRIPTION:
    Full analysis of an h5 file produced by get_results_for_single_manifest_split.py.
    Generates: PR curve, ROC curve, F1 vs threshold, confusion matrix (best
    threshold only), probability distribution (dual y-axis), and per-file
    timelines.

USAGE:
    python3 a2v_estimate_single_manifest.py \
        --h5_path <path_to_predictions.h5> \
        --out_path <output_folder> \
        [--threshold 0.125] \
        [--max_timeline_files 10]

ARGUMENTS:
    --h5_path             Path to the predictions_*.h5 file (required)
    --out_path             Output folder for plots and metrics (default: ./analysis)
    --threshold             Reference threshold for metrics (default: 0.125)
    --max_timeline_files    Max number of files to plot in the timeline (default: 10)

EXAMPLES:
    python3 a2v_estimate_single_manifest.py \
        --h5_path /mnt/A26-03-0300/a2v_estimates/predictions_Wav2VecCcasFinetune_checkpoint_best.pt_0_16_valid_0_bee-finetune-last-checkpoint-no-augmentation.h5 \
        --out_path /mnt/A26-03-0300/a2v_estimates/analysis \
        --threshold 0.125 \
        --max_timeline_files 10

    python a2v_estimate_single_manifest.py --h5_path Z:\a2v_estimates\batch_size_bs_160s\predictions_Wav2VecCcasFinetune_checkpoint_best.pt_0_16_valid_0_bee-finetune-last-checkpoint-no-augmentation_bs_160s.h5 --out_path Z:\ --threshold 0.125 --max_timeline_files 10

OUTPUT:
    - PDF plots (PR curve, ROC curve, F1 vs threshold, confusion matrix,
      probability distribution, timelines, combined PR+ROC) saved in --out_path
    - metrics_summary.json with global metrics saved in --out_path

NOTE:
    In main(), most individual plotting calls are currently commented out;
    only the combined PR+ROC stacked plot (plot_pr_roc_stacked_pdf) is active
    by default. Uncomment the other calls to regenerate the full set of plots
    and the metrics summary.
"""

# Usage example:
# python3 a2v_estimate_single_manifest.py \
#   --h5_path /mnt/A26-03-0300/a2v_estimates/predictions_Wav2VecCcasFinetune_checkpoint_best.pt_0_16_valid_0_bee-finetune-last-checkpoint-no-augmentation.h5 \
#   --out_path /mnt/A26-03-0300/a2v_estimates/analysis \
#   --threshold 0.125 \
#   --max_timeline_files 10

# python a2v_estimate_single_manifest.py --h5_path Z:\a2v_estimates\batch_size_bs_160s\predictions_Wav2VecCcasFinetune_checkpoint_best.pt_0_16_valid_0_bee-finetune-last-checkpoint-no-augmentation_bs_160s.h5 --out_path Z:\ --threshold 0.125 --max_timeline_files 10

import os
import argparse
import numpy as np
import h5py
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, f1_score, precision_score, recall_score
)


def get_parser():
    parser = argparse.ArgumentParser(description="Analysis of predictions from an animal2vec h5 file")
    parser.add_argument("--h5_path", required=True, type=str,
                        help="Path to the predictions_*.h5 file")
    parser.add_argument("--out_path", default="./analysis", type=str,
                        help="Output folder for plots and metrics")
    parser.add_argument("--threshold", default=0.125, type=float,
                        help="Reference threshold (default: 0.125)")
    parser.add_argument("--max_timeline_files", default=10, type=int,
                        help="Maximum number of files to plot in the timeline (default: 10)")
    return parser


def load_h5(h5_path):
    """Load fnames, likelihoods, and targets from every group in the h5 file."""
    all_likelihoods, all_targets = [], []
    all_fnames = []
    with h5py.File(h5_path, "r") as f:
        for k in sorted(f.keys()):
            grp = f[k]
            raw = grp["fname"][()]
            fname = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            all_fnames.append(fname)
            all_likelihoods.append(grp["likelihood"][()])
            all_targets.append(grp["target"][()])
    return all_fnames, all_likelihoods, all_targets


def flatten_data(likelihoods, targets):
    """Flatten per-file likelihood/target arrays into single 1D arrays."""
    L = np.concatenate([x.flatten() for x in likelihoods])
    T = np.concatenate([x.flatten() for x in targets])
    return L, T


def plot_pr_curve(L, T, threshold, out_path):
    """Plot the standalone Precision-Recall curve and compute the best-F1 threshold."""
    precision, recall, thresholds = precision_recall_curve(T, L)
    pr_auc = auc(recall, precision)
    f1_scores = np.where(precision + recall > 0,
                         2 * precision * recall / (precision + recall), 0)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    # --- PR Curve (standalone) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#2196F3", linewidth=2, label=f"AUC-PR = {pr_auc:.3f}")
    ax.axvline(recall[best_idx], color="orange", linestyle="--", alpha=0.8,
               label=f"Best F1={best_f1:.3f} @ thr={best_thresh:.3f}")
    ax.set_xlabel("Recall", fontsize=25)
    ax.set_ylabel("Precision", fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(out_path, "pr_curve.pdf")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Saved: {out_file}")

    return pr_auc, best_thresh, best_f1


def plot_f1_vs_threshold(L, T, threshold, out_path):
    """Plot F1 score as a function of threshold, highlighting the best threshold."""
    precision, recall, thresholds = precision_recall_curve(T, L)
    pr_auc = auc(recall, precision)
    f1_scores = np.where(precision + recall > 0,
                         2 * precision * recall / (precision + recall), 0)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    # --- F1 vs Threshold (standalone, best threshold only) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, f1_scores[:-1], color="#4CAF50", linewidth=2)
    ax.axvline(best_thresh, color="orange", linestyle="--", alpha=0.9,
               label=f"Best thr={best_thresh:.3f}  →  F1={best_f1:.3f}")
    ax.set_xlabel("Threshold", fontsize=25)
    ax.set_ylabel("F1 Score", fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # ax.set_title("F1 Score vs Threshold")
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(out_path, "f1_vs_threshold.pdf")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Saved: {out_file}")

    return pr_auc, best_thresh, best_f1


def plot_roc_curve(L, T, out_path):
    """Plot the standalone ROC curve."""
    fpr, tpr, _ = roc_curve(T, L)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#9C27B0", linewidth=2, label=f"AUC-ROC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=25)
    ax.set_ylabel("True Positive Rate", fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.set_title("ROC Curve", fontsize=25)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(out_path, "roc_curve.pdf")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Saved: {out_file}")
    return roc_auc


def plot_confusion_matrix(L, T, best_thresh, out_path):
    """Confusion matrix computed only at the best-F1 threshold."""
    preds = (L >= best_thresh).astype(int)
    cm = confusion_matrix(T.astype(int), preds)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes = ["Negative", "Positive"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=20); ax.set_yticklabels(classes, fontsize=20)
    ax.set_xlabel("Predicted", fontsize=25)
    ax.set_ylabel("True", fontsize=25)
    # ax.set_title(f"Confusion Matrix @ best-F1 thr={best_thresh:.3f}", fontsize=25)
    thresh_cm = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh_cm else "black", fontsize=17)
    plt.tight_layout()
    out_file = os.path.join(out_path, "confusion_matrix.pdf")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Saved: {out_file}")
    return tn, fp, fn, tp


def plot_prob_distribution(L, T, best_thresh, out_path):
    """Plot the distribution of predicted probabilities for positives vs negatives (dual y-axis)."""
    L_pos = L[T == 1]
    L_neg = L[T == 0]
    bins = np.linspace(0, 1, 50)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.hist(L_neg, bins=bins, alpha=0.6, color="#F44336",
             label=f"Negatives (n={len(L_neg):,})")
    ax2.hist(L_pos, bins=bins, alpha=0.6, color="#2196F3",
             label=f"Positives (n={len(L_pos):,})")

    ax1.axvline(best_thresh, color="orange", linestyle="--", linewidth=1.5,
                label=f"Best thr={best_thresh:.3f}")   # note: now uses best_thresh, not the raw threshold arg

    ax1.set_xlabel("Predicted Probability", fontsize=20)
    ax1.set_ylabel("Frame Count — Negatives", color="#F44336", fontsize=20)
    ax2.set_ylabel("Frame Count — Positives", color="#2196F3", fontsize=20)
    ax1.tick_params(axis="y", labelcolor="#F44336", labelsize=15)
    ax2.tick_params(axis="y", labelcolor="#2196F3", labelsize=15)
    ax1.tick_params(axis="x", labelsize=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=15)

    # ax1.set_title("Probability Distribution: Positives vs Negatives (dual y-axis)")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(out_path, "probability_distribution.pdf")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Saved: {out_file}")


def plot_timelines(fnames, likelihoods, targets, threshold, best_thresh, out_path, max_files=10):
    """Plot per-file timelines comparing predicted likelihood against ground truth."""
    n = min(max_files, len(fnames))
    n = 5  # hardcoded override: always plot 5 files regardless of max_files
    fig, axes = plt.subplots(n, 1, figsize=(16, 5 * n))
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        L = likelihoods[i][:, 0]
        T_gt = targets[i][:, 0]
        t = np.arange(len(L))
        ax.fill_between(t, T_gt, alpha=0.3, color="green", label="Ground truth")
        ax.plot(t, L, color="#2196F3", linewidth=0.8, alpha=0.9, label="Likelihood")
        ax.axhline(best_thresh, color="orange", linestyle="--", linewidth=0.9,
                   alpha=0.85, label=f"Best thr={best_thresh:.3f}")
        ax.set_ylim([-0.05, 1.1])
        ax.set_ylabel("Prob.", fontsize=25)
        ax.tick_params(axis="both", which="major", labelsize=20)
        # ax.set_title(Path(fnames[i]).name, fontsize=8)
        if i == 0:
            ax.legend(loc="upper right", fontsize=15, ncol=3)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Frame index", fontsize=25)
    # plt.suptitle("Timelines: Likelihood vs Ground Truth", fontsize=12, y=1.01)
    plt.tight_layout()
    out_file = os.path.join(out_path, "timelines.pdf")
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_file}")


def print_and_save_metrics(L, T, threshold, pr_auc, roc_auc, best_thresh, best_f1,
                           tn, fp, fn, tp, out_path):
    """Compute reference-threshold metrics, print a summary, and save it as JSON."""
    import json
    preds = (L >= threshold).astype(int)
    p = precision_score(T.astype(int), preds, zero_division=0)
    r = recall_score(T.astype(int), preds, zero_division=0)
    f1 = f1_score(T.astype(int), preds, zero_division=0)
    print("\n" + "="*60)
    print("  GLOBAL METRICS")
    print("="*60)
    print(f"  Total frames:        {len(T):,}")
    print(f"  Positive prevalence: {T.mean():.4f} ({T.sum():,.0f} frames)")
    print(f"  AUC-PR:              {pr_auc:.4f}")
    print(f"  AUC-ROC:             {roc_auc:.4f}")
    print(f"  Reference threshold: {threshold}")
    print(f"    Precision:         {p:.4f}")
    print(f"    Recall:            {r:.4f}")
    print(f"    F1:                {f1:.4f}")
    print(f"  Best threshold:      {best_thresh:.4f} → F1={best_f1:.4f}")
    print(f"    TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print("="*60)
    metrics = {
        "auc_pr": float(pr_auc), "auc_roc": float(roc_auc),
        "threshold": float(threshold),
        "precision_at_ref_thr": float(p),
        "recall_at_ref_thr": float(r),
        "f1_at_ref_thr": float(f1),
        "best_threshold": float(best_thresh), "best_f1": float(best_f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_frames": int(len(T)), "prevalence": float(T.mean()),
    }
    out_file = os.path.join(out_path, "metrics_summary.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {out_file}")


def plot_pr_roc_stacked_pdf(L, T, out_path):
    """Plot PR curve and ROC curve stacked vertically into a single PDF figure."""
    precision, recall, pr_thresholds = precision_recall_curve(T, L)
    pr_auc = auc(recall, precision)

    f1_scores = np.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        0
    )
    best_idx = np.argmax(f1_scores)
    best_thresh = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else pr_thresholds[-1]
    best_f1 = f1_scores[best_idx]

    fpr, tpr, _ = roc_curve(T, L)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(2, 1, figsize=(7, 10))

    # --- PR curve ---
    ax = axes[0]
    ax.plot(recall, precision, color="#2196F3", linewidth=2, label=f"AUC-PR = {pr_auc:.3f}")
    ax.scatter(
        recall[best_idx], precision[best_idx],
        color="orange", s=45, zorder=3,
        label=f"Best F1 = {best_f1:.3f} @ thr = {best_thresh:.3f}"
    )
    ax.set_xlabel("Recall", fontsize=25)
    ax.set_ylabel("Precision", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=20)
    # ax.set_title("Precision-Recall Curve", fontsize=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=15)

    # --- ROC curve ---
    ax = axes[1]
    ax.plot(fpr, tpr, color="#9C27B0", linewidth=2, label=f"AUC-ROC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=25)
    ax.set_ylabel("True Positive Rate", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=20)
    # ax.set_title("ROC Curve", fontsize=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=15)

    plt.tight_layout()
    out_file = os.path.join(out_path, "pr_roc_curves.pdf")
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_file}")

    return pr_auc, roc_auc, best_thresh, best_f1


def main(args):
    os.makedirs(args.out_path, exist_ok=True)
    print(f"\nLoading {args.h5_path} ...")
    fnames, likelihoods, targets = load_h5(args.h5_path)
    print(f"  Files loaded: {len(fnames)}")
    L, T = flatten_data(likelihoods, targets)
    print(f"  Total frames: {len(L):,}  |  Positives: {T.sum():,.0f} ({100*T.mean():.2f}%)")
    print("\nComputing metrics and plots...")
    # pr_auc, best_thresh, best_f1 = plot_pr_curve(L, T, args.threshold, args.out_path)
    # _, _, _ = plot_f1_vs_threshold(L, T, args.threshold, args.out_path)
    # roc_auc = plot_roc_curve(L, T, args.out_path)
    # tn, fp, fn, tp = plot_confusion_matrix(L, T, best_thresh, args.out_path)
    # plot_prob_distribution(L, T, best_thresh, args.out_path)
    # plot_timelines(fnames, likelihoods, targets,
    #                args.threshold, best_thresh, args.out_path, args.max_timeline_files)
    # print_and_save_metrics(L, T, args.threshold, pr_auc, roc_auc,
    #                        best_thresh, best_f1, tn, fp, fn, tp, args.out_path)
    # print(f"\nDone. Results in: {args.out_path}\n")
    plot_pr_roc_stacked_pdf(L, T, args.out_path)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)


