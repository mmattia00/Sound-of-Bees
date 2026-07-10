#!/usr/bin/env python3
"""
dsp_ground_truth_analysis.py

DESCRIPTION:
    Evaluate the DSP (mono) whoop detection pipeline against a ground truth
    CSV. Loads raw candidates from an HDF5 results file, applies a
    configurable filter, matches surviving candidates against ground truth
    events by filename + peak time, and prints a precision/recall/F1 report.

USAGE:
    python dsp_ground_truth_analysis.py --h5 <results_raw.h5> --gt <ground_truth.csv> [--out-gt <annotated_gt.csv>]

ARGUMENTS:
    --h5        Path to results_raw.h5 (required)
    --gt        Path to ground_truth.csv (required)
    --out-gt    Save the ground truth annotated with a 'found' column (optional)

EXAMPLE:
    python dsp_ground_thuth_analysis.py --h5 Z:\\postprocessing_results\\labelled_validation_dataset\\results_raw.h5 --gt Z:\\postprocessing_results\\labelled_validation_dataset\\ground_thruth.csv

OUTPUT:
    Prints an evaluation report (raw/filtered candidate counts, TP/FP/FN,
    precision, recall, F1), lists false positives and missed ground-truth
    events, and optionally writes an annotated ground truth CSV.
"""

import argparse
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py


# ---------------------------------------------------------------------------
# FILTER — edit here to change the filtering conditions
# ---------------------------------------------------------------------------

# FILTER USED IN THE FINAL PIPELINE
# def passes_filter(f0: float, duration: float, hnr_level: float) -> bool:
#     """
#     Returns True if the candidate passes the filter.
#     f0 and duration can be nan (candidates with no detected pitch).
#     """
#     if np.isnan(f0) or np.isnan(duration):
#         return False

#     cond1 = (310 <= f0 <= 350) and (0.050 <= duration <= 0.100)
#     cond2 = (430 <= f0 <= 470) and (0.110 <= duration <= 0.190)
#     cond3 = (500 <= f0 <= 600) and (0.100 <= duration <= 0.220)
#     hnr_cond = hnr_level >= 2

#     return (cond1 or cond2 or cond3) and hnr_cond


def passes_filter(f0: float, duration: float, hnr_level: float) -> bool:
    """
    Returns True if the candidate passes the filter.
    f0 and duration can be nan (candidates with no detected pitch).

    NOTE: this is currently a pass-through filter (only rejects nan pitch/
    duration). The actual acoustic filter used in the final pipeline is
    commented out above — swap it back in to restore stricter filtering.
    """
    if np.isnan(f0) or np.isnan(duration):
        return False

    return True


# ---------------------------------------------------------------------------
# group_id parsing
# ---------------------------------------------------------------------------

def parse_group_id(group_id: str):
    """Extract the source WAV filename and the peak time (seconds) from a group_id string."""
    m = re.match(
        r"^(audio_recording_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d+Z_ch_\d+)"
        r"_peak_([0-9]+(?:\.[0-9]+)?)$",
        group_id,
    )
    if not m:
        return None, None
    return m.group(1) + ".wav", float(m.group(2))


# ---------------------------------------------------------------------------
# Loading candidates from .h5
# ---------------------------------------------------------------------------

def load_candidates(h5_path: str) -> list[dict]:
    """Load all candidate groups from the HDF5 file into a list of dicts."""
    candidates = []
    with h5py.File(h5_path, "r") as f:
        for gid in f.keys():
            grp = f[gid]
            filename, peak_time = parse_group_id(gid)
            if filename is None:
                print(f"  ⚠️  group_id not parsable: {gid}", file=sys.stderr)
                continue
            f0        = float(grp["f0"       ][()]) if "f0"        in grp else np.nan
            duration  = float(grp["duration" ][()]) if "duration"  in grp else np.nan
            hnr_level = float(grp["hnr_level"][()]) if "hnr_level" in grp else np.nan
            candidates.append({
                "group_id":  gid,
                "filename":  filename,
                "peak_time": peak_time,
                "f0":        f0,
                "duration":  duration,
                "hnr_level": hnr_level,
                "matched":   False,
            })
    return candidates


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_filter(candidates: list[dict]) -> list[dict]:
    """Keep only candidates that pass passes_filter()."""
    kept = [c for c in candidates if passes_filter(c["f0"], c["duration"], c["hnr_level"])]
    return kept


# ---------------------------------------------------------------------------
# Loading ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """Load the ground truth CSV and add a 'found' column initialized to False."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["found"] = False
    return df


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match(candidates: list[dict], gt: pd.DataFrame) -> None:
    """
    Match each candidate against ground truth events for the same file:
    a candidate is a match (TP) if its peak_time falls within a GT event's
    [onset_sec, offset_sec] interval. Updates candidate['matched'] and
    gt['found'] in place.
    """
    gt_index = defaultdict(list)
    for idx, row in gt.iterrows():
        gt_index[row["filename"]].append((idx, row["onset_sec"], row["offset_sec"]))

    for cand in candidates:
        for idx, onset, offset in gt_index.get(cand["filename"], []):
            if onset <= cand["peak_time"] <= offset:
                cand["matched"] = True
                gt.at[idx, "found"] = True
                break


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(candidates_raw: list[dict], candidates_filtered: list[dict],
                 gt: pd.DataFrame) -> None:
    """Compute precision/recall/F1 and print a full evaluation report."""
    total_raw      = len(candidates_raw)
    total_filtered = len(candidates_filtered)
    tp  = sum(1 for c in candidates_filtered if c["matched"])
    fp  = total_filtered - tp
    fn  = int((~gt["found"]).sum())
    total_gt = len(gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else float("nan"))

    sep = "=" * 60
    print(f"\n{sep}")
    print("EVALUATION REPORT")
    print(sep)
    print(f"  Raw candidates (h5)             : {total_raw}")
    print(f"  Candidates after filtering      : {total_filtered}")
    print(f"  TP — found in ground truth      : {tp}")
    print(f"  FP — not found (false positives): {fp}")
    print(f"  FN — GT events not found        : {fn}  /  {total_gt} total GT")
    print(sep)
    print(f"  Precision  : {precision:.3f}")
    print(f"  Recall     : {recall:.3f}")
    print(f"  F1 score   : {f1:.3f}")
    print(sep)

    if fp > 0:
        print("\nFALSE POSITIVES:")
        for c in candidates_filtered:
            if not c["matched"]:
                f0_str  = f"{c['f0']:.1f} Hz"    if not np.isnan(c["f0"])       else "f0=nan"
                dur_str = f"{c['duration']:.3f}s" if not np.isnan(c["duration"]) else "dur=nan"
                hnr_str = f"{c['hnr_level']:.2f} dB"
                print(f"  {c['group_id']}  [{f0_str}  {dur_str}  HNR={hnr_str}]")

    missed = gt[~gt["found"]]
    if len(missed) > 0:
        print("\nGT EVENTS NOT FOUND:")
        for _, row in missed.iterrows():
            print(f"  {row['filename']}  [{row['onset_sec']:.3f} - {row['offset_sec']:.3f}]")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the mono pipeline against a ground truth CSV"
    )
    parser.add_argument("--h5",     required=True, help="Path to results_raw.h5")
    parser.add_argument("--gt",     required=True, help="Path to ground_truth.csv")
    parser.add_argument("--out-gt", default=None,
                        help="Save ground truth annotated with a 'found' column (optional)")
    args = parser.parse_args()

    print(f"  Loading raw candidates: {args.h5}")
    candidates_raw = load_candidates(args.h5)
    print(f"  → {len(candidates_raw)} raw candidates")

    print(f"  Loading ground truth  : {args.gt}")
    gt = load_ground_truth(args.gt)
    print(f"  → {len(gt)} GT events")

    candidates_filtered = apply_filter(candidates_raw)
    print(f"  → {len(candidates_filtered)} candidates after passes_filter()")

    match(candidates_filtered, gt)
    print_report(candidates_raw, candidates_filtered, gt)

    if args.out_gt:
        gt.to_csv(args.out_gt, index=False)
        print(f"  Annotated ground truth saved: {args.out_gt}")


if __name__ == "__main__":
    main()


# usage example:
# python dsp_ground_thuth_analysis.py --h5 Z:\postprocessing_results\labelled_validation_dataset\results_raw.h5 --gt Z:\postprocessing_results\labelled_validation_dataset\ground_thruth.csv