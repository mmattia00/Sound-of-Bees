#!/usr/bin/env python3
"""
comparison_btw_pipelines.py

DESCRIPTION:
    Compare DSP and animal2vec whoop detection pipelines by matching
    validated candidates from each pipeline's HDF5 database based on
    absolute peak timestamps within a configurable time window.

USAGE:
    python comparison_btw_pipelines.py <dsp_db_path> <a2v_db_path> [--window <sec>] [--verbose]

ARGUMENTS:
    dsp_db_path     Path to the (validated) DSP HDF5 database
    a2v_db_path     Path to the (validated) animal2vec HDF5 database
    --window SEC    Time threshold in seconds for a shared match (default: 30)
    --verbose       Print details for every candidate and every match

EXAMPLES:
    python comparison_btw_pipelines.py Z:\\postprocessing_results\\2026-06-01\\results_2026-06-01.h5 Z:\\postprocessing_results_animal2vec\\2026-06-01\\results_2026-06-01.h5
        # default 30s window, compact output without verbose details

    python comparison_btw_pipelines.py Z:\\postprocessing_results\\2026-06-01\\results_2026-06-01.h5 Z:\\postprocessing_results_animal2vec\\2026-06-01\\results_2026-06-01.h5 --window 15 --verbose

    python comparison_btw_pipelines.py Z:\\postprocessing_results\\2026-06-01\\results_2026-06-01.h5 Z:\\postprocessing_results_animal2vec\\2026-06-01\\results_2026-06-01.h5 --window 60 --verbose

OUTPUT:
    Prints, for both databases, the total candidates, validated-True counts,
    matched ("shared") events within the time window, and pipeline-only
    events. Returns a dict with all computed metrics.
"""

import argparse
import re
from datetime import datetime, timezone

import h5py


# ── helpers ───────────────────────────────────────────────────────────────────

# Matches group names like: 2026-06-01T14_23_05.123Z..._peak_12.34
_GNAME_RE = re.compile(
    r'^(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d+(?:\.\d+)?)Z'
    r'.*_peak_([0-9]+(?:\.[0-9]+)?)$'
)


def build_abs_peak_time(gname: str) -> float:
    """Parse a group name and return the absolute UTC peak timestamp (epoch seconds)."""
    m = _GNAME_RE.match(gname)
    if m is None:
        raise ValueError(f"Group name not parsable: '{gname}'")

    date_str = m.group(1)
    hh       = int(m.group(2))
    mm       = int(m.group(3))
    ss_full  = float(m.group(4))
    peak_rel = float(m.group(5))

    ss_int = int(ss_full)
    us     = round((ss_full - ss_int) * 1_000_000)

    dt = datetime(
        year        = int(date_str[:4]),
        month       = int(date_str[5:7]),
        day         = int(date_str[8:10]),
        hour        = hh,
        minute      = mm,
        second      = ss_int,
        microsecond = us,
        tzinfo      = timezone.utc,
    )
    return dt.timestamp() + peak_rel


def _is_validated_true(val) -> bool:
    """Return True if the stored sound_validated value represents boolean True."""
    try:
        return val is True or str(val) in ('True', "b'True'", '1')
    except Exception:
        return False


def _epoch_to_utc_str(epoch: float) -> str:
    """Format an epoch timestamp as a human-readable UTC string."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f UTC')


# ── loader ────────────────────────────────────────────────────────────────────

def load_validated_abs_times(db_path: str,
                              label: str,
                              verbose: bool = False) -> tuple[int, list[float]]:
    """
    Load the total candidate count and the list of absolute peak times for
    all groups marked sound_validated == True in the given HDF5 database.
    """
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  LOADING  [{label}]  →  {db_path}")
    print(sep)

    tot       = 0
    true_cnt  = 0
    skipped   = 0
    abs_times = []

    with h5py.File(db_path, 'r') as f:
        group_names = sorted(f.keys())
        n_groups    = len(group_names)
        print(f"  Groups found in database : {n_groups}")

        for idx, gname in enumerate(group_names, start=1):
            tot += 1
            grp = f[gname]

            if 'sound_validated' not in grp:
                status = "NO_FIELD"
            else:
                val    = grp['sound_validated'][()]
                status = "TRUE" if _is_validated_true(val) else f"OTHER({val})"

            try:
                abs_t    = build_abs_peak_time(gname)
                abs_str  = _epoch_to_utc_str(abs_t)
                parse_ok = True
            except ValueError as e:
                abs_str  = f"PARSE ERROR: {e}"
                abs_t    = None
                parse_ok = False
                skipped += 1

            if verbose:
                validated_marker = "✅" if status == "TRUE" else "  "
                print(f"  {validated_marker} [{idx:>5}/{n_groups}]  {gname}")
                print(f"          sound_validated : {status}")
                print(f"          abs peak time   : {abs_str}")
                if status == "TRUE" and not parse_ok:
                    print(f"          ⚠️  peak time not added (parse failed)")

            if status == "TRUE":
                true_cnt += 1
                if parse_ok:
                    abs_times.append(abs_t)

    print(f"\n  ── Summary [{label}] ────────────────────────────────")
    print(f"  Total candidates read     : {tot}")
    print(f"  sound_validated = True    : {true_cnt}")
    print(f"  sound_validated ≠ True    : {tot - true_cnt}")
    print(f"  Parse errors (skipped)    : {skipped}")
    print(f"  Peak times ready to match : {len(abs_times)}")
    print(sep)

    return tot, abs_times


# ── comparison ────────────────────────────────────────────────────────────────

def compare_pipelines(dsp_db_path: str,
                      a2v_db_path: str,
                      window_sec: float = 30.0,
                      verbose: bool = False) -> dict:
    """
    Load validated candidates from both pipelines and greedily match them
    by absolute peak time within `window_sec`, then compute and print
    comparison metrics (shared / DSP-only / A2V-only events).
    """

    print("\n" + "=" * 60)
    print("  PIPELINE COMPARISON  —  DSP  vs  animal2vec")
    print("=" * 60)
    print(f"  DSP database : {dsp_db_path}")
    print(f"  A2V database : {a2v_db_path}")
    print(f"  Window       : ±{window_sec:.1f} s")
    print(f"  Verbose      : {'on' if verbose else 'off'}")
    print("=" * 60)

    # ── loading ────────────────────────────────────────────────────────────
    tot_dsp, dsp_times = load_validated_abs_times(dsp_db_path, "DSP", verbose=verbose)
    tot_a2v, a2v_times = load_validated_abs_times(a2v_db_path, "A2V", verbose=verbose)

    # --- debug print all abs times ---
    print(f"\n  DSP validated abs times ({len(dsp_times)}):")
    for t in dsp_times:
        print(f"     DSP : {_epoch_to_utc_str(t)}")
    print(f"\n  A2V validated abs times ({len(a2v_times)}):")
    for t in a2v_times:
        print(f"     A2V : {_epoch_to_utc_str(t)}")

    true_dsp = len(dsp_times)
    true_a2v = len(a2v_times)

    # ── greedy matching ───────────────────────────────────────────────────
    # For each DSP candidate (sorted by time), find the nearest unmatched
    # A2V candidate within the time window. Each A2V candidate can only be
    # matched once.
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  GREEDY MATCHING  —  window = {window_sec:.1f} s")
    print(sep)
    if verbose:
        print(f"  Each validated DSP candidate searches for the nearest")
        print(f"  not-yet-matched A2V candidate within ±{window_sec:.1f} s.\n")

    a2v_used    = [False] * len(a2v_times)
    shared      = 0
    dsp_sorted  = sorted(enumerate(dsp_times), key=lambda x: x[1])

    for _, dsp_t in dsp_sorted:
        matched = False
        for j, a2v_t in enumerate(a2v_times):
            if a2v_used[j]:
                continue
            delta = abs(dsp_t - a2v_t)
            if delta < window_sec:
                if verbose:
                    print(f"  ✅ MATCH")
                    print(f"     DSP peak : {_epoch_to_utc_str(dsp_t)}")
                    print(f"     A2V peak : {_epoch_to_utc_str(a2v_t)}")
                    print(f"     |Δt|     : {delta:.3f} s  (< {window_sec:.1f} s)")
                shared      += 1
                a2v_used[j]  = True
                matched      = True
                break

        if not matched and verbose:
            print(f"  ❌ NO MATCH  DSP peak: {_epoch_to_utc_str(dsp_t)}")

    if verbose:
        unmatched_a2v = [a2v_times[j] for j, used in enumerate(a2v_used) if not used]
        if unmatched_a2v:
            print(f"\n  A2V candidates with no matching DSP event ({len(unmatched_a2v)}):")
            for t in unmatched_a2v:
                print(f"     A2V only : {_epoch_to_utc_str(t)}")

    # ── metrics ───────────────────────────────────────────────────────────
    dsp_only      = true_dsp - shared
    a2v_only      = true_a2v - shared
    precision_dsp = true_dsp / tot_dsp if tot_dsp > 0 else float('nan')
    precision_a2v = true_a2v / tot_a2v if tot_a2v > 0 else float('nan')

    # ── final report (always printed) ─────────────────────────────────────
    sep2 = "=" * 60
    print(f"\n{sep2}")
    print(f"  FINAL RESULTS")
    print(sep2)
    print(f"  {'Metric':40s}  {'DSP':>8}  {'A2V':>8}")
    print(f"  {'─'*56}")
    print(f"  {'Total candidates':40s}  {tot_dsp:>8}  {tot_a2v:>8}")
    print(f"  {'Validated True':40s}  {true_dsp:>8}  {true_a2v:>8}")
    print(f"  {'Precision ratio':40s}  {precision_dsp:>7.1%}  {precision_a2v:>7.1%}")
    print(f"  {'─'*56}")
    print(f"  {'Shared  |Δt| < {:.0f}s'.format(window_sec):40s}  {shared:>8}")
    print(f"  {'DSP only':40s}  {dsp_only:>8}")
    print(f"  {'A2V only':40s}  {a2v_only:>8}")
    print(sep2)
    print()
    if shared == 0:
        print("  ℹ️  No events in common — try increasing --window.")
    elif shared == true_dsp and shared == true_a2v:
        print("  🎯 Perfect overlap: all validated events are shared.")
    else:
        if dsp_only > 0:
            print(f"  🔵 {dsp_only} event(s) detected only by the DSP pipeline.")
        if a2v_only > 0:
            print(f"  🟠 {a2v_only} event(s) detected only by animal2vec.")
        if shared > 0:
            print(f"  🟢 {shared} event(s) confirmed by both pipelines.")
    print()

    return dict(
        tot_dsp=tot_dsp,
        true_dsp=true_dsp,
        precision_dsp=precision_dsp,
        tot_a2v=tot_a2v,
        true_a2v=true_a2v,
        precision_a2v=precision_a2v,
        shared=shared,
        dsp_only=dsp_only,
        a2v_only=a2v_only,
        window_sec=window_sec,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare DSP and animal2vec whoop detection pipelines."
    )
    parser.add_argument("dsp_db_path", help="Path to the (validated) DSP HDF5 database")
    parser.add_argument("a2v_db_path", help="Path to the (validated) animal2vec HDF5 database")
    parser.add_argument(
        "--window", type=float, default=30.0, metavar="SEC",
        help="Time threshold in seconds for a shared match (default: 30)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print details for every candidate and every match",
    )
    args = parser.parse_args()

    compare_pipelines(
        dsp_db_path=args.dsp_db_path,
        a2v_db_path=args.a2v_db_path,
        window_sec=args.window,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()


# ── usage examples ────────────────────────────────────────────────────────────
# python comparison_btw_pipelines.py Z:\postprocessing_results\2026-06-01\results_2026-06-01.h5 Z:\postprocessing_results_animal2vec\2026-06-01\results_2026-06-01.h5 # default 30s window, compact output without verbose
# python comparison_btw_pipelines.py Z:\postprocessing_results\2026-06-01\results_2026-06-01.h5 Z:\postprocessing_results_animal2vec\2026-06-01\results_2026-06-01.h5 --window 15 --verbose
# python comparison_btw_pipelines.py Z:\postprocessing_results\2026-06-01\results_2026-06-01.h5 Z:\postprocessing_results_animal2vec\2026-06-01\results_2026-06-01.h5 --window 60 --verbose