#!/usr/bin/env python3
"""
extract_plots_from_h5_db.py

DESCRIPTION:
    Analyzes multiple days of bioacoustic signals from a root folder
    containing daily subfolders, each with a results_<YYYY-MM-DD>.h5 file.

    For each day it plots:
     - F0 distribution
     - duration distribution
     - channels involved distribution (with multiside counter)
     - localization heatmap for side A and side B
     - daily stop-signal timeline (15 min histogram + rug)

    At the end it also generates aggregated multi-day plots, including:
     - aggregated F0 / duration / channels distributions
     - aggregated localization heatmaps
     - aggregated "typical day" timeline (all days folded onto one 24h axis)
     - aggregated continuous multi-day timeline
     - aggregated scatter plot: precise duration vs mean F0

USAGE:
    python extract_plots_from_h5_db.py <input_root> --output-dir <output_dir> [--show] [--mic-coords <json_path>]

ARGUMENTS:
    input_root        Root folder containing daily subfolders (each with a
                       results_<YYYY-MM-DD>.h5 file)
    --output-dir       Root output folder for generated plots (required)
    --show              Show plots in an interactive window (in addition to
                        saving them to disk)
    --mic-coords        JSON file with microphone coordinates
                        (default: mic_coordinates.json next to this script)

EXAMPLE:
    python extract_plots_from_h5_db.py Z:\postprocessing_results --output-dir Z:\postprocessing_results\plots
    python extract_plots_from_h5_db.py Z:\postprocessing_results --output-dir Z:\postprocessing_results\plots --show
    python extract_plots_from_h5_db.py Z:\postprocessing_results --output-dir Z:\postprocessing_results\plots --mic-coords C:\configs\mic_coordinates.json

OUTPUT:
    Creates per-metric subfolders under --output-dir (frequency, duration,
    channels, heatmaps_sideA/B, daily_timelines, aggregated timelines,
    scatter_plots) and saves one PDF plot per day plus aggregated PDFs
    summarizing all days combined.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates


_DEFAULT_MIC_COORDS = Path(__file__).parent / "mic_coordinates.json"
_SIDE_A_RANGE = range(0, 16)
_SIDE_B_RANGE = range(16, 32)
_RAW_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})\.(\d+)Z")


def _is_sound_validated(grp: h5py.Group) -> bool:
    """Return True if the group's 'sound_validated' field evaluates to True, handling multiple stored types (bool, bytes, str, int)."""
    if "sound_validated" not in grp:
        return False
    raw = grp["sound_validated"][()]
    if isinstance(raw, np.bool_):
        return bool(raw)
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode().strip() == "True"
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("b'"):
            s = s[2:-1]
        return s == "True"
    try:
        return bool(int(raw))
    except (ValueError, TypeError):
        pass
    return str(raw).strip() in ("True", "1")


def _parse_raw_name(raw_name: str) -> datetime | None:
    """Extract a UTC datetime from a raw recording filename (e.g. containing '2026-05-27T14_23_05.123Z'). Returns None if no match."""
    if isinstance(raw_name, (bytes, np.bytes_)):
        raw_name = raw_name.decode()
    m = _RAW_NAME_RE.search(str(raw_name))
    if not m:
        return None
    date_str, hh, mm, ss, frac = m.groups()
    frac = (frac + "000000")[:6]
    dt_str = f"{date_str} {hh}:{mm}:{ss}.{frac}"
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)


def _load_mic_coords(json_path: Path) -> dict:
    """Load microphone pixel coordinates (cam0/cam1) from a JSON file into a {channel_index: (x, y)} dict."""
    with open(json_path) as fh:
        data = json.load(fh)
    coords: dict[int, tuple[int, int]] = {}
    for cam_key in ("cam0", "cam1"):
        for pt in data[cam_key]["microphone_points"]:
            coords[pt["id"] - 1] = (pt["x"], pt["y"])
    return coords


def _extract_date_from_name(path_like: str) -> str | None:
    """Extract a YYYY-MM-DD date substring from a path or filename, if present."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(path_like))
    return m.group(1) if m else None


def _build_output_dirs(root: Path) -> dict[str, Path]:
    """Create (if missing) and return the mapping of output subfolder names to their Path, one per plot category."""
    dirs = {
        "frequency": root / "frequency_distributions",
        "duration": root / "duration_distributions",
        "channels": root / "channels_distributions",
        "heatmap_a": root / "heatmaps_sideA",
        "heatmap_b": root / "heatmaps_sideB",
        "timeline_daily": root / "daily_timelines",
        "timeline_typical": root / "aggregated_typical_day_timelines",
        "timeline_continuous": root / "aggregated_continuous_timelines",
        "scatter": root / "scatter_plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the multi-day H5 analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze multiple .h5 databases in daily subfolders."
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Root folder containing daily subfolders"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root output folder"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots in an interactive window"
    )
    parser.add_argument(
        "--mic-coords",
        type=Path,
        default=_DEFAULT_MIC_COORDS,
        help=f"JSON file with microphone coordinates (default: {_DEFAULT_MIC_COORDS})"
    )
    return parser.parse_args()


def plot_histogram(values: list,
                   title: str,
                   xlabel: str,
                   output_path: Path | None = None,
                   show: bool = False,
                   color: str = "#2196F3",
                   discrete: bool = False,
                   bins=None) -> None:
    """
    Plot a histogram of `values` with a median line and a stats box
    (n, mean, std, min, max). Supports discrete integer binning
    (one bar per integer value) or continuous binning.
    """
    if not values:
        print(f"[WARN] No data for '{title}', skipping.")
        return

    arr = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))

    if discrete:
        # one bin per integer value, centered on the tick
        arr_int = arr.astype(int)
        min_v = int(arr_int.min())
        max_v = int(arr_int.max())
        hist_bins = np.arange(min_v - 0.5, max_v + 1.5, 1)
        ax.hist(
            arr_int,
            bins=hist_bins,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            rwidth=0.85
        )
        ax.set_xticks(np.arange(min_v, max_v + 1, 1))
    else:
        hist_bins = bins if bins is not None else min(max(int(np.sqrt(len(arr)) * 2), 20), 80)
        ax.hist(
            arr,
            bins=hist_bins,
            color=color,
            edgecolor="white",
            linewidth=0.6
        )

    median_val = np.median(arr)
    ax.axvline(
        median_val,
        color="#ce4b1c",
        linewidth=1.5,
        linestyle="--",
        label=f"Median: {median_val:.3g}"
    )

    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel("Count", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.legend(fontsize=15, loc="best")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    stats_text = (
        f"n={len(arr)}\n"
        f"mean={arr.mean():.3g}\n"
        f"std={arr.std():.3g}\n"
        f"min={arr.min():.3g}\n"
        f"max={arr.max():.3g}"
    )
    ax.text(
        0.97, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=15,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.8,
            edgecolor="#cccccc"
        )
    )

    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
        print(f"[OK] Plot saved: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_localization_heatmap(hit_counts: np.ndarray,
                              mic_coords_side: dict[int, tuple[int, int]],
                              title: str,
                              output_path: Path | None = None,
                              show: bool = False,
                              cmap: str = "viridis") -> None:
    """
    Plot a hexagonal-tile heatmap of stop-signal hit counts over the
    microphone array layout for one side (A or B). Hexagon size is
    auto-derived from the median nearest-neighbor distance between mics.
    """
    if hit_counts.sum() == 0:
        print(f"[WARN] No hits for '{title}', skipping heatmap.")
        return

    from matplotlib.patches import RegularPolygon
    from matplotlib.collections import PatchCollection
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    from scipy.spatial import KDTree

    sorted_chs = sorted(mic_coords_side)
    xs = np.array([mic_coords_side[ch][0] for ch in sorted_chs], dtype=float)
    ys = np.array([mic_coords_side[ch][1] for ch in sorted_chs], dtype=float)
    zs = np.array([hit_counts[i] for i in range(len(sorted_chs))], dtype=float)

    HEX_RADIUS_FACTOR = 0.52

    # auto-size hexagons from the median nearest-neighbor distance between mics
    tree = KDTree(np.column_stack([xs, ys]))
    dists, _ = tree.query(np.column_stack([xs, ys]), k=2)
    median_nn = np.median(dists[:, 1])
    hex_radius = median_nn * HEX_RADIUS_FACTOR

    norm = mcolors.Normalize(vmin=0, vmax=max(zs.max(), 1))
    cmap_obj = mcm.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(10, 7))

    patches = []
    for x, y in zip(xs, ys):
        patches.append(
            RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                orientation=0,
                linewidth=0.8,
                edgecolor="#333333",
            )
        )

    col = PatchCollection(
        patches,
        cmap=cmap_obj,
        norm=norm,
        match_original=False,
        linewidth=0.8,
        edgecolor="#333333"
    )
    col.set_array(zs)
    ax.add_collection(col)

    # label each hexagon with its channel number and hit count
    for ch, x, y, count in zip(sorted_chs, xs, ys, zs):
        label_color = "white" if norm(count) < 0.45 else "black"
        ax.text(
            x, y,
            "ch" + str(ch) + chr(10) + "(" + str(int(count)) + ")",
            fontsize=12,
            ha="center",
            va="center",
            color=label_color,
            fontweight="bold",
            zorder=3
        )

    pad = hex_radius * 1.4
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85)
    cbar.set_label("Stop signal intensity (hit count)", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel("X (px)", fontsize=25)
    ax.set_ylabel("Y (px)", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
        print("[OK] Heatmap saved: " + str(output_path))
    if show:
        plt.show()
    plt.close(fig)


def plot_daily_timeline(abs_times_utc: list[datetime],
                        day_label: str,
                        output_path: Path | None = None,
                        show: bool = False) -> None:
    """
    Plot a 24h timeline for a single day: a 15-minute-bin histogram on top
    and an event rug plot (one tick per event) below, both on a shared
    UTC time axis.
    """
    if not abs_times_utc:
        print(f"[WARN] No timestamped events for '{day_label}', skipping timeline.")
        return

    day = abs_times_utc[0].date()
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(hours=24)
    dt_values = np.array([mdates.date2num(t) for t in abs_times_utc])
    x_start = mdates.date2num(day_start)
    x_end = mdates.date2num(day_end)

    fig, (ax_main, ax_rug) = plt.subplots(
        2, 1,
        figsize=(14, 6),
        gridspec_kw={"height_ratios": [6, 1], "hspace": 0.08},
        sharex=True
    )

    # 96 bins over 24h → 15-minute resolution
    bin_edges = np.linspace(x_start, x_end, 97)
    counts, edges = np.histogram(dt_values, bins=bin_edges)
    bin_width = edges[1] - edges[0]

    ax_main.bar(
        edges[:-1], counts,
        width=bin_width * 0.92,
        align="edge",
        color="#83bfea",
        alpha=0.72,
        label="Count per 15 min"
    )
    ax_main.set_xlim(x_start, x_end)
    ax_main.set_ylim(bottom=0)
    ax_main.set_ylabel("No. of stop signals", fontsize=25)
    ax_main.tick_params(axis="both", which="major", labelsize=20)
    ax_main.legend(fontsize=15, loc="upper left")
    ax_main.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_main.grid(axis="x", linestyle="--", linewidth=0.35, alpha=0.45)
    ax_main.spines[["top", "right"]].set_visible(False)

    ax_rug.eventplot(
        dt_values,
        orientation="horizontal",
        lineoffsets=0.5,
        linelengths=0.8,
        linewidths=0.6,
        color="#83bfea",
        alpha=0.55
    )
    ax_rug.set_ylim(0, 1)
    ax_rug.set_yticks([])
    ax_rug.set_ylabel("events", fontsize=20, labelpad=2)
    ax_rug.spines[["top", "right", "left"]].set_visible(False)
    ax_rug.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax_rug.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_rug.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
    plt.setp(ax_rug.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=20)
    ax_rug.set_xlabel("UTC time", fontsize=25)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
        print(f"[OK] Timeline saved: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_aggregated_typical_day_timeline(abs_times_utc: list[datetime],
                                         prefix: str,
                                         output_path: Path | None = None,
                                         show: bool = False) -> None:
    """
    Fold all event timestamps (regardless of their actual date) onto a
    single reference 24h day, then reuse plot_daily_timeline() to show the
    aggregated "typical day" activity pattern across the whole date range.
    """
    if not abs_times_utc:
        print("[WARN] No timestamps for aggregated typical-day timeline.")
        return

    ref_day = datetime(2000, 1, 1, tzinfo=timezone.utc)
    folded = [
        ref_day.replace(
            hour=t.hour,
            minute=t.minute,
            second=t.second,
            microsecond=t.microsecond
        )
        for t in abs_times_utc
    ]
    plot_daily_timeline(folded, f"aggregated typical day {prefix}", output_path=output_path, show=show)


def plot_aggregated_continuous_timeline(abs_times_utc: list[datetime],
                                        prefix: str,
                                        output_path: Path | None = None,
                                        show: bool = False) -> None:
    """
    Plot a single continuous timeline spanning the entire multi-day date
    range: a histogram (15-min resolution) on top and an event rug plot
    below, with date-aware tick formatting.
    """
    if not abs_times_utc:
        print("[WARN] No timestamps for aggregated continuous timeline.")
        return

    abs_times_utc = sorted(abs_times_utc)
    dt_values = np.array([mdates.date2num(t) for t in abs_times_utc])

    fig, (ax_main, ax_rug) = plt.subplots(
        2, 1,
        figsize=(16, 6),
        gridspec_kw={"height_ratios": [6, 1], "hspace": 0.08},
        sharex=True
    )

    # 15-minute bins across the whole date range (min 48 bins)
    total_days = max((abs_times_utc[-1] - abs_times_utc[0]).days + 1, 1)
    n_bins = max(total_days * 24 * 4, 48)
    counts, edges = np.histogram(dt_values, bins=n_bins)
    bin_width = edges[1] - edges[0]

    ax_main.bar(
        edges[:-1], counts,
        width=bin_width * 0.92,
        align="edge",
        color="#6A1B9A",
        alpha=0.7,
        label="Count over time"
    )
    ax_main.set_ylabel("No. of stop signals", fontsize=11)
    ax_main.set_title(
        f"Continuous multi-day timeline — {prefix}  (n={len(abs_times_utc)} events, UTC)",
        fontsize=13,
        fontweight="bold",
        pad=10
    )
    ax_main.legend(fontsize=10, loc="upper left")
    ax_main.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_main.grid(axis="x", linestyle="--", linewidth=0.35, alpha=0.45)
    ax_main.spines[["top", "right"]].set_visible(False)

    ax_rug.eventplot(
        dt_values,
        orientation="horizontal",
        lineoffsets=0.5,
        linelengths=0.8,
        linewidths=0.5,
        color="#6A1B9A",
        alpha=0.5
    )
    ax_rug.set_ylim(0, 1)
    ax_rug.set_yticks([])
    ax_rug.spines[["top", "right", "left"]].set_visible(False)

    locator = mdates.AutoDateLocator(minticks=10, maxticks=18)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_rug.xaxis.set_major_locator(locator)
    ax_rug.xaxis.set_major_formatter(formatter)
    plt.setp(ax_rug.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax_rug.set_xlabel("UTC time", fontsize=11)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
        print(f"[OK] Continuous timeline saved: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_duration_vs_f0_scatter(duration_values: list[float],
                                f0_values: list[float],
                                title: str,
                                output_path: Path | None = None,
                                show: bool = False) -> None:
    """Plot a scatter of precise duration vs mean F0 across all aggregated validated candidates."""
    if not duration_values or not f0_values:
        print(f"[WARN] No data for '{title}', skipping scatter.")
        return

    x = np.array(duration_values, dtype=float)
    y = np.array(f0_values, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        x, y,
        s=42,
        alpha=0.65,
        color="#83bfea",
        edgecolors="white",
        linewidths=0.5
    )

    ax.set_xlabel("Precise duration [s]", fontsize=25)
    ax.set_ylabel("Mean F0 [Hz]", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
        print(f"[OK] Scatter plot saved: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def analyse_day(h5_path: Path,
                output_dirs: dict[str, Path],
                mic_coords_path: Path,
                show: bool = False,
                prefix: str | None = None) -> dict:
    """
    Load a single day's H5 database, extract sound_validated=True candidates
    (F0, duration, involved channels, absolute timestamps), generate all
    per-day plots, and return a dict of the raw values plus per-day
    counters for later aggregation across days.
    """
    if not h5_path.is_file():
        print(f"[ERROR] File not found: {h5_path}", file=sys.stderr)
        return {
            "f0_values": [],
            "duration_values": [],
            "n_channels_values": [],
            "abs_times_utc": [],
            "hits_side_a": np.zeros(16, dtype=int),
            "hits_side_b": np.zeros(16, dtype=int),
            "multiside_candidates_counter": 0,
            "validated_count": 0,
            "skipped_count": 0,
            "ts_parse_errors": 0,
            "coords_side_a": {},
            "coords_side_b": {},
            "day_label": None,
        }

    if not mic_coords_path.is_file():
        print(f"[ERROR] Coordinates JSON not found: {mic_coords_path}", file=sys.stderr)
        sys.exit(1)

    mic_coords = _load_mic_coords(mic_coords_path)
    coords_side_a = {ch: mic_coords[ch] for ch in range(0, 16) if ch in mic_coords}
    coords_side_b = {ch: mic_coords[ch] for ch in range(16, 32) if ch in mic_coords}

    f0_values: list[float] = []
    duration_values: list[float] = []
    n_channels_values: list[int] = []
    abs_times_utc: list[datetime] = []
    hits_side_a = np.zeros(16, dtype=int)
    hits_side_b = np.zeros(16, dtype=int)
    multiside_candidates_counter = 0
    validated_count = 0
    skipped_count = 0
    ts_parse_errors = 0

    print(f"[INFO] Opening: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        group_names = sorted(f.keys())
        print(f"[INFO] Total groups found: {len(group_names)}")
        for gname in group_names:
            grp = f[gname]
            if not _is_sound_validated(grp):
                skipped_count += 1
                continue

            validated_count += 1
            try:
                raw_name = grp.attrs.get("raw_name", "")
                ch = int(grp["ch"][()])
                peak_time = float(grp["peak_time"][()])
                f0_mean = float(grp["f0_mean"][()])
                precise_duration = float(grp["precise_duration"][()])
                ch_involved_raw = grp["channels_involved"][()].tolist()
            except KeyError as e:
                print(f"[WARN] Group '{gname}' — missing field {e}, skipping.", file=sys.stderr)
                skipped_count += 1
                validated_count -= 1
                continue

            ch_involved_unique = list(dict.fromkeys(ch_involved_raw))
            n_ch = len(ch_involved_unique)
            has_side_a = any(c in _SIDE_A_RANGE for c in ch_involved_unique)
            has_side_b = any(c in _SIDE_B_RANGE for c in ch_involved_unique)

            if has_side_a and has_side_b:
                multiside_candidates_counter += 1

            if ch in _SIDE_A_RANGE:
                hits_side_a[ch] += 1
            elif ch in _SIDE_B_RANGE:
                hits_side_b[ch - 16] += 1

            file_dt = _parse_raw_name(raw_name)
            if file_dt is not None:
                abs_times_utc.append(file_dt + timedelta(seconds=peak_time))
            else:
                ts_parse_errors += 1

            f0_values.append(f0_mean)
            duration_values.append(precise_duration)
            n_channels_values.append(n_ch)

    print(f"[INFO] Processing completed — validated: {validated_count}, skipped/non-validated: {skipped_count}")

    day_label = abs_times_utc[0].strftime("%Y-%m-%d") if abs_times_utc else (
        prefix or _extract_date_from_name(h5_path.as_posix()) or h5_path.parent.name
    )

    if validated_count == 0:
        print("[WARN] No groups with sound_validated=True found.", file=sys.stderr)
        return {
            "f0_values": f0_values,
            "duration_values": duration_values,
            "n_channels_values": n_channels_values,
            "abs_times_utc": abs_times_utc,
            "hits_side_a": hits_side_a,
            "hits_side_b": hits_side_b,
            "multiside_candidates_counter": multiside_candidates_counter,
            "validated_count": validated_count,
            "skipped_count": skipped_count,
            "ts_parse_errors": ts_parse_errors,
            "coords_side_a": coords_side_a,
            "coords_side_b": coords_side_b,
            "day_label": day_label,
        }

    prefix = prefix or day_label
    multiside_note = f"  [multiside: {multiside_candidates_counter}/{validated_count}]"

    # 5 ms-wide duration bins, aligned to the data range for this day
    duration_bins_day = np.arange(
        np.floor(min(duration_values) / 0.005) * 0.005,
        np.ceil(max(duration_values) / 0.005) * 0.005 + 0.005,
        0.005
    )

    plot_histogram(
        f0_values,
        "F0 distribution (fundamental frequency)",
        "Mean F0 [Hz]",
        output_dirs["frequency"] / f"{prefix}_f0_distribution.pdf",
        show,
        "#83bfea",
        False
    )

    plot_histogram(
        duration_values,
        "Signal duration distribution",
        "Precise duration [s]",
        output_dirs["duration"] / f"{prefix}_duration_distribution.pdf",
        show,
        "#83bfea",
        False,
        bins=duration_bins_day
    )

    plot_histogram(
        n_channels_values,
        f"Distribution of No. of involved channels{multiside_note}",
        "No. of involved channels",
        output_dirs["channels"] / f"{prefix}_channels_distribution.pdf",
        show,
        "#83bfea",
        True
    )

    plot_localization_heatmap(
        hits_side_a,
        coords_side_a,
        f"Stop signal heatmap — Side A (ch 0–15)  [n={hits_side_a.sum()}]",
        output_dirs["heatmap_a"] / f"{prefix}_heatmap_sideA.pdf",
        show,
        "viridis"
    )

    plot_localization_heatmap(
        hits_side_b,
        coords_side_b,
        f"Stop signal heatmap — Side B (ch 16–31)  [n={hits_side_b.sum()}]",
        output_dirs["heatmap_b"] / f"{prefix}_heatmap_sideB.pdf",
        show,
        "viridis"
    )

    if abs_times_utc:
        plot_daily_timeline(
            abs_times_utc,
            day_label,
            output_dirs["timeline_daily"] / f"{prefix}_daily_timeline.pdf",
            show
        )
    else:
        print("[WARN] No valid timestamps available, skipping timeline.")

    print(f"[DONE] Daily plots saved for: {prefix}")
    return {
        "f0_values": f0_values,
        "duration_values": duration_values,
        "n_channels_values": n_channels_values,
        "abs_times_utc": abs_times_utc,
        "hits_side_a": hits_side_a,
        "hits_side_b": hits_side_b,
        "multiside_candidates_counter": multiside_candidates_counter,
        "validated_count": validated_count,
        "skipped_count": skipped_count,
        "ts_parse_errors": ts_parse_errors,
        "coords_side_a": coords_side_a,
        "coords_side_b": coords_side_b,
        "day_label": day_label,
    }


def analyse_multiple_days(input_root: Path,
                          output_root: Path,
                          mic_coords_path: Path,
                          show: bool = False) -> dict:
    """
    Iterate over every daily subfolder in input_root, run analyse_day() on
    each results_<date>.h5 file found, accumulate all values across days,
    then generate the aggregated multi-day plots and return a summary dict.
    """
    if not input_root.is_dir():
        print(f"[ERROR] Input folder not found: {input_root}", file=sys.stderr)
        sys.exit(1)

    output_dirs = _build_output_dirs(output_root)
    day_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])

    if not day_dirs:
        print(f"[ERROR] No daily subfolders found in: {input_root}", file=sys.stderr)
        sys.exit(1)

    all_f0_values: list[float] = []
    all_duration_values: list[float] = []
    all_n_channels_values: list[int] = []
    all_abs_times_utc: list[datetime] = []
    all_hits_side_a = np.zeros(16, dtype=int)
    all_hits_side_b = np.zeros(16, dtype=int)
    total_multiside_candidates_counter = 0
    total_validated_count = 0
    total_skipped_count = 0
    total_ts_parse_errors = 0
    analysed_days: list[str] = []
    coords_side_a = None
    coords_side_b = None

    for day_dir in day_dirs:
        day_prefix = _extract_date_from_name(day_dir.name) or day_dir.name
        h5_path = day_dir / f"results_{day_prefix}.h5"

        if not h5_path.is_file():
            print(f"[WARN] Expected file not found, skipping: {h5_path}")
            continue

        results = analyse_day(
            h5_path=h5_path,
            output_dirs=output_dirs,
            mic_coords_path=mic_coords_path,
            show=show,
            prefix=day_prefix,
        )

        all_f0_values.extend(results["f0_values"])
        all_duration_values.extend(results["duration_values"])
        all_n_channels_values.extend(results["n_channels_values"])
        all_abs_times_utc.extend(results["abs_times_utc"])
        all_hits_side_a += results["hits_side_a"]
        all_hits_side_b += results["hits_side_b"]
        total_multiside_candidates_counter += results["multiside_candidates_counter"]
        total_validated_count += results["validated_count"]
        total_skipped_count += results["skipped_count"]
        total_ts_parse_errors += results["ts_parse_errors"]
        coords_side_a = results["coords_side_a"]
        coords_side_b = results["coords_side_b"]

        if results["day_label"]:
            analysed_days.append(results["day_label"])

    if not analysed_days:
        print("[ERROR] No day was successfully analysed.", file=sys.stderr)
        sys.exit(1)

    analysed_days = sorted(set(analysed_days))
    global_prefix = f"{analysed_days[0]}--{analysed_days[-1]}"
    multiside_note = f"  [multiside: {total_multiside_candidates_counter}]"

    # 5 ms-wide duration bins, aligned to the full aggregated data range
    duration_bins_all = np.arange(
        np.floor(min(all_duration_values) / 0.005) * 0.005,
        np.ceil(max(all_duration_values) / 0.005) * 0.005 + 0.005,
        0.005
    )

    plot_histogram(
        all_f0_values,
        "Aggregated F0 distribution (fundamental frequency)",
        "F0 [Hz]",
        output_dirs["frequency"] / f"{global_prefix}_f0_distribution.pdf",
        show,
        "#83bfea",
        False
    )

    plot_histogram(
        all_duration_values,
        "Aggregated signal duration distribution",
        "Precise duration [s]",
        output_dirs["duration"] / f"{global_prefix}_duration_distribution.pdf",
        show,
        "#83bfea",
        False,
        bins=duration_bins_all
    )

    plot_histogram(
        all_n_channels_values,
        f"Aggregated distribution of No. of involved channels{multiside_note}",
        "No. of involved channels",
        output_dirs["channels"] / f"{global_prefix}_channels_distribution.pdf",
        show,
        "#83bfea",
        True
    )

    if coords_side_a is not None and coords_side_b is not None:
        plot_localization_heatmap(
            all_hits_side_a,
            coords_side_a,
            f"Aggregated spatial distribution — Side A (n={all_hits_side_a.sum()})",
            output_dirs["heatmap_a"] / f"{global_prefix}_heatmap_sideA.pdf",
            show,
            "viridis"
        )
        plot_localization_heatmap(
            all_hits_side_b,
            coords_side_b,
            f"Aggregated spatial distribution — Side B (n={all_hits_side_b.sum()})",
            output_dirs["heatmap_b"] / f"{global_prefix}_heatmap_sideB.pdf",
            show,
            "viridis"
        )

    plot_aggregated_typical_day_timeline(
        all_abs_times_utc,
        global_prefix,
        output_dirs["timeline_typical"] / f"{global_prefix}_typical_day_timeline.pdf",
        show
    )

    plot_aggregated_continuous_timeline(
        all_abs_times_utc,
        global_prefix,
        output_dirs["timeline_continuous"] / f"{global_prefix}_continuous_timeline.pdf",
        show
    )

    plot_duration_vs_f0_scatter(
        all_duration_values,
        all_f0_values,
        "Aggregated duration vs F0",
        output_dirs["scatter"] / f"{global_prefix}_duration_vs_f0_scatter.pdf",
        show
    )

    print(f"[DONE] Multi-day analysis completed: {global_prefix}")
    return {
        "all_f0_values": all_f0_values,
        "all_duration_values": all_duration_values,
        "all_n_channels_values": all_n_channels_values,
        "all_abs_times_utc": all_abs_times_utc,
        "all_hits_side_a": all_hits_side_a,
        "all_hits_side_b": all_hits_side_b,
        "total_multiside_candidates_counter": total_multiside_candidates_counter,
        "total_validated_count": total_validated_count,
        "total_skipped_count": total_skipped_count,
        "total_ts_parse_errors": total_ts_parse_errors,
        "analysed_days": analysed_days,
        "global_prefix": global_prefix,
        "output_dirs": output_dirs,
    }


def main() -> None:
    """Entry point: parse CLI args, run the multi-day analysis, and unpack the results (used for downstream inspection/debugging)."""
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_dir.resolve()
    mic_coords_path = args.mic_coords.resolve()

    results = analyse_multiple_days(
        input_root=input_root,
        output_root=output_root,
        mic_coords_path=mic_coords_path,
        show=args.show,
    )

    all_f0_values = results["all_f0_values"]
    all_duration_values = results["all_duration_values"]
    all_n_channels_values = results["all_n_channels_values"]
    all_abs_times_utc = results["all_abs_times_utc"]
    all_hits_side_a = results["all_hits_side_a"]
    all_hits_side_b = results["all_hits_side_b"]
    total_multiside_candidates_counter = results["total_multiside_candidates_counter"]
    total_validated_count = results["total_validated_count"]
    total_skipped_count = results["total_skipped_count"]
    total_ts_parse_errors = results["total_ts_parse_errors"]
    analysed_days = results["analysed_days"]
    global_prefix = results["global_prefix"]
    output_dirs = results["output_dirs"]

    _ = (
        all_f0_values, all_duration_values, all_n_channels_values,
        all_abs_times_utc, all_hits_side_a, all_hits_side_b,
        total_multiside_candidates_counter, total_validated_count,
        total_skipped_count, total_ts_parse_errors,
        analysed_days, global_prefix, output_dirs,
    )


if __name__ == "__main__":
    main()
