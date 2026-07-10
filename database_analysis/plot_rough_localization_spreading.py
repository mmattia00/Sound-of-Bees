#!/usr/bin/env python3
"""
plot_rough_localization_spreading.py

DESCRIPTION:
    Plays back the audio segments of sound_validated=True candidates in
    sequence, computes the HNR (Harmonics-to-Noise Ratio) on each involved
    channel's segment, and plots an HNR heatmap for side A (ch 0-15) and
    side B (ch 16-31) for each candidate.

USAGE:
    python plot_rough_localization_spreading.py <h5_path> <audio_folder> --mic-coords <mic_coords_json> [--padding <seconds>]

ARGUMENTS:
    h5_path         Path to the HDF5 candidates database
    audio_folder    Folder with the raw multichannel WAV recordings
    --mic-coords    Path to the JSON file with microphone coordinates (required)
    --padding       Seconds of padding before/after the segment (default: 0.3)

EXAMPLE:
    python plot_rough_localization_spreading.py Z:\\postprocessing_results\\2026-06-09\\results_2026-06-09.h5 Z:\\recordings2026\\audio\\2026-06-09 --mic-coords Z:\\Sound-of-Bees\\postprocessing\\calibration.json

NOTE:
    Actual audio playback (sd.play/sd.wait) is currently commented out in
    the main loop — only HNR computation and heatmap plotting are active.
    Only candidates with more than 5 involved channels are processed.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial import KDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from classes.whoop_detector import WhoopDetector


PADDING_SEC = 0.3
_SIDE_A_RANGE = range(0, 16)
_SIDE_B_RANGE = range(16, 32)
GREY_NEUTRAL = "#888888"


HNR_DETECTOR_PARAMS = dict(
    window_length_ms=50, hop_length_ms=10,
    f0_min=250, f0_max=700, window_type='hamming',
    lowpass_cutoff=15000, highpass_cutoff=2500,
    normalize=True, target_rms=0.1,
)
HNR_DETECT_PARAMS = dict(
    percentile=80, offset=4, window_sec=0.5, merge_overlaps=True,
)


def is_sound_validated(grp) -> bool:
    """Return True if the group's 'sound_validated' field is truthy (handles bool, string, and byte-string representations)."""
    if "sound_validated" not in grp:
        return False
    try:
        val = grp["sound_validated"][()]
        return val is True or str(val) in ("True", "b'True'", "1")
    except Exception:
        return False


def compute_hnr_on_segment(segment: np.ndarray, sr: int) -> float:
    """Run WhoopDetector on a single-channel audio segment and return the max peak HNR value found (NaN if no peak detected or on error)."""
    try:
        detector = WhoopDetector(signal=segment, sr=sr, **HNR_DETECTOR_PARAMS)
        detector.detect(**HNR_DETECT_PARAMS)
        peak_infos = detector.get_peak_info()
        if not peak_infos:
            return np.nan
        return max(info['peak_hnr_value'] for info in peak_infos)
    except Exception as e:
        print(f"      [WARN] HNR computation failed: {e}")
        return np.nan


def load_mic_coords(json_path: str) -> dict:
    """Load microphone pixel coordinates (cam0/cam1) from a JSON file into a {channel_index: (x, y)} dict."""
    with open(json_path) as fh:
        data = json.load(fh)
    coords: dict[int, tuple[int, int]] = {}
    for cam_key in ("cam0", "cam1"):
        for pt in data[cam_key]["microphone_points"]:
            coords[pt["id"] - 1] = (pt["x"], pt["y"])
    return coords


def plot_candidate_hnr_heatmap(hnr_per_ch: dict[int, float],
                                mic_coords_side: dict[int, tuple[int, int]],
                                title: str,
                                output_path: str = None,
                                show: bool = True,
                                cmap: str = "viridis",
                                vmin: float = None,
                                vmax: float = None,
                                mirror_x: bool = False) -> None:
    """
    Hexagonal per-side heatmap, color = candidate's HNR per channel.
    Channels not involved (absent from hnr_per_ch) -> neutral grey.

    vmin/vmax: if provided, used for color normalization (shared scale
               between side A and side B). If None, computed locally from
               this side's data.
    mirror_x: if True, mirrors the X coordinates (for side B), so the
              geometry appears reflected relative to side A.
    """
    if not mic_coords_side:
        print(f"[WARN] No coordinates for '{title}', skipping heatmap.")
        return

    sorted_chs = sorted(mic_coords_side)
    xs = np.array([mic_coords_side[ch][0] for ch in sorted_chs], dtype=float)
    ys = np.array([mic_coords_side[ch][1] for ch in sorted_chs], dtype=float)
    zs = np.array([hnr_per_ch.get(ch, np.nan) for ch in sorted_chs], dtype=float)

    valid_mask = ~np.isnan(zs)
    if valid_mask.sum() == 0:
        print(f"[WARN] No involved channels in '{title}', skipping heatmap.")
        return

    # mirroring: reflects X coordinates about their own center axis
    if mirror_x:
        xs = xs.max() + xs.min() - xs

    HEX_RADIUS_FACTOR = 0.52
    tree = KDTree(np.column_stack([xs, ys]))
    dists, _ = tree.query(np.column_stack([xs, ys]), k=2)
    median_nn = np.median(dists[:, 1])
    hex_radius = median_nn * HEX_RADIUS_FACTOR

    # use provided vmin/vmax (shared scale) if available, otherwise compute locally
    if vmin is None or vmax is None:
        vmin_local = float(np.nanmin(zs))
        vmax_local = float(np.nanmax(zs))
        vmin = vmin_local if vmin is None else vmin
        vmax = vmax_local if vmax is None else vmax
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = mcm.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(10, 7))

    patches = []
    colors = []
    for x, y, z in zip(xs, ys, zs):
        patches.append(
            RegularPolygon((x, y), numVertices=6, radius=hex_radius,
                           orientation=0, linewidth=0.8, edgecolor="#333333")
        )
        colors.append(GREY_NEUTRAL if np.isnan(z) else cmap_obj(norm(z)))

    col = PatchCollection(patches, match_original=False,
                          linewidth=0.8, edgecolor="#333333")
    col.set_facecolor(colors)
    ax.add_collection(col)

    for ch, x, y, z in zip(sorted_chs, xs, ys, zs):
        if np.isnan(z):
            label = f"ch{ch}\n(--)"
            label_color = "white"
        else:
            label = f"ch{ch}\n({z:.1f})"
            label_color = "white" if norm(z) < 0.45 else "black"
        ax.text(x, y, label, fontsize=12, ha="center", va="center",
                color=label_color, fontweight="bold", zorder=3)

    pad = hex_radius * 1.4
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85)
    cbar.set_label("HNR (dB)", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel("X (pixel)", fontsize=25)
    ax.set_ylabel("Y (pixel)", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180)
        print(f"[OK] Heatmap saved: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def playback_sound_validated(db_path: str, audio_base_path: str,
                              mic_coords_path: str,
                              padding_sec: float = PADDING_SEC) -> None:
    """
    Open the H5 candidates database, iterate over all sound_validated=True
    groups with more than 5 involved channels, load the corresponding raw
    audio segment (with padding), compute HNR per channel, and plot the
    side A / side B HNR heatmaps for each candidate.
    """
    mic_coords = load_mic_coords(mic_coords_path)
    coords_side_a = {ch: mic_coords[ch] for ch in _SIDE_A_RANGE if ch in mic_coords}
    coords_side_b = {ch: mic_coords[ch] for ch in _SIDE_B_RANGE if ch in mic_coords}

    with h5py.File(db_path, "r") as f:
        group_names = sorted(f.keys())
        n_total = len(group_names)

        validated = [g for g in group_names if is_sound_validated(f[g])]
        n_valid = len(validated)

        print(f"[INFO] {n_total} total groups in the database")
        print(f"[INFO] {n_valid} candidates with sound_validated = True\n")

        if n_valid == 0:
            print("  No validated candidates found.")
            return

        for i, gname in enumerate(validated, start=1):
            grp = f[gname]

            try:
                raw_name = grp.attrs.get("raw_name", "")
                ch_strong = int(grp["ch"][()])
                start_peak = float(grp["start_peak"][()])
                end_peak = float(grp["end_peak"][()])
                ch_involved_raw = grp["channels_involved"][()]
                ch_involved = sorted(set(
                    ch_involved_raw.tolist()
                    if hasattr(ch_involved_raw, "tolist")
                    else [int(ch_involved_raw)]
                ))
            except KeyError as e:
                print(f"[WARN] {gname}: missing field {e} — skipping")
                continue

            # only process candidates with a substantial number of involved channels
            if len(ch_involved) > 5:
                audio_path = os.path.join(audio_base_path, raw_name)
                if not os.path.exists(audio_path):
                    print(f"[SKIP] [{i}/{n_valid}] {gname}: file not found -> {audio_path}")
                    continue

                print(f"\n[{i}/{n_valid}] Candidate: {gname}")
                print(f"  raw_name     : {raw_name}")
                print(f"  ch_involved  : {ch_involved}")
                print(f"  start-end    : {start_peak:.3f}s - {end_peak:.3f}s")

                try:
                    with sf.SoundFile(audio_path) as snd:
                        sr = snd.samplerate
                        total_samples = len(snd)
                        t_start = max(0.0, start_peak - padding_sec)
                        t_end = min(total_samples / sr, end_peak + padding_sec)
                        s_start = int(t_start * sr)
                        s_end = min(int(t_end * sr), total_samples)
                        snd.seek(s_start)
                        chunk = snd.read(s_end - s_start)
                except Exception as e:
                    print(f"  [ERROR] audio read failed: {e} — skipping")
                    continue

                other_channels = [c for c in ch_involved if c != ch_strong]
                channels_to_show = [ch_strong] + other_channels

                hnr_per_ch: dict[int, float] = {}

                for ch in channels_to_show:
                    if chunk.ndim > 1:
                        if ch >= chunk.shape[1]:
                            print(f"    [WARN] ch={ch} out of range ({chunk.shape[1]} channels) — skipping")
                            continue
                        segment = chunk[:, ch]
                    else:
                        segment = chunk

                    hnr_value = compute_hnr_on_segment(segment, sr)
                    hnr_per_ch[ch] = hnr_value
                    hnr_str = f"{hnr_value:.2f} dB" if not np.isnan(hnr_value) else "N/A"

                    # Actual playback is disabled — uncomment to hear each channel segment
                    # try:
                    #     print(f"    -> playing ch_{ch} ({len(segment)/sr:.2f}s)  HNR={hnr_str}"
                    #         f"{' * strong' if ch == ch_strong else ''}")
                    #     sd.play(segment, sr)
                    #     sd.wait()
                    # except Exception as e:
                    #     print(f"    [ERROR] ch_{ch}: {e}")

                hnr_a = {ch: v for ch, v in hnr_per_ch.items() if ch in _SIDE_A_RANGE}
                hnr_b = {ch: v for ch, v in hnr_per_ch.items() if ch in _SIDE_B_RANGE}

                # shared color scale between side A and side B for direct comparison
                all_vals = [v for v in list(hnr_a.values()) + list(hnr_b.values())
                            if not np.isnan(v)]
                if all_vals:
                    shared_vmin = min(all_vals)
                    shared_vmax = max(all_vals)
                else:
                    shared_vmin = shared_vmax = None

                plot_candidate_hnr_heatmap(
                    hnr_a, coords_side_a,
                    f"{gname} — Side A (ch 0-15) HNR",
                    show=True,
                    vmin=shared_vmin, vmax=shared_vmax,
                )
                plot_candidate_hnr_heatmap(
                    hnr_b, coords_side_b,
                    f"{gname} — Side B (ch 16-31) HNR",
                    show=True,
                    vmin=shared_vmin, vmax=shared_vmax,
                    mirror_x=True,
                )

    print(f"\n[DONE] {n_valid} candidates processed.")


def main():
    """Parse CLI arguments, validate input paths, and run the sound-validated playback/HNR/heatmap pipeline."""
    parser = argparse.ArgumentParser(
        description="Plays back sound_validated candidates, computes HNR, and plots heatmaps"
    )
    parser.add_argument("h5_path", help="Path to the HDF5 candidates database")
    parser.add_argument("audio_folder", help="Folder with the raw multichannel WAV recordings")
    parser.add_argument("--mic-coords", required=True, help="Path to the JSON file with microphone coordinates")
    parser.add_argument("--padding", type=float, default=PADDING_SEC,
                         help="Seconds of padding before/after the segment")
    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        print(f"H5 file not found: {args.h5_path}")
        sys.exit(1)
    if not os.path.isdir(args.audio_folder):
        print(f"Audio folder not found: {args.audio_folder}")
        sys.exit(1)
    if not os.path.exists(args.mic_coords):
        print(f"Coordinates file not found: {args.mic_coords}")
        sys.exit(1)

    print(f"Playing back sound_validated candidates from: {args.h5_path}")
    playback_sound_validated(args.h5_path, args.audio_folder, args.mic_coords, args.padding)


if __name__ == "__main__":
    main()


# usage
# python playback_sound_validated_hnr_heatmap.py Z:\postprocessing_results\2026-06-09\results_2026-06-09.h5 Z:\recordings2026\audio\2026-06-09 --mic-coords Z:\Sound-of-Bees\postprocessing\calibration.json