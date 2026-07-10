"""
This script implements the main multichannel post-processing pipeline for whoop detection.
It reads multichannel WAV recordings, skips known broken channels and noisy time windows,
detects candidate peaks per channel, extracts pitch and harmonic features, and writes
the accepted candidates into an HDF5 database for later analysis and visualization.
It doesn't use any multithreading to speed up the processing analysing channels in parallel.
"""

import os
import re
import argparse
from dataclasses import dataclass
import sys
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone
import zoneinfo

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp_signal
import soundfile as sf
import sounddevice as sd
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from classes.whoop_detector import WhoopDetector
from classes.pitch_detector import PitchDetector
from classes.harmonics_analyzer import HarmonicsAnalyzer


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

TOLERANCE_SEC = 1.2

CHANNEL_BROKEN_1BASED = [2, 3, 8, 13, 21, 25, 27, 28, 31]
CHANNEL_BROKEN = [x - 1 for x in CHANNEL_BROKEN_1BASED]  # 0-based

STARTING_FILE = "audio_recording_2025-09-20T00_00_47.909112Z.wav"

TZ_KONSTANZ = zoneinfo.ZoneInfo("Europe/Berlin")

FIGURES = {
    'fig_size':           (12, 5),
    'label_fontsize':     14,
    'title_fontsize':     16,
    'legend_fontsize':    12,
    'tick_fontsize':      14,
    'colorbar_labelsize': 14,
    'colorbar_ticksize':  12,
}

HNR_DETECTOR_PARAMS = dict(
    window_length_ms=50, hop_length_ms=10,
    f0_min=250, f0_max=700, window_type='hamming',
    lowpass_cutoff=15000, highpass_cutoff=2500,
    normalize=True, target_rms=0.1,
)
HNR_DETECT_PARAMS = dict(
    percentile=98, offset=4, window_sec=0.5, merge_overlaps=True,
)
PITCH_PARAMS = dict(
    length_queue=5, hz_threshold=25, threshold_increment=1.3,
    padding_start_ms=5, padding_end_ms=25, freq_min=200, freq_max=600,
)
HARMONIC_PARAMS = dict(
    window_duration_ms=10, num_harmonics=10, bandwidth_hz=100,
    prominence_threshold_ratio=0.07, plot_core=False, plot_verbose=False, verbose=False,
)


# ---------------------------------------------------------------------------
# Dataclass features
# ---------------------------------------------------------------------------

@dataclass
class WhoopFeatures:
    """Feature extraction results container (all indices are 0-based)."""
    raw_name: str
    ch: int = 0
    peak_time: float = 0.0
    start_peak: float = 0.0
    end_peak: float = 0.0
    rough_duration: float = 0.0
    date: Optional[str] = None
    time: Optional[str] = None
    sr: Optional[int] = None

    hnr_level:          Optional[float] = None
    f0_mean:            Optional[float] = None
    precise_start_peak: Optional[float] = None
    precise_end_peak:   Optional[float] = None
    precise_duration:   Optional[float] = None
    weighted_shr:       Optional[float] = None
    max_aligned_peaks:  Optional[int]   = None

    channels_involved: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _scalar_or_nan(val) -> np.ndarray:
    return np.array(val) if val is not None else np.array(np.nan)

def _array_or_empty(val) -> np.ndarray:
    return np.array(val) if val is not None else np.array([])


def append_features_to_h5(group_name: str, features: WhoopFeatures, hdf5_path: str) -> None:
    """Write a WhoopFeatures instance as an HDF5 group."""
    with h5py.File(hdf5_path, 'a') as f:
        if group_name in f:
            print(f"⚠️  {group_name} already exists, skipping")
            return

        grp = f.create_group(group_name)
        grp.attrs['date']     = features.date or ""
        grp.attrs['time']     = features.time or ""
        grp.attrs['raw_name'] = features.raw_name or ""

        scalars = dict(
            ch=features.ch, peak_time=features.peak_time,
            start_peak=features.start_peak, end_peak=features.end_peak,
            rough_duration=features.rough_duration, sr=features.sr,
            hnr_level=features.hnr_level, f0_mean=features.f0_mean,
            precise_start_peak=features.precise_start_peak,
            precise_end_peak=features.precise_end_peak,
            precise_duration=features.precise_duration,
            weighted_shr=features.weighted_shr,
            max_aligned_peaks=features.max_aligned_peaks,
        )
        for name, val in scalars.items():
            grp.create_dataset(name, data=_scalar_or_nan(val))

        grp.create_dataset('channels_involved', data=_array_or_empty(features.channels_involved))

    print(f"  ✓ saved: {group_name}")


# ---------------------------------------------------------------------------
# HDF5 summary
# ---------------------------------------------------------------------------

def print_h5_summary(hdf5_path: str) -> None:
    if not os.path.exists(hdf5_path):
        print(f"  File {hdf5_path} not found.")
        return

    header = (f"{'group_name':<72} {'ch':>4}  {'peak_time':>12}  {'hnr_level':>10}"
              f"  {'channels_involved':<20}  {'raw_name':<50}")
    sep = "-" * len(header)

    rows = []
    with h5py.File(hdf5_path, 'r') as f:
        for gname in f.keys():
            grp       = f[gname]
            ch        = int(grp['ch'][()]) if 'ch' in grp else -1
            peak_time = float(grp['peak_time'][()]) if 'peak_time' in grp else np.nan
            hnr_level = float(grp['hnr_level'][()]) if 'hnr_level' in grp else np.nan
            raw_name  = grp.attrs.get('raw_name', 'N/A')
            if 'channels_involved' in grp:
                arr    = grp['channels_involved'][()]
                ch_str = str(arr.tolist()) if arr.size > 0 else "[]"
            else:
                ch_str = "N/A"
            rows.append((gname, ch, peak_time, hnr_level, ch_str, raw_name))

    rows.sort(key=lambda r: (r[5], r[2]))

    print(sep)
    print(header)
    print(sep)
    for gname, ch, peak_time, hnr_level, ch_str, raw_name in rows:
        hnr_str = f"{hnr_level:>10.2f}" if not np.isnan(hnr_level) else f"{'nan':>10}"
        print(f"{gname:<72} {ch:>4}  {peak_time:>12.3f}  {hnr_str}"
              f"  {ch_str:<20}  {raw_name:<50}")
    print(sep)
    print(f"  Total groups: {len(rows)}")


# ---------------------------------------------------------------------------
# Analyze temp.h5 -> results.h5
# ---------------------------------------------------------------------------

def analyze_temp_file(temp_hdf5_path: str, results_hdf5_path: str) -> None:
    while True:
        with h5py.File(temp_hdf5_path, 'r') as f:
            group_names = list(f.keys())

        if not group_names:
            break

        ref_name = group_names[0]
        with h5py.File(temp_hdf5_path, 'r') as f:
            ref_time = float(f[ref_name]['peak_time'][()])

        cluster = []
        with h5py.File(temp_hdf5_path, 'r') as f:
            for name in group_names:
                grp = f[name]
                pt  = float(grp['peak_time'][()])
                # Peaks are clustered when they occur within a short temporal tolerance.
                if abs(pt - ref_time) <= TOLERANCE_SEC:
                    cluster.append({
                        'name':      name,
                        'peak_time': pt,
                        'hnr_level': float(grp['hnr_level'][()]) if 'hnr_level' in grp else np.nan,
                        'ch':        int(grp['ch'][()]),
                    })

        channels_involved = sorted(c['ch'] for c in cluster)
        # The best representative is the candidate with the strongest HNR.
        best = max(cluster, key=lambda c: c['hnr_level'] if not np.isnan(c['hnr_level']) else -np.inf)

        print(f"  event t={ref_time:.3f}s | chs={channels_involved} | "
              f"best ch={best['ch']} HNR={best['hnr_level']:.2f} dB")

        with h5py.File(temp_hdf5_path, 'r') as src, \
             h5py.File(results_hdf5_path, 'a') as dst:
            result_name = best['name']
            if result_name not in dst:
                src.copy(result_name, dst, name=result_name)
                if 'channels_involved' in dst[result_name]:
                    del dst[result_name]['channels_involved']
                dst[result_name].create_dataset(
                    'channels_involved', data=np.array(channels_involved, dtype=np.int32)
                )

        with h5py.File(temp_hdf5_path, 'a') as f:
            for c in cluster:
                if c['name'] in f:
                    del f[c['name']]


# ---------------------------------------------------------------------------
# Noisy moments
# ---------------------------------------------------------------------------

def load_noisy_moments(csv_path: str = "noisy_moments.csv") -> list:
    """
    Read noisy_moments.csv with rows: YYYY-MM-DD, HH:MM, HH:MM (Konstanz local time).
    Returns a list of tuples (start_utc, end_utc) as timezone-aware UTC datetimes.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, csv_path)

    noisy = []
    if not os.path.exists(csv_path):
        print(f"  ⚠️  {csv_path} not found, no filter applied.")
        return noisy

    with open(csv_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 3:
                continue
            date_str, start_str, end_str = parts
            start_local = datetime.strptime(f"{date_str} {start_str}", "%Y-%m-%d %H:%M")
            end_local   = datetime.strptime(f"{date_str} {end_str}",   "%Y-%m-%d %H:%M")
            start_utc = start_local.replace(tzinfo=TZ_KONSTANZ).astimezone(timezone.utc)
            end_utc   = end_local.replace(tzinfo=TZ_KONSTANZ).astimezone(timezone.utc)
            noisy.append((start_utc, end_utc))

    print(f"  Loaded {len(noisy)} noisy window(s) from {csv_path}")
    return noisy


def _is_noisy(audio_filename: str, noisy_windows: list) -> bool:
    """Return True if the file's UTC timestamp falls inside a noisy window."""
    if not noisy_windows:
        return False

    m = re.search(r'(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})', audio_filename)
    if not m:
        return False

    file_dt = datetime(
        int(m.group(1)[:4]), int(m.group(1)[5:7]), int(m.group(1)[8:]),
        int(m.group(2)), int(m.group(3)), int(m.group(4)),
        tzinfo=timezone.utc,
    )
    return any(start_utc <= file_dt <= end_utc for start_utc, end_utc in noisy_windows)


# ---------------------------------------------------------------------------
# Parse filename
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> Dict[str, str]:
    pattern = r"audio_recording_(\d{4}-\d{2}-\d{2})T(\d{2}_\d{2}_\d{2}\.\d{6})Z(?:_ch_\d+)?\.wav"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")
    return {"date": match.group(1), "time": match.group(2).replace("_", ":")}


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(audio: np.ndarray, sr: int, plot: bool = False,
                        title: str = "", **fig_kw) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (and optionally plot) the spectrogram. Returns (spec_dB, frequencies, times)."""
    nperseg = 1024
    frequencies, times, spec_data = sp_signal.spectrogram(
        audio, fs=sr, nperseg=nperseg, noverlap=nperseg // 2,
        nfft=nperseg * 4, window='hann', scaling='density',
    )
    spec_dB = 20.0 * np.log10(spec_data + 1e-10)

    if plot:
        vmax = np.nanmax(spec_dB)
        fig, ax = plt.subplots(figsize=fig_kw.get('fig_size', (10, 6)))
        im = ax.pcolormesh(times, frequencies, spec_dB, shading='gouraud',
                           cmap='hot', vmin=vmax - 80, vmax=vmax)
        ax.set_ylabel('Frequency (Hz)', fontsize=fig_kw.get('label_fontsize', 12))
        ax.set_xlabel('Time (s)',        fontsize=fig_kw.get('label_fontsize', 12))
        ax.set_title(title,              fontsize=fig_kw.get('title_fontsize', 14), fontweight='bold')
        ax.tick_params(labelsize=fig_kw.get('tick_fontsize', 12))
        ax.set_ylim([0, 20000])
        ax.set_xlim([times[0], times[-1]])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', size=fig_kw.get('colorbar_labelsize', 12),
                       fontweight='bold', labelpad=15)
        cbar.ax.tick_params(labelsize=fig_kw.get('colorbar_ticksize', 12))
        plt.tight_layout()
        plt.show()

    return spec_dB, frequencies, times


def plot_spectrogram_from_raw(raw_audio_path: str, start_peak: float, end_peak: float,
                               ch: int, sr: int = 48000, **fig_kw) -> None:
    """Reload the segment from the original WAV and plot its spectrogram (post-processing use)."""
    audio, sr = sf.read(raw_audio_path)
    segment   = audio[int(start_peak * sr):int(end_peak * sr), ch]
    compute_spectrogram(segment, sr, plot=True,
                        title=f"Spectrogram Ch{ch} @ t={start_peak:.3f}s", **fig_kw)


# ---------------------------------------------------------------------------
# Processing: single peak
# ---------------------------------------------------------------------------

def _process_peak(segment: np.ndarray, sr: int, info: dict, ch: int,
                  date_time_info: dict, audio_filename: str,
                  listen: bool, plot: bool, verbose: bool,
                  temp_h5: str) -> None:

    if listen:
        # Optional playback is useful when visually and acoustically checking a candidate.
        sd.play(segment, sr)
        sd.wait()
        sd.sleep(500)

    pitch_detector = PitchDetector(segment, sr)
    f0, best_queue_dic, all_queues, _, whoop_pitch_info = pitch_detector.estimate_f0(**PITCH_PARAMS)

    if whoop_pitch_info is None:
        if verbose:
            print(f"     no whoop pitch detected")
        return

    if verbose:
        pitch_detector.print_whoop_info()
    else:
        print(f"     f0={f0:.1f} Hz", end="")

    if plot:
        # Plot the pitch estimation and the spectrogram when interactive inspection is requested.
        pitch_detector._plot_results(
            np.asarray(pitch_detector.compute_fundamental_frequencies(freq_min=200, freq_max=600)),
            best_queue_dic['frame_indices'], all_queues, f0, **FIGURES,
        )
        compute_spectrogram(segment, sr, plot=True,
                            title=f"Spectrogram Ch{ch}", **FIGURES)

    rel_start = whoop_pitch_info['start_sample_padded'] / sr
    rel_end   = whoop_pitch_info['end_sample_padded']   / sr

    # Harmonic analysis refines the acoustic characterization of the same candidate segment.
    harmonic_params = {**HARMONIC_PARAMS, 'plot_core': plot, 'verbose': verbose}
    harmonic_results = HarmonicsAnalyzer(sr=sr, nfft=8192).compute_weighted_shr(
        whoop_segment=segment, best_queue_dic=best_queue_dic,
        **harmonic_params, **FIGURES,
    )

    if verbose:
        print(f"     WHAR={harmonic_results['whar']:.3f}  "
              f"max_aligned_peaks={harmonic_results['alignment_max_peaks']}")
    else:
        print(f"  whar={harmonic_results['whar']:.3f}  "
              f"peaks={harmonic_results['alignment_max_peaks']}")

    feat = WhoopFeatures(
        raw_name=audio_filename,
        ch=ch, date=date_time_info['date'], time=date_time_info['time'], sr=sr,
        peak_time=info['peak_time'],
        start_peak=info['window_start'], end_peak=info['window_end'],
        rough_duration=info['window_end'] - info['window_start'],
        hnr_level=info['peak_hnr_value'],
        f0_mean=f0,
        precise_start_peak=info['window_start'] + rel_start,
        precise_end_peak=info['window_start'] + rel_end,
        precise_duration=rel_end - rel_start,
        weighted_shr=harmonic_results['whar'],
        max_aligned_peaks=harmonic_results['alignment_max_peaks'],
    )

    group_name = f"{audio_filename.replace('.wav', '')}_ch_{ch}_peak_{feat.peak_time}"
    append_features_to_h5(group_name, feat, hdf5_path=temp_h5)


# ---------------------------------------------------------------------------
# Processing: single channel
# ---------------------------------------------------------------------------

def _process_channel(ch: int, signal_multichannel: np.ndarray, sr: int,
                     date_time_info: dict, audio_filename: str,
                     listen: bool, plot: bool, verbose: bool,
                     temp_h5: str) -> None:
    channel_signal = signal_multichannel[:, ch]

    # Each channel is processed independently so peak detection and feature extraction
    # can stay localized to the acoustically useful channel.
    detector = WhoopDetector(signal=channel_signal, sr=sr, **HNR_DETECTOR_PARAMS)
    detector.detect(**HNR_DETECT_PARAMS)

    n_peaks = len(detector.peak_times_)

    if n_peaks == 0:
        if verbose:
            print(f"  CH {ch:02d} | no peaks (thr={detector.threshold_:.2f} dB)")
        return

    print(f"  CH {ch:02d} | {n_peaks} peak(s) | thr={detector.threshold_:.2f} dB")

    if plot:
        detector.plot_analysis(ch_num=ch)

    for idx, (info, segment) in enumerate(zip(detector.get_peak_info(), detector.extract_segments())):
        if verbose:
            print(f"    peak {idx}: t={info['peak_time']:.3f}s  "
                  f"[{info['window_start']:.3f}-{info['window_end']:.3f}]s  "
                  f"HNR={info['peak_hnr_value']:.2f} dB  ({len(segment)/sr:.3f}s)")
        else:
            print(f"    peak {idx}: t={info['peak_time']:.3f}s  "
                  f"HNR={info['peak_hnr_value']:.2f} dB", end="  ")
        try:
            _process_peak(segment, sr, info, ch, date_time_info, audio_filename,
                          listen=listen, plot=plot, verbose=verbose,
                          temp_h5=temp_h5)
        except Exception as e:
            print(f"\n    ⚠️  peak {idx} error: {e}")


# ---------------------------------------------------------------------------
# Processing: single file
# ---------------------------------------------------------------------------

def _process_file(audio_file: str, audio_filename: str, results_folder: str,
                  listen: bool, plot: bool, verbose: bool,
                  noisy_windows: list) -> None:

    if _is_noisy(audio_filename, noisy_windows):
        print(f"  ⏭️  skipped (noisy window)")
        return

    date_time_info = parse_filename(audio_filename)

    try:
        signal_multichannel, sr = sf.read(audio_file)
    except Exception as e:
        print(f"  ⚠️  load error: {e}")
        return

    duration = len(signal_multichannel) / sr
    n_ch     = signal_multichannel.shape[1]
    print(f"  {sr} Hz  {duration:.1f}s  {n_ch}ch")

    os.makedirs(results_folder, exist_ok=True)
    temp_h5    = os.path.join(results_folder, 'temp.h5')
    results_h5 = os.path.join(results_folder, f"results_{date_time_info['date']}.h5")

    # Create a fresh temp.h5 for this file and remove any leftover from a previous crash.
    if os.path.exists(temp_h5):
        os.remove(temp_h5)
        print(f"  ⚠️  leftover temp.h5 removed")
    with h5py.File(temp_h5, 'w'):
        pass
    print(f"  temp.h5 created")

    try:
        for ch in range(n_ch):
            if ch in CHANNEL_BROKEN:
                continue
            try:
                _process_channel(ch, signal_multichannel, sr, date_time_info, audio_filename,
                                 listen=listen, plot=plot, verbose=verbose,
                                 temp_h5=temp_h5)
            except Exception as e:
                print(f"  [CH {ch:02d}] ⚠️  {e}")

        # Temporary candidate groups are consolidated into the final per-day HDF5 file.
        print(f"\n  → consolidate temp.h5 into {os.path.basename(results_h5)}")
        if verbose:
            print_h5_summary(temp_h5)
        analyze_temp_file(temp_h5, results_h5)
        print_h5_summary(results_h5)

    finally:
        # Always remove temp.h5, even if an unexpected exception occurs.
        if os.path.exists(temp_h5):
            os.remove(temp_h5)
            print(f"  temp.h5 removed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch pipeline for whoop analysis -- classical DSP"
    )
    parser.add_argument("-f", "--input-folder",   required=True,
                        help="Folder containing multichannel WAV files")
    parser.add_argument("-r", "--results-folder", default=".",
                        help="Folder where results are saved (HDF5)")
    parser.add_argument("--listen",  action="store_true",
                        help="Play each candidate segment")
    parser.add_argument("--plot",    action="store_true",
                        help="Show HNR, pitch, and spectrogram plots")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("PIPELINE CLASSICAL SIGNAL PROCESSING")
    if any([args.listen, args.plot, args.verbose]):
        flags = " ".join(f"--{k}" for k, v in
                         [("listen", args.listen), ("plot", args.plot), ("verbose", args.verbose)] if v)
        print(f"flags: {flags}")
    print("="*80)

    noisy_windows = load_noisy_moments("noisy_moments.csv")

    audio_files = sorted(f for f in os.listdir(args.input_folder) if f.endswith('.wav'))

    try:
        audio_files = audio_files[audio_files.index(STARTING_FILE):]
        print(f"Starting from: {STARTING_FILE}")
    except ValueError:
        print("Starting file not found, processing from the beginning.")

    if not audio_files:
        print(f"No .wav files in {args.input_folder}")
        return

    print(f"Files to process: {len(audio_files)}\n")

    for idx, audio_filename in enumerate(audio_files, 1):
        print(f"\n[{idx}/{len(audio_files)}] {audio_filename}")
        print("-" * 80)
        _process_file(
            os.path.join(args.input_folder, audio_filename), audio_filename,
            args.results_folder, listen=args.listen, plot=args.plot, verbose=args.verbose,
            noisy_windows=noisy_windows,
        )

    print("\n" + "="*80)
    print("DONE")


if __name__ == "__main__":
    main()