"""
This script implements a mono-channel post-processing pipeline for raw whoop candidates.
It parses each WAV file, detects peaks in the signal, estimates pitch when possible,
and stores every candidate directly into an HDF5 database without applying the final
whoop filtering stage used by the multichannel pipeline.
"""
# Usage:
# python postprocessing_pipeline_classical_dsp_MONO.py -f E:\training_dataset_finetuning\raw_audios -r Z:\postprocessing_results\labelled_validation_dataset --verbose

import os
import re
import argparse
from dataclasses import dataclass
from typing import Optional, Dict
import sys

import numpy as np
import soundfile as sf
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from classes.whoop_detector import WhoopDetector
from classes.pitch_detector import PitchDetector


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

STARTING_FILE = "audio_recording_2025-09-20T00_00_47.909112Z_ch_04.wav"

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


# ---------------------------------------------------------------------------
# Dataclass features
# ---------------------------------------------------------------------------

@dataclass
class WhoopFeatures:
    """Raw candidate -- no filtering applied."""
    group_id:  str
    raw_name:  str
    peak_time: float = 0.0
    duration:  Optional[float] = None  # nan if pitch is not detected
    f0:        Optional[float] = None  # nan if pitch is not detected
    hnr_level: Optional[float] = None


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _scalar_or_nan(val) -> np.ndarray:
    return np.array(val) if val is not None else np.array(np.nan)


def append_features_to_h5(features: WhoopFeatures, hdf5_path: str) -> None:
    with h5py.File(hdf5_path, 'a') as f:
        if features.group_id in f:
            print(f"  ⚠️  {features.group_id} already exists, skipping")
            return
        grp = f.create_group(features.group_id)
        grp.attrs['raw_name'] = features.raw_name
        grp.create_dataset('peak_time', data=np.array(features.peak_time))
        grp.create_dataset('duration',  data=_scalar_or_nan(features.duration))
        grp.create_dataset('f0',        data=_scalar_or_nan(features.f0))
        grp.create_dataset('hnr_level', data=_scalar_or_nan(features.hnr_level))
    print(f"  ✓ saved: {features.group_id}")


# ---------------------------------------------------------------------------
# HDF5 summary
# ---------------------------------------------------------------------------

def print_h5_summary(hdf5_path: str) -> None:
    if not os.path.exists(hdf5_path):
        print(f"  File {hdf5_path} not found.")
        return

    header = (f"{'group_id':<80}  {'peak_time':>12}  {'duration':>10}"
              f"  {'f0':>8}  {'hnr_level':>10}")
    sep = "-" * len(header)

    rows = []
    with h5py.File(hdf5_path, 'r') as f:
        for gid in f.keys():
            grp       = f[gid]
            peak_time = float(grp['peak_time'][()]) if 'peak_time' in grp else np.nan
            duration  = float(grp['duration' ][()]) if 'duration'  in grp else np.nan
            f0        = float(grp['f0'       ][()]) if 'f0'        in grp else np.nan
            hnr_level = float(grp['hnr_level'][()]) if 'hnr_level' in grp else np.nan
            rows.append((gid, peak_time, duration, f0, hnr_level))

    rows.sort(key=lambda r: r[1])

    print(sep)
    print(header)
    print(sep)
    for gid, peak_time, duration, f0, hnr_level in rows:
        dur_str = f"{duration:>10.3f}" if not np.isnan(duration)  else f"{'nan':>10}"
        f0_str  = f"{f0:>8.1f}"        if not np.isnan(f0)        else f"{'nan':>8}"
        hnr_str = f"{hnr_level:>10.2f}" if not np.isnan(hnr_level) else f"{'nan':>10}"
        print(f"{gid:<80}  {peak_time:>12.3f}  {dur_str}  {f0_str}  {hnr_str}")
    print(sep)
    print(f"  Total candidates: {len(rows)}")


# ---------------------------------------------------------------------------
# Parse filename
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> Dict[str, str]:
    pattern = (
        r"audio_recording_"
        r"(\d{4}-\d{2}-\d{2})"
        r"T(\d{2}_\d{2}_\d{2}\.\d+)Z"
        r"_ch_(\d+)\.wav"
    )
    m = re.match(pattern, filename)
    if not m:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")
    return {
        "date":    m.group(1),
        "time":    m.group(2).replace("_", ":"),
        "channel": m.group(3),
    }


# ---------------------------------------------------------------------------
# Processing: single peak -- always save, with or without pitch
# ---------------------------------------------------------------------------

def _process_peak(segment: np.ndarray, sr: int, info: dict,
                  audio_filename: str, verbose: bool,
                  results_h5: str) -> None:

    # Use the peak timestamp as the stable identifier for the candidate group.
    stem     = audio_filename.replace('.wav', '')
    group_id = f"{stem}_peak_{info['peak_time']:.6f}"
    hnr      = info['peak_hnr_value']

    # Try pitch estimation; if it fails, keep the candidate and store f0/duration as NaN.
    f0, duration = None, None
    try:
        pitch_detector = PitchDetector(segment, sr)
        f0_est, _, _, _, whoop_pitch_info = pitch_detector.estimate_f0(**PITCH_PARAMS)
        if whoop_pitch_info is not None:
            rel_start = whoop_pitch_info['start_sample_padded'] / sr
            rel_end   = whoop_pitch_info['end_sample_padded']   / sr
            f0        = f0_est
            duration  = rel_end - rel_start
            # The extracted pitch window defines the more precise candidate duration.
            if verbose:
                print(f"     f0={f0:.1f} Hz  duration={duration:.3f}s")
            else:
                print(f"  f0={f0:.1f} Hz  dur={duration:.3f}s")
        else:
            if verbose:
                print("     no whoop pitch detected -- saved with f0=nan")
            else:
                print("  [no pitch]")
    except Exception as e:
        print(f"  ⚠️  pitch error: {e}")

    feat = WhoopFeatures(
        group_id=group_id,
        raw_name=audio_filename,
        peak_time=info['peak_time'],
        duration=duration,
        f0=f0,
        hnr_level=hnr,
    )
    append_features_to_h5(feat, hdf5_path=results_h5)


# ---------------------------------------------------------------------------
# Processing: single mono file
# ---------------------------------------------------------------------------

def _process_file(audio_file: str, audio_filename: str,
                  results_folder: str, verbose: bool) -> None:

    try:
        info = parse_filename(audio_filename)
    except ValueError as e:
        print(f"  ⚠️  {e}")
        return

    try:
        signal, sr = sf.read(audio_file)
    except Exception as e:
        print(f"  ⚠️  load error: {e}")
        return

    if signal.ndim == 2:
        signal = signal[:, 0]
        print("  ⚠️  file is not mono, extracted channel 0")

    duration = len(signal) / sr
    print(f"  {sr} Hz  {duration:.1f}s  ch={info['channel']}")

    os.makedirs(results_folder, exist_ok=True)
    results_h5 = os.path.join(results_folder, "results_raw.h5")

    # Peak detection is run on the full mono signal; each detected segment becomes a raw candidate.
    detector = WhoopDetector(signal=signal, sr=sr, **HNR_DETECTOR_PARAMS)
    detector.detect(**HNR_DETECT_PARAMS)

    n_peaks = len(detector.peak_times_)
    if n_peaks == 0:
        if verbose:
            print(f"  no peaks (thr={detector.threshold_:.2f} dB)")
        return

    print(f"  {n_peaks} peak(s)  thr={detector.threshold_:.2f} dB")

    for idx, (peak_info, segment) in enumerate(
            zip(detector.get_peak_info(), detector.extract_segments())):
        if verbose:
            print(f"    peak {idx}: t={peak_info['peak_time']:.3f}s  "
                  f"[{peak_info['window_start']:.3f}-{peak_info['window_end']:.3f}]s  "
                  f"HNR={peak_info['peak_hnr_value']:.2f} dB")
        else:
            print(f"    peak {idx}: t={peak_info['peak_time']:.3f}s  "
                  f"HNR={peak_info['peak_hnr_value']:.2f} dB", end="  ")
        try:
            _process_peak(segment, sr, peak_info, audio_filename,
                          verbose=verbose, results_h5=results_h5)
        except Exception as e:
            print(f"\n    ⚠️  peak {idx} error: {e}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch pipeline -- mono, raw HNR candidates (no filtering)"
    )
    parser.add_argument("-f", "--input-folder",   required=True,
                        help="Folder containing mono WAV files (_ch_NN.wav)")
    parser.add_argument("-r", "--results-folder", default=".",
                        help="Folder where results_raw.h5 will be saved")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("PIPELINE CLASSICAL SIGNAL PROCESSING — MONO  [NO FILTER]")
    if args.verbose:
        print("flags: --verbose")
    print("="*80 + "\n")

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
        print(f"[{idx}/{len(audio_files)}] {audio_filename}")
        print("-" * 80)
        _process_file(
            os.path.join(args.input_folder, audio_filename),
            audio_filename, args.results_folder, verbose=args.verbose,
        )

    print("="*80)
    print("DONE")


if __name__ == "__main__":
    main()



