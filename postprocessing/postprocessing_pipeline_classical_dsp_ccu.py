"""
This script implements a parallel post-processing pipeline for multichannel audio.
It detects whoop-like candidates in each channel, extracts per-candidate features,
filters and groups nearby peaks across channels, and stores the accepted events
in HDF5 databases for later analysis.
It is designed to process channels in parallel using multiple CPU cores and so to
be run on the ccu cluster.
"""

import os
import re
import argparse
from dataclasses import dataclass
import sys
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import zoneinfo



import numpy as np
import soundfile as sf
import h5py



sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))



from classes.whoop_detector import WhoopDetector
from classes.pitch_detector import PitchDetector
from classes.harmonics_analyzer import HarmonicsAnalyzer




# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------



TOLERANCE_SEC = 1.2



STARTING_FILE = "2026-05-27T03_00_23.516288Z.wav"



TZ_KONSTANZ = zoneinfo.ZoneInfo("Europe/Berlin")



F0_OUTLIER_K = 2.5   # MAD multiplier for f0 outlier filtering
MAD_FLOOR_HZ = 15.0  # minimum f0 variability floor (Hz)



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
# Helpers HDF5
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
# Stampa HDF5
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
    print(f"  Totale gruppi: {len(rows)}")




# ---------------------------------------------------------------------------
# Whoop candidate filter
# ---------------------------------------------------------------------------



def passes_filter(f0_mean: float, precise_duration: float,
                  hnr_level: float, channels_involved: int) -> bool:
    # The final decision is based on a small set of hand-tuned acoustic rules.
    # A candidate must match one of the allowed f0/duration bands and also pass the
    # minimum HNR and multi-channel support checks.
    cond1 = (310 <= f0_mean <= 350) and (0.050 <= precise_duration <= 0.100)
    cond2 = (430 <= f0_mean <= 470) and (0.110 <= precise_duration <= 0.190)
    cond3 = (500 <= f0_mean <= 600) and (0.100 <= precise_duration <= 0.220)
    hnr_cond      = hnr_level >= 2
    channels_cond = channels_involved >= 2
    return (cond1 or cond2 or cond3) and hnr_cond and channels_cond




# ---------------------------------------------------------------------------
# F0 outlier filtering with MAD
# ---------------------------------------------------------------------------



def _filter_f0_outliers(cluster: list, hdf5_path: str, k: float = F0_OUTLIER_K) -> list:
    """
    Remove cluster members with anomalous f0_mean values (MAD-based).
    Loads f0_mean from hdf5_path. Elements without f0_mean are excluded.
    Returns the filtered cluster; if all members would be removed, returns the original cluster.
    """
    # Use the median and MAD to reject f0 values that do not agree with the local cluster.
    f0_values = []
    with h5py.File(hdf5_path, 'r') as f:
        for c in cluster:
            grp = f[c['name']]
            f0 = float(grp['f0_mean'][()]) if 'f0_mean' in grp else np.nan
            f0_values.append(f0)


    f0_arr = np.array(f0_values)
    valid_mask = ~np.isnan(f0_arr)


    if valid_mask.sum() < 2:
        return cluster


    valid_f0  = f0_arr[valid_mask]
    median_f0 = np.median(valid_f0)
    mad_raw   = np.median(np.abs(valid_f0 - median_f0))
    mad       = max(mad_raw, MAD_FLOOR_HZ)
    threshold = k * mad
    filtered  = []
    for c, f0 in zip(cluster, f0_values):
        if np.isnan(f0) or abs(f0 - median_f0) > threshold:
            print(f"    ⚠️  removed f0 outlier: ch={c['ch']} f0={f0:.1f} Hz "
                  f"(median={median_f0:.1f}, MAD_raw={mad_raw:.1f}, MAD_eff={mad:.1f}, thr=±{threshold:.1f})")
            with h5py.File(hdf5_path, 'a') as f_h5:
                if c['name'] in f_h5:
                    del f_h5[c['name']]
        else:
            filtered.append(c)


    if not filtered:
        print(f"    ⚠️  all members were removed by the f0 filter, restoring the original cluster")
        return cluster


    return filtered


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
                # Peaks are grouped when they occur within a short temporal tolerance.
                if abs(pt - ref_time) <= TOLERANCE_SEC:
                    cluster.append({
                        'name':      name,
                        'peak_time': pt,
                        'hnr_level': float(grp['hnr_level'][()]) if 'hnr_level' in grp else np.nan,
                        'ch':        int(grp['ch'][()]),
                    })


        cluster = _filter_f0_outliers(cluster, temp_hdf5_path)


        channels_involved = sorted(c['ch'] for c in cluster)
    # The best representative for the cluster is the one with the strongest HNR.
        best = max(cluster, key=lambda c: c['hnr_level'] if not np.isnan(c['hnr_level']) else -np.inf)


        # ── read additional fields for the whoop filter ────────────────────
        with h5py.File(temp_hdf5_path, 'r') as f:
            best_grp         = f[best['name']]
            f0_mean          = float(best_grp['f0_mean'][()]) if 'f0_mean' in best_grp else np.nan
            precise_duration = float(best_grp['precise_duration'][()]) if 'precise_duration' in best_grp else np.nan


        # ── whoop filter ────────────────────────────────────────────────────
        if np.isnan(f0_mean) or np.isnan(precise_duration) or \
           not passes_filter(f0_mean, precise_duration, best['hnr_level'], len(channels_involved)):
            print(f"  ✗ rejected  t={ref_time:.3f}s  f0={f0_mean:.1f} Hz  "
                  f"dur={precise_duration*1000:.1f} ms  HNR={best['hnr_level']:.2f} dB  "
                  f"chs={len(channels_involved)}")
            with h5py.File(temp_hdf5_path, 'a') as f:
                for c in cluster:
                    if c['name'] in f:
                        del f[c['name']]
            continue


        print(f"  ✓ event t={ref_time:.3f}s | chs={channels_involved} | "
              f"best ch={best['ch']} HNR={best['hnr_level']:.2f} dB  "
              f"f0={f0_mean:.1f} Hz  dur={precise_duration*1000:.1f} ms")


        # Only accepted clusters are copied from the temporary staging file into the final database.
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
    # Supports both the old format (audio_recording_YYYY-...) and the new one (YYYY-...)
    pattern = r"(?:audio_recording_)?(\d{4}-\d{2}-\d{2})T(\d{2}_\d{2}_\d{2}\.\d{6})Z(?:_ch_\d+)?\.wav"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")
    return {"date": match.group(1), "time": match.group(2).replace("_", ":")}




# ---------------------------------------------------------------------------
# Processing: single peak -> returns WhoopFeatures or None
# ---------------------------------------------------------------------------



def _process_peak(segment: np.ndarray, sr: int, info: dict, ch: int,
                  date_time_info: dict, audio_filename: str) -> Optional[WhoopFeatures]:


    # First estimate the pitch on the extracted peak segment, then refine the candidate
    # boundaries and compute harmonic features for the final event description.
    pitch_detector = PitchDetector(segment, sr)
    f0, best_queue_dic, _, _, whoop_pitch_info = pitch_detector.estimate_f0(**PITCH_PARAMS)


    if whoop_pitch_info is None:
        return None


    rel_start = whoop_pitch_info['start_sample_padded'] / sr
    rel_end   = whoop_pitch_info['end_sample_padded']   / sr


    harmonic_results = HarmonicsAnalyzer(sr=sr, nfft=8192).compute_weighted_shr(
        whoop_segment=segment, best_queue_dic=best_queue_dic, **HARMONIC_PARAMS,
    )


    return WhoopFeatures(
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




# ---------------------------------------------------------------------------
# Processing: single channel -> worker executed in a subprocess
# ---------------------------------------------------------------------------



def _channel_worker(args: tuple) -> list:
    """
    Top-level function required for pickling with ProcessPoolExecutor.
    Receives a tuple of arguments and returns a list of (group_name, WhoopFeatures).
    """
    ch, channel_signal, sr, date_time_info, audio_filename = args


    results = []
    try:
        # Run the whoop detector independently on each channel so peaks can be extracted in parallel.
        detector = WhoopDetector(signal=channel_signal, sr=sr, **HNR_DETECTOR_PARAMS)
        detector.detect(**HNR_DETECT_PARAMS)


        n_peaks = len(detector.peak_times_)
        if n_peaks == 0:
            print(f"  CH {ch:02d} | no peaks (thr={detector.threshold_:.2f} dB)")
            return results


        print(f"  CH {ch:02d} | {n_peaks} peak(s) | thr={detector.threshold_:.2f} dB")


        for idx, (info, segment) in enumerate(zip(detector.get_peak_info(), detector.extract_segments())):
            print(f"    CH {ch:02d} peak {idx}: t={info['peak_time']:.3f}s  "
                  f"HNR={info['peak_hnr_value']:.2f} dB", end="  ")
            try:
                feat = _process_peak(segment, sr, info, ch, date_time_info, audio_filename)
                if feat is not None:
                    print(f"f0={feat.f0_mean:.1f} Hz  whar={feat.weighted_shr:.3f}  "
                          f"peaks={feat.max_aligned_peaks}")
                    group_name = (f"{audio_filename.replace('.wav', '')}"
                                  f"_ch_{ch}_peak_{feat.peak_time}")
                    results.append((group_name, feat))
                else:
                    print("no whoop pitch detected")
            except Exception as e:
                print(f"\n    ⚠️  CH {ch:02d} peak {idx} error: {e}")


    except Exception as e:
        print(f"  [CH {ch:02d}] ⚠️  {e}")


    return results




# ---------------------------------------------------------------------------
# Processing: single file (parallel orchestration)
# ---------------------------------------------------------------------------



def _process_file(audio_file: str, audio_filename: str, results_folder: str,
                  num_workers: int, noisy_windows: list) -> None:


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
    print(f"  {sr} Hz  {duration:.1f}s  {n_ch}ch  workers={num_workers}")


    os.makedirs(results_folder, exist_ok=True)
    temp_h5    = os.path.join(results_folder, 'temp.h5')
    results_h5 = os.path.join(results_folder, f"results_{date_time_info['date']}.h5")


    # temp.h5 is a staging file: workers write candidate peaks there first, then
    # the clustering/filtering pass decides what should survive in the final database.
    if os.path.exists(temp_h5):
        os.remove(temp_h5)
        print(f"  ⚠️  leftover temp.h5 removed")
    with h5py.File(temp_h5, 'w'):
        pass
    print(f"  temp.h5 created")


    active_channels = list(range(n_ch))
    worker_args = [
        (ch, signal_multichannel[:, ch].copy(), sr, date_time_info, audio_filename)
        for ch in active_channels
    ]


    try:
        all_results: list[tuple[str, WhoopFeatures]] = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_channel_worker, args): args[0] for args in worker_args}
            for future in as_completed(futures):
                ch = futures[future]
                try:
                    channel_results = future.result()
                    all_results.extend(channel_results)
                except Exception as e:
                    print(f"  [CH {ch:02d}] ⚠️  worker crashed: {e}")


        for group_name, feat in all_results:
            append_features_to_h5(group_name, feat, hdf5_path=temp_h5)


        analyze_temp_file(temp_h5, results_h5)


    finally:
        if os.path.exists(temp_h5):
            os.remove(temp_h5)
            print(f"  temp.h5 removed")




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch pipeline for whoop analysis -- classical DSP (parallel)"
    )
    parser.add_argument("-f", "--input-folder",   required=True,
                        help="Folder containing multichannel WAV files")
    parser.add_argument("-r", "--results-folder", default=".",
                        help="Folder where results are saved (HDF5)")
    parser.add_argument("-w", "--num-workers",    type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()


    print("\n" + "="*80)
    print("PIPELINE CLASSICAL SIGNAL PROCESSING  [parallel]")
    print(f"workers: {args.num_workers}")
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
            args.results_folder, num_workers=args.num_workers,
            noisy_windows=noisy_windows,
        )


    print("\n" + "="*80)
    print("DONE")




if __name__ == "__main__":
    main()