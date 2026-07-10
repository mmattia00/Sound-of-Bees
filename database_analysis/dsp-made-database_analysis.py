#!/usr/bin/env python3
"""
dsp-made-database_analysis.py

DESCRIPTION:
    Toolkit for filtering and manually validating whoop candidates from an
    HDF5 results database. Provides HNR and channel statistics, interactive
    sound validation (spectrogram + audio playback), and interactive video
    validation (synchronized camera clips) via a subcommand-based CLI.

USAGE:
    python dsp-made-database_analysis.py <command> [arguments]

COMMANDS:
    filter                    Filter candidates by f0 / duration / HNR thresholds
    validate                  Sound validation (spectrogram + audio playback)
    validate_video            Video validation for sound_validated=True candidates
    inspect                   Inspect all channels of the filtered DB
    analyze_sound_validated    Show spectrogram/audio for sound_validated=True candidates
    analyze_video_validated    Show clips for video_validated=True candidates
    hnr                       HNR statistics on the database
    channels                  channels_involved distribution (main vs filtered)

EXAMPLES:
    python dsp-made-database_analysis.py filter              2026-05-27 Z:\postprocessing_results\2026-05-27\whoops_filtered.h5

    python dsp-made-database_analysis.py validate            2026-05-27
    python dsp-made-database_analysis.py validate            2026-05-27 --no-listen

    python dsp-made-database_analysis.py validate_video      2026-05-27 Z:\video_clips\2026-05-27

    python dsp-made-database_analysis.py inspect             2026-05-27
    python dsp-made-database_analysis.py analyze_sound_validated 2026-05-27
    python dsp-made-database_analysis.py analyze_video_validated 2026-05-27
    python dsp-made-database_analysis.py hnr                 2026-05-27
    python dsp-made-database_analysis.py channels            2026-05-27 Z:\postprocessing_results\2026-05-27\whoops_filtered.h5

DEPENDENCIES:
    pip install h5py numpy soundfile sounddevice matplotlib scipy python-vlc
    + VLC Media Player installed on the system (Windows/Linux/macOS) for video playback
"""

import argparse
import h5py
import numpy as np
import os
import subprocess
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from typing import Optional

import platform

if platform.system() == "Windows":
    _vlc_win_path = r"C:\Program Files\VideoLAN\VLC"
    if os.path.exists(_vlc_win_path):
        os.add_dll_directory(_vlc_win_path)

# now this import can find libvlc.dll
try:
    import vlc as _vlc_module
    VLC_AVAILABLE = True
except Exception:
    VLC_AVAILABLE = False


# ── path templates (hardcoded) ─────────────────────────────────────────────
# windows path
DB_BASE_DIR    = r"Z:\postprocessing_results"
AUDIO_BASE_DIR = r"Z:\recordings2026\audio"

# linux path
# DB_BASE_DIR    = r"/mnt/A26-03-0300/postprocessing_results"
# AUDIO_BASE_DIR = r"/mnt/A26-03-0300/recordings2026/audio"

DB_FILENAME_TEMPLATE = "results_{date}.h5"


def build_paths(date_str: str) -> tuple[str, str]:
    """Build the database path and audio base directory for a given date (YYYY-MM-DD)."""
    db_path    = os.path.join(DB_BASE_DIR,    date_str, DB_FILENAME_TEMPLATE.format(date=date_str))
    audio_base = os.path.join(AUDIO_BASE_DIR, date_str)
    return db_path, audio_base


def passes_filter(f0_mean: float, precise_duration: float, hnr_level: float, channels_involved: int) -> bool:
    """
    Return True if the candidate satisfies the acoustic filter conditions:
    one of three (f0, duration) bands, HNR above threshold, and multi-channel
    involvement.
    """
    cond1 = (310 <= f0_mean <= 350) and (0.050 <= precise_duration <= 0.100)
    cond2 = (430 <= f0_mean <= 470) and (0.110 <= precise_duration <= 0.190)
    cond3 = (500 <= f0_mean <= 600) and (0.100 <= precise_duration <= 0.220)
    hnr_cond      = hnr_level >= 2
    channels_cond = channels_involved >= 2
    return (cond1 or cond2 or cond3) and hnr_cond and channels_cond


def filter_h5(input_path: str, output_path: str) -> None:
    """
    Read all candidate groups from input_path, keep only those passing
    passes_filter(), and write them (with attributes and datasets) to a new
    HDF5 file at output_path.
    """
    total = 0
    passed = 0
    skipped_nan = 0

    with h5py.File(input_path, 'r') as fin, h5py.File(output_path, 'w') as fout:
        group_names = list(fin.keys())
        n_total = len(group_names)
        print(f"[INFO] Groups found in input: {n_total}")
        print(f"[INFO] Starting filter...\n")

        for i, group_name in enumerate(group_names, start=1):
            total += 1
            grp_in = fin[group_name]

            # read the required fields; skip the candidate if any is missing
            try:
                f0_mean           = float(grp_in['f0_mean'][()])
                precise_duration  = float(grp_in['precise_duration'][()])
                hnr_level         = float(grp_in['hnr_level'][()])
                ch_inv_raw        = grp_in['channels_involved'][()]
                channels_involved = int(ch_inv_raw) if np.ndim(ch_inv_raw) == 0 else int(np.asarray(ch_inv_raw).shape[0])
            except KeyError as e:
                print(f"[WARN] {group_name}: missing field {e} — skipping")
                skipped_nan += 1
                continue

            if np.isnan(f0_mean) or np.isnan(precise_duration) or np.isnan(hnr_level):
                skipped_nan += 1
                continue

            if not passes_filter(f0_mean, precise_duration, hnr_level, channels_involved):
                continue

            # copy the group (attributes + datasets) into the output file
            grp_out = fout.create_group(group_name)
            for attr_key, attr_val in grp_in.attrs.items():
                grp_out.attrs[attr_key] = attr_val
            for ds_name in grp_in:
                grp_in.copy(ds_name, grp_out)

            passed += 1
            if i % 500 == 0:
                print(f"[{i:>6}/{n_total}] processed — passed so far: {passed}")

    print(f"\n[DONE] Total candidates  : {total}")
    print(f"       Passed filter     : {passed}")
    print(f"       Rejected          : {total - passed - skipped_nan}")
    print(f"       Skipped (NaN/miss): {skipped_nan}")
    print(f"       Output written to : {output_path}")


def HNR_estimation(db_path: str, verbose: bool = True) -> dict:
    """
    Compute descriptive statistics (mean, median, std, percentiles) of the
    'hnr_level' field across all groups in the database, and estimate a
    noise threshold (mean - 1 std) to classify likely-noise vs likely-signal
    events. Optionally prints a formatted summary.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"File not found: {db_path}")

    hnr_values = []
    group_names = []

    with h5py.File(db_path, 'r') as f:
        for gname in f.keys():
            grp = f[gname]
            if 'hnr_level' in grp:
                val = float(grp['hnr_level'][()])
                if not np.isnan(val):
                    hnr_values.append(val)
                    group_names.append(gname)

    if not hnr_values:
        print("  ⚠️  No HNR values found in the database.")
        return {}

    hnr    = np.array(hnr_values)
    mean   = float(np.mean(hnr))
    median = float(np.median(hnr))
    std    = float(np.std(hnr))
    minimum = float(np.min(hnr))
    maximum = float(np.max(hnr))
    p25    = float(np.percentile(hnr, 25))
    p75    = float(np.percentile(hnr, 75))

    above_mean  = int(np.sum(hnr > mean))
    below_mean  = int(np.sum(hnr <= mean))
    above_p75   = int(np.sum(hnr > p75))
    below_p25   = int(np.sum(hnr < p25))

    noise_threshold = mean - std
    likely_noise    = int(np.sum(hnr < noise_threshold))
    likely_signal   = int(np.sum(hnr >= noise_threshold))

    stats = dict(
        total_events=len(hnr), mean=mean, median=median, std=std,
        min=minimum, max=maximum, p25=p25, p75=p75,
        above_mean=above_mean, below_mean=below_mean,
        above_p75=above_p75, below_p25=below_p25,
        noise_threshold=noise_threshold,
        likely_noise=likely_noise, likely_signal=likely_signal,
        hnr_values=hnr, group_names=group_names,
    )

    if verbose:
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  HNR ESTIMATION  —  {os.path.basename(db_path)}")
        print(sep)
        print(f"  Total events:         {len(hnr):>6}")
        print(f"  Mean HNR:             {mean:>8.2f} dB")
        print(f"  Median HNR:           {median:>8.2f} dB")
        print(f"  Std dev:              {std:>8.2f} dB")
        print(f"  Min / Max:            {minimum:>6.2f}  /  {maximum:.2f} dB")
        print(f"  25th / 75th percentile: {p25:>6.2f}  /  {p75:.2f} dB")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Above mean:           {above_mean:>6}  ({100*above_mean/len(hnr):.1f}%)")
        print(f"  Below/equal mean:     {below_mean:>6}  ({100*below_mean/len(hnr):.1f}%)")
        print(f"  Above 75th perc.:     {above_p75:>6}  ({100*above_p75/len(hnr):.1f}%)")
        print(f"  Below 25th perc.:     {below_p25:>6}  ({100*below_p25/len(hnr):.1f}%)")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Noise threshold (mean - 1σ): {noise_threshold:.2f} dB")
        print(f"  Likely noise:         {likely_noise:>6}  ({100*likely_noise/len(hnr):.1f}%)")
        print(f"  Likely signal:        {likely_signal:>6}  ({100*likely_signal/len(hnr):.1f}%)")
        print(sep + "\n")

    return stats


def channels_involved_analysis(main_db_path: str, filtered_db_path: str) -> dict:
    """
    Compare the distribution of 'channels_involved' between the main
    (unfiltered) and the filtered database, printing per-database stats
    and a side-by-side comparison table.
    """
    def analyze_db(db_path: str, label: str) -> dict:
        counts = []
        with h5py.File(db_path, 'r') as f:
            for gname in f.keys():
                grp = f[gname]
                if 'channels_involved' in grp:
                    val = grp['channels_involved'][()]
                    n = int(val) if np.ndim(val) == 0 else int(np.asarray(val).shape[0])
                    counts.append(n)
                else:
                    counts.append(0)

        counts = np.array(counts)
        total  = len(counts)
        multi  = int(np.sum(counts > 1))
        single = int(np.sum(counts == 1))
        zero   = int(np.sum(counts == 0))

        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  CHANNELS INVOLVED  —  {label}")
        print(sep)
        print(f"  Total events:             {total:>6}")
        print(f"  Channels = 0 (missing):   {zero:>6}  ({100*zero/total:.1f}%)")
        print(f"  Channels = 1 (single mic):{single:>6}  ({100*single/total:.1f}%)")
        print(f"  Channels > 1 (multi mic): {multi:>6}  ({100*multi/total:.1f}%)")
        print(f"  ─────────────────────────────────────────────")
        if total > 0:
            print(f"  Mean channels:            {np.mean(counts):>8.2f}")
            print(f"  Max channels:             {int(np.max(counts)):>6}")
            for k in sorted(set(counts.tolist())):
                n_k = int(np.sum(counts == k))
                print(f"  Channels == {k:<3}:            {n_k:>6}  ({100*n_k/total:.1f}%)")
        print(sep + "\n")

        return dict(total=total, zero=zero, single=single, multi=multi,
                    multi_pct=100 * multi / total if total > 0 else 0.0,
                    counts=counts)

    stats_main     = analyze_db(main_db_path,     os.path.basename(main_db_path))
    stats_filtered = analyze_db(filtered_db_path, os.path.basename(filtered_db_path))

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  COMPARISON: multi-channel counts in the two databases")
    print(sep)
    print(f"  {'':30s}  {'Main':>10}  {'Filtered':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Total events':30s}  {stats_main['total']:>10}  {stats_filtered['total']:>10}")
    print(f"  {'Channels > 1':30s}  {stats_main['multi']:>10}  {stats_filtered['multi']:>10}")
    print(f"  {'% multi-channel':30s}  {stats_main['multi_pct']:>9.1f}%  {stats_filtered['multi_pct']:>9.1f}%")
    print(sep + "\n")

    return dict(main=stats_main, filtered=stats_filtered)


AUDIO_BASE_PATH = r"Z:\recordings2026\audio\2026-05-27"
PADDING_SEC = 0.3

from scipy import signal as scipy_signal


def compute_spectrogram(audio: np.ndarray, sr: int,
                        nperseg: int = 1024,
                        fmin: int = 100, fmax: int = 8000) -> tuple:
    """Compute a dB-scaled spectrogram of `audio`, cropped to the [fmin, fmax] frequency band."""
    freqs, times, Sxx = scipy_signal.spectrogram(
        audio, fs=sr, nperseg=nperseg, noverlap=nperseg // 2,
        nfft=nperseg * 4, window='hann', scaling='density',
    )
    S_db = 20 * np.log10(Sxx + 1e-10)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[freq_mask], times, S_db[freq_mask, :]


def inspect_filtered_db(db_path: str,
                        audio_base: str = AUDIO_BASE_PATH,
                        padding_sec: float = PADDING_SEC) -> None:
    """
    Loop through every group in the filtered database, print its metadata,
    and for each involved channel plot the spectrogram (with start/peak/end
    markers and f0 line) while playing back the corresponding audio segment.
    """
    with h5py.File(db_path, 'r') as f:
        group_names = sorted(f.keys())
        n_total = len(group_names)
        print(f"[INFO] {n_total} events in the database\n")

        for i, gname in enumerate(group_names, start=1):
            grp = f[gname]

            raw_name          = grp.attrs.get('raw_name', '')
            ch                = int(grp['ch'][()])
            peak_time         = float(grp['peak_time'][()])
            start_peak        = float(grp['start_peak'][()])
            end_peak          = float(grp['end_peak'][()])
            hnr_level         = float(grp['hnr_level'][()])
            f0_mean           = float(grp['f0_mean'][()])
            precise_duration  = float(grp['precise_duration'][()])
            weighted_shr      = float(grp['weighted_shr'][()])
            max_aligned_peaks = int(grp['max_aligned_peaks'][()])
            ch_involved       = grp['channels_involved'][()].tolist()

            print(f"{'─'*70}")
            print(f"  [{i}/{n_total}]  {gname}")
            print(f"  raw_name        : {raw_name}")
            print(f"  ch (best)       : {ch}")
            print(f"  peak_time       : {peak_time:.3f} s")
            print(f"  start/end_peak  : {start_peak:.3f} – {end_peak:.3f} s")
            print(f"  precise_duration: {precise_duration*1000:.1f} ms")
            print(f"  f0_mean         : {f0_mean:.1f} Hz")
            print(f"  hnr_level       : {hnr_level:.2f} dB")
            print(f"  weighted_shr    : {weighted_shr:.4f}")
            print(f"  max_align_peaks : {max_aligned_peaks}")
            print(f"  channels_inv    : {ch_involved}")

            audio_path = os.path.join(audio_base, raw_name)
            if not os.path.exists(audio_path):
                print(f"  ⚠️  file not found: {audio_path} — skip\n")
                continue

            try:
                with sf.SoundFile(audio_path) as snd:
                    sr = snd.samplerate
                    total_samples = len(snd)
                    t_start = max(0.0, start_peak - padding_sec)
                    t_end   = min(total_samples / sr, end_peak + padding_sec)
                    s_start = int(t_start * sr)
                    s_end   = min(int(t_end * sr), total_samples)
                    snd.seek(s_start)
                    chunk = snd.read(s_end - s_start)
            except Exception as e:
                print(f"  ⚠️  error reading audio: {e} — skip\n")
                continue

            ch_involved = sorted(set(ch_involved))
            for ch in ch_involved:
                segment = chunk[:, ch] if chunk.ndim > 1 else chunk
                freqs, times, S_db = compute_spectrogram(segment, sr)
                times_abs = times + t_start

                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#1a1a1a')
                ax.set_facecolor('#1a1a1a')
                vmax = np.nanmax(S_db)
                vmin = vmax - 80
                ax.pcolormesh(times_abs, freqs, S_db, shading='gouraud',
                              cmap='inferno', norm=Normalize(vmin=vmin, vmax=vmax))
                for t, ls, lbl in [(start_peak, '--', 'start'), (peak_time, '-', 'peak'), (end_peak, '--', 'end')]:
                    ax.axvline(t, color='cyan', linewidth=1.0, linestyle=ls, alpha=0.7, label=lbl)
                ax.axhline(f0_mean, color='lime', linewidth=1.0, linestyle=':', alpha=0.8, label=f'f0 {f0_mean:.0f} Hz')
                ax.set_xlabel('Time (s)', color='white')
                ax.set_ylabel('Frequency (Hz)', color='white')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#444')
                ax.set_title(f"[{i}/{n_total}]  ch={ch}  f0={f0_mean:.0f} Hz  HNR={hnr_level:.1f} dB  dur={precise_duration*1000:.0f} ms  chs={ch_involved}", color='white', fontsize=9)
                ax.legend(fontsize=7, facecolor='#333', labelcolor='white', loc='upper right')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)} Hz"))
                try:
                    print(f"  playback segment   : {len(segment) / sr:.3f} s")
                    sd.play(segment, sr)
                    sd.wait()
                except Exception as e:
                    print(f"  ⚠️  error playing audio: {e}")
                plt.tight_layout()
                plt.show()
                plt.close(fig)

    print(f"\n[DONE] All {n_total} events inspected.")


def sound_validation(db_path: str,
                     audio_base_path: str,
                     listen_verbose: bool = True,
                     padding_sec: float = PADDING_SEC) -> None:
    """
    Interactive manual validation: shows the spectrogram of the strong
    channel and plays the audio segment. Writes the 'sound_validated'
    field (True/False).

    State logic:
      - field absent    → create a None placeholder → proceed
      - field == None    → proceed
      - field != None    → skip (already validated)
    """
    true_validated = 0
    false_validated = 0

    with h5py.File(db_path, 'r+') as f:
        group_names = sorted(f.keys())
        n_total = len(group_names)
        print(f"[INFO] {n_total} events in the database\n")

        for i, gname in enumerate(group_names, start=1):
            grp = f[gname]

            # ── state check ────────────────────────────────────────────────
            if 'sound_validated' in grp:
                current_val = grp['sound_validated'][()]
                already_done = False
                try:
                    v = str(current_val)
                    if v not in ('None', "b'None'", ''):
                        already_done = True
                except Exception:
                    pass
                if already_done:
                    print(f"[SKIP] [{i}/{n_total}] {gname}  (already validated: {current_val})")
                    if current_val in (True, 'True', "b'True'", '1'):
                        true_validated += 1
                    elif current_val in (False, 'False', "b'False'", '0'):
                        false_validated += 1
                    continue
            else:
                grp.create_dataset('sound_validated', data="None")

            # ── read fields ───────────────────────────────────────────────
            try:
                raw_name          = grp.attrs.get('raw_name', '')
                ch                = int(grp['ch'][()])
                peak_time         = float(grp['peak_time'][()])
                start_peak        = float(grp['start_peak'][()])
                end_peak          = float(grp['end_peak'][()])
                hnr_level         = float(grp['hnr_level'][()])
                f0_mean           = float(grp['f0_mean'][()])
                precise_duration  = float(grp['precise_duration'][()])
                weighted_shr      = float(grp['weighted_shr'][()])
                max_aligned_peaks = int(grp['max_aligned_peaks'][()])
                ch_involved_raw   = grp['channels_involved'][()]
                ch_involved = sorted(set(
                    ch_involved_raw.tolist()
                    if hasattr(ch_involved_raw, 'tolist')
                    else [int(ch_involved_raw)]
                ))
            except KeyError as e:
                print(f"[WARN] {gname}: missing field {e} — skip\n")
                continue

            print(f"\n{'─'*70}")
            print(f"  [{i}/{n_total}]  {gname}")
            print(f"  raw_name        : {raw_name}")
            print(f"  ch (best)       : {ch}")
            print(f"  peak_time       : {peak_time:.3f} s")
            print(f"  start/end_peak  : {start_peak:.3f} – {end_peak:.3f} s")
            print(f"  precise_duration: {precise_duration*1000:.1f} ms")
            print(f"  f0_mean         : {f0_mean:.1f} Hz")
            print(f"  hnr_level       : {hnr_level:.2f} dB")
            print(f"  weighted_shr    : {weighted_shr:.4f}")
            print(f"  max_align_peaks : {max_aligned_peaks}")
            print(f"  channels_inv    : {ch_involved}")

            audio_path = os.path.join(audio_base_path, raw_name)
            segment: Optional[np.ndarray] = None
            sr: int = 0
            t_start: float = 0.0

            if not os.path.exists(audio_path):
                print(f"  ⚠️  file not found: {audio_path} — spectrogram unavailable")
            else:
                try:
                    with sf.SoundFile(audio_path) as snd:
                        sr            = snd.samplerate
                        total_samples = len(snd)
                        t_start       = max(0.0, start_peak - padding_sec)
                        t_end         = min(total_samples / sr, end_peak + padding_sec)
                        s_start       = int(t_start * sr)
                        s_end         = min(int(t_end * sr), total_samples)
                        snd.seek(s_start)
                        chunk = snd.read(s_end - s_start)
                    segment = chunk[:, ch] if chunk.ndim > 1 else chunk
                except Exception as e:
                    print(f"  ⚠️  error reading audio: {e} — spectrogram unavailable")

            if segment is not None and sr > 0:
                freqs, times, S_db = compute_spectrogram(segment, sr)
                times_abs = times + t_start

                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#1a1a1a')
                ax.set_facecolor('#1a1a1a')
                vmax = np.nanmax(S_db)
                vmin = vmax - 80
                ax.pcolormesh(times_abs, freqs, S_db, shading='gouraud',
                              cmap='inferno', norm=Normalize(vmin=vmin, vmax=vmax))
                for t, ls, lbl in [(start_peak, '--', 'start'), (peak_time, '-', 'peak'), (end_peak, '--', 'end')]:
                    ax.axvline(t, color='cyan', linewidth=1.0, linestyle=ls, alpha=0.7, label=lbl)
                ax.axhline(f0_mean, color='lime', linewidth=1.0, linestyle=':', alpha=0.8, label=f'f0 {f0_mean:.0f} Hz')
                ax.set_xlabel('Time (s)', color='white')
                ax.set_ylabel('Frequency (Hz)', color='white')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#444')
                ax.set_title(f"[{i}/{n_total}]  ch={ch} (strong)  f0={f0_mean:.0f} Hz  HNR={hnr_level:.1f} dB  dur={precise_duration*1000:.0f} ms", color='white', fontsize=9)
                ax.legend(fontsize=7, facecolor='#333', labelcolor='white', loc='upper right')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)} Hz"))

                if listen_verbose:
                    try:
                        print(f"  ▶ playback segment : {len(segment) / sr:.3f} s")
                        sd.play(segment, sr)
                        sd.wait()
                    except Exception as e:
                        print(f"  ⚠️  error playing audio: {e}")

                plt.tight_layout()
                plt.show()
                plt.close(fig)

            # ── prompt ────────────────────────────────────────────────────
            while True:
                answer = input("\n  Sound validation passed? [Y / N / exit] > ").strip().lower()
                if answer in ('y', 'yes'):
                    del grp['sound_validated']
                    grp.create_dataset('sound_validated', data=True)
                    print(f"  ✅  {gname} → sound_validated = True")
                    break
                elif answer in ('n', 'no'):
                    del grp['sound_validated']
                    grp.create_dataset('sound_validated', data=False)
                    print(f"  ❌  {gname} → sound_validated = False")
                    break
                elif answer == 'exit':
                    print(f"  ⏸   Validation interrupted. '{gname}' left as None.")
                    print(f"\n[EXIT] Exiting after {i-1} events validated.\n")
                    return
                else:
                    print("  Invalid answer — type Y, N or exit.")

    print(f"\n[DONE] All {n_total} events processed.")
    print(f"  Validated True : {true_validated}/{n_total} ({100*true_validated/n_total:.1f}%)")
    print(f"  Validated False: {false_validated}/{n_total} ({100*false_validated/n_total:.1f}%)")


def analyze_sound_validated_candidates(db_path: str,
                                        audio_base_path: str,
                                        padding_sec: float = PADDING_SEC) -> None:
    """
    Iterate over all candidates with sound_validated == True and, for each,
    display the spectrogram and play back the audio for the strong channel
    plus every other involved channel.
    """
    with h5py.File(db_path, 'r') as f:
        group_names = sorted(f.keys())
        n_total = len(group_names)

        validated = []
        for gname in group_names:
            grp = f[gname]
            if 'sound_validated' not in grp:
                continue
            try:
                val = grp['sound_validated'][()]
                if val is True or str(val) in ('True', "b'True'", '1'):
                    validated.append(gname)
            except Exception:
                continue

        n_valid = len(validated)
        print(f"[INFO] {n_total} total groups in the database")
        print(f"[INFO] {n_valid} candidates with sound_validated = True\n")
        if n_valid == 0:
            print("  No validated candidates found.")
            return

        for i, gname in enumerate(validated, start=1):
            grp = f[gname]

            try:
                raw_name          = grp.attrs.get('raw_name', '')
                ch_strong         = int(grp['ch'][()])
                peak_time         = float(grp['peak_time'][()])
                start_peak        = float(grp['start_peak'][()])
                end_peak          = float(grp['end_peak'][()])
                hnr_level         = float(grp['hnr_level'][()])
                f0_mean           = float(grp['f0_mean'][()])
                precise_duration  = float(grp['precise_duration'][()])
                weighted_shr      = float(grp['weighted_shr'][()])
                max_aligned_peaks = int(grp['max_aligned_peaks'][()])
                ch_involved_raw   = grp['channels_involved'][()]
                ch_involved = sorted(set(
                    ch_involved_raw.tolist()
                    if hasattr(ch_involved_raw, 'tolist')
                    else [int(ch_involved_raw)]
                ))
            except KeyError as e:
                print(f"[WARN] {gname}: missing field {e} — skip\n")
                continue

            print(f"\n{'═'*70}")
            print(f"  [{i}/{n_valid}]  {gname}  ✅ sound_validated")
            print(f"{'─'*70}")
            print(f"  raw_name        : {raw_name}")
            print(f"  ch (strong)     : {ch_strong}")
            print(f"  channels_inv    : {ch_involved}")
            print(f"  peak_time       : {peak_time:.3f} s")
            print(f"  start/end_peak  : {start_peak:.3f} – {end_peak:.3f} s")
            print(f"  precise_duration: {precise_duration*1000:.1f} ms")
            print(f"  f0_mean         : {f0_mean:.1f} Hz")
            print(f"  hnr_level       : {hnr_level:.2f} dB")
            print(f"  weighted_shr    : {weighted_shr:.4f}")
            print(f"  max_align_peaks : {max_aligned_peaks}")

            audio_path = os.path.join(audio_base_path, raw_name)
            if not os.path.exists(audio_path):
                print(f"  ⚠️  file not found: {audio_path} — skip\n")
                continue

            try:
                with sf.SoundFile(audio_path) as snd:
                    sr            = snd.samplerate
                    total_samples = len(snd)
                    t_start       = max(0.0, start_peak - padding_sec)
                    t_end         = min(total_samples / sr, end_peak + padding_sec)
                    s_start       = int(t_start * sr)
                    s_end         = min(int(t_end * sr), total_samples)
                    snd.seek(s_start)
                    chunk = snd.read(s_end - s_start)
            except Exception as e:
                print(f"  ⚠️  error reading audio: {e} — skip\n")
                continue

            other_channels  = [c for c in ch_involved if c != ch_strong]
            channels_to_show = [ch_strong] + other_channels

            for ch_idx, ch in enumerate(channels_to_show):
                is_strong = (ch == ch_strong)
                ch_label  = f"ch={ch} ★ strong" if is_strong else f"ch={ch}"

                if chunk.ndim > 1:
                    if ch < chunk.shape[1]:
                        segment = chunk[:, ch]
                    else:
                        print(f"  ⚠️  ch={ch} out of range ({chunk.shape[1]} channels) — skip")
                        continue
                else:
                    segment = chunk

                freqs, times, S_db = compute_spectrogram(segment, sr)
                times_abs = times + t_start

                fig, ax = plt.subplots(figsize=(15, 7))
                fig.patch.set_facecolor('#1a1a1a')
                ax.set_facecolor('#1a1a1a')
                vmax = np.nanmax(S_db)
                vmin = vmax - 80
                ax.pcolormesh(times_abs, freqs, S_db, shading='gouraud',
                              cmap='inferno', norm=Normalize(vmin=vmin, vmax=vmax))
                for t, ls, lbl in [(peak_time, '-', 'peak')]:
                    ax.axvline(t, color='cyan', linewidth=1.0, linestyle=ls, alpha=0.7, label=lbl)
                ax.axhline(f0_mean, color='lime', linewidth=1.0, linestyle=':', alpha=0.8, label=f'f0 {f0_mean:.0f} Hz')
                border_color = '#FFD700' if is_strong else '#888888'
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2.0 if is_strong else 1.0)
                ax.set_xlabel('Time (s)', color='white', fontsize=24)
                ax.set_ylabel('Frequency (Hz)', color='white', fontsize=24)
                ax.tick_params(colors='white', labelsize=20)
                # ax.set_title(f"[{i}/{n_valid}]  {ch_label}  |  f0={f0_mean:.0f} Hz  HNR={hnr_level:.1f} dB  dur={precise_duration*1000:.0f} ms  [{ch_idx+1}/{len(channels_to_show)}]", color='white', fontsize=9)
                ax.legend(fontsize=15, facecolor='#333', labelcolor='white', loc='upper right')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)} Hz"))
                try:
                    print(f"  ▶ [{ch_label}]  playback: {len(segment) / sr:.3f} s")
                    sd.play(segment, sr)
                    sd.wait()
                except Exception as e:
                    print(f"  ⚠️  error playing audio ch={ch}: {e}")
                plt.tight_layout()
                plt.show()
                plt.close(fig)

    print(f"\n[DONE] {n_valid} candidates analyzed.")


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO PLAYER  (tkinter + python-vlc)
# ─────────────────────────────────────────────────────────────────────────────

_VIDEO_SUFFIXES = ["_cam0_sync.mp4", "_cam1_sync.mp4", "_zoomed.mp4"]


def _play_video_blocking(video_path: str) -> None:
    """
    Open the video in a minimal tkinter player using python-vlc.

    Features:
      • Persistent window: stays open at the end of the video (frozen on the
        last frame), same behavior as VLC.
      • Play / Pause   — button and Spacebar
      • Restart        — ⏮ button or R key (restarts from the beginning)
      • Seek bar       — slider updated in real time; draggable
      • Current time / total duration label
      • Close the window to move on to the next video

    Dependencies:
      pip install python-vlc
      + VLC Media Player installed on the system (Windows/Linux/macOS)

    Fallback:
      If python-vlc is unavailable or VLC is not installed, falls back to
      subprocess with mpv/ffplay (previous behavior).
    """
    import platform
    import shutil

    if not os.path.exists(video_path):
        print(f"  ⚠️  clip not found: {video_path} — skip")
        return

    # ── try the tkinter+vlc player ────────────────────────────────────────
    try:
        import vlc
        import tkinter as tk
        from tkinter import ttk
        _play_vlc_tkinter(video_path)
        return
    except ImportError:
        print("  [INFO] python-vlc not found — trying system player...")
    except Exception as e:
        print(f"  [WARN] vlc/tkinter player failed ({e}) — trying system player...")

    # ── fallback: blocking external player ─────────────────────────────────
    system = platform.system()

    def try_player(cmd: list[str]) -> bool:
        exe = shutil.which(cmd[0])
        if exe is None:
            return False
        try:
            print(f"  ▶ opening with {cmd[0]}: {os.path.basename(video_path)}")
            subprocess.run([exe] + cmd[1:] + [video_path], check=False)
            return True
        except Exception as e:
            print(f"  ⚠️  {cmd[0]} failed: {e}")
            return False

    if try_player(["mpv", "--loop=inf", "--title=" + os.path.basename(video_path)]):
        return
    if try_player(["vlc", "--repeat", "--no-play-and-exit"]):
        return
    if try_player(["ffplay", "-loop", "0"]):
        return

    print(f"  ⚠️  No player available — opening with the system handler...")
    if system == "Windows":
        os.startfile(video_path)
        input("  [press ENTER after closing the player] ")
    elif system == "Darwin":
        subprocess.run(["open", video_path])
        input("  [press ENTER after closing the player] ")
    else:
        subprocess.run(["xdg-open", video_path])
        input("  [press ENTER after closing the player] ")


def _play_vlc_tkinter(video_path: str) -> None:
    """
    Minimal tkinter + python-vlc player.

    Layout:
      ┌──────────────────────────────────────────┐
      │              video frame                 │
      ├──────────────────────────────────────────┤
      │  00:03 / 00:07   ━━━━━━━━━━━━━━━━━━━━   │  ← seek bar
      │  [⏮ Restart]  [⏸ Pause]  [✕ Close]     │  ← buttons
      └──────────────────────────────────────────┘

    At the end of the video, the player stays open on the last frame;
    the user can Restart or close the window manually.
    """
    import vlc
    import tkinter as tk
    from tkinter import ttk
    import platform

    # ── create the VLC instance (quiet) ────────────────────────────────────
    instance = vlc.Instance("--no-xlib", "--quiet")
    player   = instance.media_player_new()
    media    = instance.media_new(video_path)
    player.set_media(media)

    # ── tkinter window ───────────────────────────────────────────────────
    root = tk.Tk()
    title = os.path.basename(video_path)
    root.title(title)
    root.configure(bg="#1a1a1a")
    root.resizable(True, True)
    root.attributes("-fullscreen", True)          # ← ADDED

    # video frame
    video_frame = tk.Frame(root, bg="black", width=900, height=506)
    video_frame.pack(fill=tk.BOTH, expand=True)
    video_frame.pack_propagate(False)

    # ── embed VLC in the frame ──────────────────────────────────────────────
    # Must wait for the frame to be rendered before passing the window handle
    root.update()
    system = platform.system()
    if system == "Windows":
        player.set_hwnd(video_frame.winfo_id())
    elif system == "Darwin":
        player.set_nsobject(video_frame.winfo_id())
    else:
        player.set_xwindow(video_frame.winfo_id())

    # ── controls ──────────────────────────────────────────────────────────
    controls = tk.Frame(root, bg="#1a1a1a")
    controls.pack(fill=tk.X, padx=8, pady=(4, 0))

    # time label
    time_label = tk.Label(controls, text="00:00 / 00:00", bg="#1a1a1a",
                          fg="#cccccc", font=("Consolas", 10))
    time_label.pack(side=tk.LEFT, padx=(0, 8))

    # seek bar
    seek_var   = tk.DoubleVar(value=0.0)
    seek_bar   = ttk.Scale(controls, from_=0, to=1000,
                           orient=tk.HORIZONTAL, variable=seek_var)
    seek_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

    # ── buttons ──────────────────────────────────────────────────────────
    btn_frame = tk.Frame(root, bg="#1a1a1a")
    btn_frame.pack(fill=tk.X, padx=8, pady=(4, 8))

    BTN_STYLE = dict(bg="#2e2e2e", fg="#eeeeee", relief=tk.FLAT,
                     font=("Segoe UI", 10), padx=12, pady=4,
                     activebackground="#444444", activeforeground="#ffffff",
                     cursor="hand2")

    # internal state
    state = {"seeking": False, "ended": False}

    def restart(_=None):
        state["ended"] = False
        player.stop()
        player.set_media(instance.media_new(video_path))
        player.play()
        btn_pause.config(text="⏸  Pause")

    def toggle_pause(_=None):
        if state["ended"]:
            restart()
            return
        if player.is_playing():
            player.pause()
            btn_pause.config(text="▶  Play")
        else:
            player.play()
            btn_pause.config(text="⏸  Pause")

    def on_close():
        player.stop()
        root.destroy()

    btn_restart = tk.Button(btn_frame, text="⏮  Restart", command=restart, **BTN_STYLE)
    btn_restart.pack(side=tk.LEFT, padx=(0, 6))

    btn_pause = tk.Button(btn_frame, text="⏸  Pause", command=toggle_pause, **BTN_STYLE)
    btn_pause.pack(side=tk.LEFT, padx=(0, 6))

    btn_close = tk.Button(btn_frame, text="✕  Close", command=on_close, **BTN_STYLE)
    btn_close.pack(side=tk.RIGHT)

    # ── seek bar: drag ──────────────────────────────────────────────────────
    def on_seek_press(_):
        state["seeking"] = True

    def on_seek_release(_):
        state["seeking"] = False
        pos = seek_var.get() / 1000.0
        player.set_position(pos)

    seek_bar.bind("<ButtonPress-1>",   on_seek_press)
    seek_bar.bind("<ButtonRelease-1>", on_seek_release)

    # ── keybinds ──────────────────────────────────────────────────────────
    root.bind("<space>", toggle_pause)
    root.bind("r",       restart)
    root.bind("R",       restart)
    root.bind("<Escape>", lambda _: on_close())

    # ── update loop ─────────────────────────────────────────────────────────
    def fmt_time(ms: int) -> str:
        s   = ms // 1000
        m   = s  // 60
        s  %= 60
        return f"{m:02d}:{s:02d}"

    def update_ui():
        if not root.winfo_exists():
            return

        vlc_state = player.get_state()

        # ── detect end of video ────────────────────────────────────────
        if vlc_state == vlc.State.Ended and not state["ended"]:
            state["ended"] = True
            # stays on the last frame without closing — identical to VLC
            btn_pause.config(text="▶  Play")

        # ── update seek bar and time label ──────────────────────────────
        if not state["seeking"]:
            pos      = player.get_position()           # 0.0 – 1.0
            duration = player.get_length()              # ms  (-1 if unknown)
            cur_ms   = player.get_time()                # ms

            if duration > 0:
                seek_var.set(pos * 1000)
                time_label.config(
                    text=f"{fmt_time(max(0, cur_ms))} / {fmt_time(duration)}"
                )

        root.after(200, update_ui)

    # ── start the video and the update loop ──────────────────────────────────
    player.play()
    print(f"  ▶ {title}  — close the window to continue")
    root.after(200, update_ui)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def video_validation(db_path: str,
                     clips_dir: str) -> None:
    """
    Iterate over every group in the database with sound_validated == True and
    request manual validation of the synchronized video clips.

    For each candidate, the three clips are shown in sequence:
        <id>_cam0_sync.mp4
        <id>_cam1_sync.mp4
        <id>_zoomed.mp4

    After watching all three, the user is asked whether to validate the video.
    The result is written to the 'video_validated' field of the HDF5 group.

    State logic (identical to sound_validation):
      - 'video_validated' absent   → create None placeholder → proceed
      - 'video_validated' == None   → proceed
      - 'video_validated' != None   → already validated → skip
    """
    true_validated  = 0
    false_validated = 0
    validated_true_candidates = []

    with h5py.File(db_path, 'r+') as f:
        group_names = sorted(f.keys())
        n_total = len(group_names)

        # ── pre-filter: only sound_validated == True ──────────────────────
        sound_valid_names = []
        for gname in group_names:
            grp = f[gname]
            if 'sound_validated' not in grp:
                continue
            try:
                val = grp['sound_validated'][()]
                if val is True or str(val) in ('True', "b'True'", '1'):
                    sound_valid_names.append(gname)
            except Exception:
                continue

        n_candidates = len(sound_valid_names)
        print(f"[INFO] {n_total} total groups in the database")
        print(f"[INFO] {n_candidates} candidates with sound_validated = True\n")

        if n_candidates == 0:
            print("  No sound-validated candidates found. Run sound validation first.")
            return

        for i, gname in enumerate(sound_valid_names, start=1):
            grp = f[gname]

            # ── check video_validated state ───────────────────────────────
            if 'video_validated' in grp:
                current_val = grp['video_validated'][()]
                already_done = False
                try:
                    v = str(current_val)
                    if v not in ('None', "b'None'", ''):
                        already_done = True
                except Exception:
                    pass
                if already_done:
                    print(f"[SKIP] [{i}/{n_candidates}] {gname}  (already video-validated: {current_val})")
                    if current_val in (True, 'True', "b'True'", '1'):
                        true_validated += 1
                        validated_true_candidates.append(gname)
                    elif current_val in (False, 'False', "b'False'", '0'):
                        false_validated += 1
                    continue
            else:
                grp.create_dataset('video_validated', data="None")

            # ── read informational fields ─────────────────────────────────
            try:
                ch               = int(grp['ch'][()])
                f0_mean          = float(grp['f0_mean'][()])
                precise_duration = float(grp['precise_duration'][()])
                hnr_level        = float(grp['hnr_level'][()])
            except KeyError:
                f0_mean = precise_duration = hnr_level = float('nan')

            print(f"\n{'─'*70}")
            print(f"  [{i}/{n_candidates}]  {gname}")
            print(f"  f0_mean         : {f0_mean:.1f} Hz")
            print(f"  hnr_level       : {hnr_level:.2f} dB")
            print(f"  precise_duration: {precise_duration*1000:.1f} ms")
            print(f"  channel         : {ch}")

            # ── read ch and choose the clip ─────────────────────────────────
            try:
                ch         = int(grp['ch'][()])
                f0_mean    = float(grp['f0_mean'][()])
                precise_duration = float(grp['precise_duration'][()])
                hnr_level  = float(grp['hnr_level'][()])
            except KeyError as e:
                f0_mean = precise_duration = hnr_level = float('nan')
                ch = -1
            # SHOW FULL FRAME

            # ch 0-15  → cam1_sync  |  ch 16-31 → cam0_sync
            if 0 <= ch <= 15:
                suffix   = "_cam1_sync.mp4"
                cam_label = "cam1"
            else:
                suffix   = "_cam0_sync.mp4"
                cam_label = "cam0"

            print(f"\n{'─'*70}")
            print(f"  [{i}/{n_candidates}]  {gname}")
            print(f"  f0_mean         : {f0_mean:.1f} Hz")
            print(f"  hnr_level       : {hnr_level:.2f} dB")
            print(f"  precise_duration: {precise_duration*1000:.1f} ms")
            print(f"  channel         : {ch}  →  {cam_label} -> look for label {ch+1}")

            # input(f"\n  ▶ Press ENTER to see the {cam_label} clip... ")

            clip_path = os.path.join(clips_dir, gname + suffix)

            if not os.path.exists(clip_path):
                print(f"  ⚠️  clip not found: {os.path.basename(clip_path)} — skip")
                clips_found = 0
            else:
                clips_found = 1
                print(f"\n  ── {os.path.basename(clip_path)}")
                _play_video_blocking(clip_path)
                print(f"  ✔  Clip closed.")

            # SHOW ZOOMED
            zoom_suffix = "_zoomed.mp4"
            zoom_clip_path = os.path.join(clips_dir, gname + zoom_suffix)
            if not os.path.exists(zoom_clip_path):
                print(f"  ⚠️  zoomed clip not found: {os.path.basename(zoom_clip_path)} — skip")
            else:
                clips_found += 1
                print(f"\n  ── {os.path.basename(zoom_clip_path)}")
                _play_video_blocking(zoom_clip_path)
                print(f"  ✔  Clip closed.")

            if clips_found == 0:
                print(f"  ⚠️  No clips found for {gname} in {clips_dir} — skip")
                continue

            # ── prompt ────────────────────────────────────────────────────
            while True:
                answer = input(
                    f"\n  Video validation passed? [{clips_found}/3 clips seen]  [Y / N / exit] > "
                ).strip().lower()

                if answer in ('y', 'yes'):
                    del grp['video_validated']
                    grp.create_dataset('video_validated', data=True)
                    true_validated += 1
                    print(f"  ✅  {gname} → video_validated = True")
                    break

                elif answer in ('n', 'no'):
                    del grp['video_validated']
                    grp.create_dataset('video_validated', data=False)
                    false_validated += 1
                    print(f"  ❌  {gname} → video_validated = False")
                    break

                elif answer == 'exit':
                    print(f"  ⏸   Validation interrupted. '{gname}' left as None.")
                    print(f"\n[EXIT] Exiting after {i-1} video-validated events.\n")
                    return

                else:
                    print("  Invalid answer — type Y, N or exit.")

    total_processed = true_validated + false_validated
    print(f"\n[DONE] {n_candidates} candidates processed.")
    if total_processed > 0:
        print(f"  Video validated True : {true_validated}/{total_processed} ({100*true_validated/total_processed:.1f}%)")
        print(f"  Video validated False: {false_validated}/{total_processed} ({100*false_validated/total_processed:.1f}%)")
        for gname in validated_true_candidates:
            print(f"    - {gname}")


def analyze_video_validated_candidates(db_path: str,
                                    clips_dir: str) -> None:
    """
    Iterate over all candidates with video_validated == True, print metadata,
    and play back the full-frame clip (cam0/cam1 depending on channel) and
    the zoomed clip for each.
    """
    with h5py.File(db_path, 'r') as f:
        group_names = sorted(f.keys())
        n_total = len(group_names)

        validated = []
        for gname in group_names:
            grp = f[gname]
            if 'video_validated' not in grp:
                continue
            try:
                val = grp['video_validated'][()]
                if val is True or str(val) in ('True', "b'True'", '1'):
                    validated.append(gname)
            except Exception:
                continue

        n_valid = len(validated)
        print(f"[INFO] {n_total} total groups in the database")
        print(f"[INFO] {n_valid} candidates with video_validated = True\n")

        if n_valid == 0:
            print("  No video-validated candidates found.")
            return

        for i, gname in enumerate(validated, start=1):
            grp = f[gname]

            try:
                raw_name          = grp.attrs.get('raw_name', '')
                ch                = int(grp['ch'][()])
                peak_time         = float(grp['peak_time'][()])
                start_peak        = float(grp['start_peak'][()])
                end_peak          = float(grp['end_peak'][()])
                hnr_level         = float(grp['hnr_level'][()])
                f0_mean           = float(grp['f0_mean'][()])
                precise_duration  = float(grp['precise_duration'][()])
                weighted_shr      = float(grp['weighted_shr'][()])
                max_aligned_peaks = int(grp['max_aligned_peaks'][()])
                ch_involved_raw   = grp['channels_involved'][()]
                ch_involved = sorted(set(
                    ch_involved_raw.tolist()
                    if hasattr(ch_involved_raw, 'tolist')
                    else [int(ch_involved_raw)]
                ))
            except KeyError as e:
                print(f"[WARN] {gname}: missing field {e} — skip\n")
                continue

            print(f"\n{'═'*70}")
            print(f"  [{i}/{n_valid}]  {gname}  ✅ video_validated")
            print(f"{'─'*70}")
            print(f"  raw_name        : {raw_name}")
            print(f"  ch              : {ch}")
            print(f"  channels_inv    : {ch_involved}")
            print(f"  peak_time       : {peak_time:.3f} s")
            print(f"  start/end_peak  : {start_peak:.3f} – {end_peak:.3f} s")
            print(f"  precise_duration: {precise_duration*1000:.1f} ms")
            print(f"  f0_mean         : {f0_mean:.1f} Hz")
            print(f"  hnr_level       : {hnr_level:.2f} dB")
            print(f"  weighted_shr    : {weighted_shr:.4f}")
            print(f"  max_align_peaks : {max_aligned_peaks}")

            clips_shown = 0

            # 1) correct FULL FRAME, only if ch is within the frame
            if 0 <= ch <= 15:
                full_suffix = "_cam1_sync.mp4"
                cam_label = "cam1"
                ch_in_frame = True
            elif 16 <= ch <= 31:
                full_suffix = "_cam0_sync.mp4"
                cam_label = "cam0"
                ch_in_frame = True
            else:
                full_suffix = None
                cam_label = "unknown"
                ch_in_frame = False

            if ch_in_frame and full_suffix is not None:
                full_clip_path = os.path.join(clips_dir, gname + full_suffix)
                print(f"  full frame      : {cam_label} (ch={ch})")

                if os.path.exists(full_clip_path):
                    print(f"\n  ── FULL FRAME: {os.path.basename(full_clip_path)}")
                    _play_video_blocking(full_clip_path)
                    print("  ✔  Full-frame closed.")
                    clips_shown += 1
                else:
                    print(f"  ⚠️  full-frame clip not found: {os.path.basename(full_clip_path)}")

            # 2) ZOOMED always, if it exists
            zoom_clip_path = os.path.join(clips_dir, gname + "_zoomed.mp4")
            if os.path.exists(zoom_clip_path):
                print(f"\n  ── ZOOMED: {os.path.basename(zoom_clip_path)}")
                _play_video_blocking(zoom_clip_path)
                print("  ✔  Zoomed closed.")
                clips_shown += 1
            else:
                print(f"  ⚠️  zoomed clip not found: {os.path.basename(zoom_clip_path)}")

            if clips_shown == 0:
                print(f"  ⚠️  No clips found for {gname}")
                continue

            input(f"\n  Press ENTER to move to the next candidate ({i}/{n_valid})...")

    print(f"\n[DONE] {n_valid} video-validated candidates analyzed.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Parse CLI arguments and dispatch to the requested subcommand."""
    parser = argparse.ArgumentParser(
        description="Filter and validate whoop candidates from an HDF5 database."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_filter = subparsers.add_parser("filter", help="Filter candidates by f0 / duration / HNR")
    p_filter.add_argument("date",        help="Date YYYY-MM-DD")
    p_filter.add_argument("output_path", help="Filtered output .h5")

    p_val = subparsers.add_parser("validate", help="Sound validation (spectrogram + audio)")
    p_val.add_argument("date", help="Date YYYY-MM-DD")
    p_val.add_argument("--no-listen", dest="listen", action="store_false",
                       help="Disable audio playback")

    p_vval = subparsers.add_parser("validate_video",
        help="Video validation for sound_validated=True candidates")
    p_vval.add_argument("date",      help="Date YYYY-MM-DD")
    p_vval.add_argument("clips_dir", help="Folder containing the video clips (.mp4)")

    p_insp = subparsers.add_parser("inspect", help="Inspect all channels of the filtered DB")
    p_insp.add_argument("date", help="Date YYYY-MM-DD")

    p_ana = subparsers.add_parser("analyze_sound_validated",
        help="Show spectrogram/audio for sound_validated=True candidates")
    p_ana.add_argument("date", help="Date YYYY-MM-DD")

    p_hnr = subparsers.add_parser("hnr", help="HNR statistics on the database")
    p_hnr.add_argument("date", help="Date YYYY-MM-DD")

    p_ch = subparsers.add_parser("channels",
        help="channels_involved distribution (main vs filtered)")
    p_ch.add_argument("date",             help="Date YYYY-MM-DD")
    p_ch.add_argument("filtered_db_path", help="Path to the filtered .h5 database")

    p_avv = subparsers.add_parser(
            "analyze_video_validated",
            help="Show clips for video_validated=True candidates"
        )
    p_avv.add_argument("date", help="Date YYYY-MM-DD")

    args = parser.parse_args()

    if args.command == "filter":
        db_path, _ = build_paths(args.date)
        print(f"[INFO] Input DB  : {db_path}")
        print(f"[INFO] Output DB : {args.output_path}")
        filter_h5(db_path, args.output_path)

    elif args.command == "validate":
        db_path, audio_base = build_paths(args.date)
        print(f"[INFO] DB         : {db_path}")
        print(f"[INFO] Audio base : {audio_base}")
        sound_validation(db_path=db_path, audio_base_path=audio_base, listen_verbose=args.listen)

    elif args.command == "validate_video":
        db_path, _ = build_paths(args.date)
        print(f"[INFO] DB         : {db_path}")
        print(f"[INFO] Clips dir  : {args.clips_dir}")
        video_validation(db_path=db_path, clips_dir=args.clips_dir)

    elif args.command == "inspect":
        db_path, audio_base = build_paths(args.date)
        print(f"[INFO] DB         : {db_path}")
        print(f"[INFO] Audio base : {audio_base}")
        inspect_filtered_db(db_path, audio_base=audio_base)

    elif args.command == "analyze_sound_validated":
        db_path, audio_base = build_paths(args.date)
        print(f"[INFO] DB         : {db_path}")
        print(f"[INFO] Audio base : {audio_base}")
        analyze_sound_validated_candidates(db_path=db_path, audio_base_path=audio_base)

    elif args.command == "hnr":
        db_path, _ = build_paths(args.date)
        print(f"[INFO] DB : {db_path}")
        HNR_estimation(db_path)

    elif args.command == "channels":
        main_db, _ = build_paths(args.date)
        print(f"[INFO] Main DB     : {main_db}")
        print(f"[INFO] Filtered DB : {args.filtered_db_path}")
        channels_involved_analysis(main_db, args.filtered_db_path)

    elif args.command == "analyze_video_validated":
        db_path, _ = build_paths(args.date)
        clips_dir = os.path.join(DB_BASE_DIR, args.date)
        print(f"[INFO] DB         : {db_path}")
        print(f"[INFO] Clips dir  : {clips_dir}")
        analyze_video_validated_candidates(db_path=db_path, clips_dir=clips_dir)


if __name__ == "__main__":
    main()


# ── usage examples ───────────────────────────────────────────────────────────
# python dsp-made-database_analysis.py filter              2026-05-27 Z:\postprocessing_results\2026-05-27\whoops_filtered.h5
#
# python dsp-made-database_analysis.py validate            2026-05-27
# python dsp-made-database_analysis.py validate            2026-05-27 --no-listen
#
# python dsp-made-database_analysis.py validate_video      2026-05-27 Z:\video_clips\2026-05-27
#
# python dsp-made-database_analysis.py inspect             2026-05-27
# python dsp-made-database_analysis.py analyze_sound_validated 2026-05-27
# python dsp-made-database_analysis.py analyze_video_validated 2026-05-27
# python dsp-made-database_analysis.py hnr                 2026-05-27
# python dsp-made-database_analysis.py channels            2026-05-27 Z:\postprocessing_results\2026-05-27\whoops_filtered.h5
