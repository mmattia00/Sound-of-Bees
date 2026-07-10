#!/usr/bin/env python3
"""
animal2vec-made-database_analysis.py

DESCRIPTION:
    Inspect an animal2vec HDF5 database: print fields, show spectrograms,
    play back audio channels, filter candidates, compute statistics, and
    run interactive manual sound validation.

USAGE:
    python animal2vec-made-database_analysis.py <command> [args...]

COMMANDS:
    inspect <db_path> <audio_base_path>
        Inspect the database (strong channel, all events). Prints info and
        shows spectrogram + playback for every event.

    filter <input_path> <output_path>
        Filter candidates (f0 non-NaN, cue >= 0.4, channels > 1) and write
        the surviving groups to a new HDF5 file.

    stats <db_path>
        Compute and print cue_level and channels_involved statistics.

    validate <db_path> <audio_base_path> [--no-listen]
        Interactive manual validation (spectrogram + audio playback).
        Modifies the database in-place by setting 'sound_validated'.

    analyze_sound_validated <db_path> <audio_base_path>
        Show spectrogram and play audio for every candidate with
        sound_validated == True.

    reset_validation <db_path>
        Reset the 'sound_validated' state for all groups in the database.
        Modifies the database in-place.

EXAMPLES:
    python animal2vec-made-database_analysis.py validate Z:\\postprocessing_results_animal2vec\\2026-05-27\\results_2026-05-27.h5 Z:\\recordings2026\\audio\\2026-05-27
    python animal2vec-made-database_analysis.py filter   candidates.h5 filtered.h5
    python animal2vec-made-database_analysis.py inspect  filtered.h5   E:/soundofbees/2025-09-15
    python animal2vec-made-database_analysis.py stats    filtered.h5
    python animal2vec-made-database_analysis.py validate filtered.h5   E:/soundofbees/2025-09-15
    python animal2vec-made-database_analysis.py validate filtered.h5   E:/soundofbees/2025-09-15 --no-listen
    python animal2vec-made-database_analysis.py analyze_sound_validated filtered.h5 E:/soundofbees/2025-09-15
"""

import argparse
import h5py
import numpy as np
import os
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from scipy import signal as scipy_signal


# AUDIO_BASE_PATH = r"Z:\recordings2026\audio\2026-05-29"
AUDIO_BASE_PATH = r"E:\soundofbees\2025-09-15"
PADDING_SEC = 0.3


def compute_spectrogram(audio: np.ndarray, sr: int,
                        nperseg: int = 1024,
                        fmin: int = 100, fmax: int = 8000) -> tuple:
    """Compute a dB-scaled spectrogram of the given audio, restricted to [fmin, fmax] Hz."""
    freqs, times, Sxx = scipy_signal.spectrogram(
        audio,
        fs=sr,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nperseg * 4,
        window='hann',
        scaling='density',
    )

    S_db = 20 * np.log10(Sxx + 1e-10)

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[freq_mask], times, S_db[freq_mask, :]


def inspect_animal2vec_db(db_path: str,
                          audio_base: str = AUDIO_BASE_PATH,
                          padding_sec: float = PADDING_SEC) -> None:
    """
    Iterate row by row through the animal2vec database, print the main
    fields, and show the spectrogram of the audio segment, loading only
    the strictly necessary chunk from the WAV file.
    Closing the window automatically advances to the next row.
    """
    with h5py.File(db_path, 'r') as f:
        group_names = sorted(f.keys())
        n_total = len(group_names)
        print(f"[INFO] {n_total} events in the database\n")

        for i, gname in enumerate(group_names, start=1):
            grp = f[gname]

            try:
                # ── attributes ───────────────────────────────────────────
                date_attr = grp.attrs.get('date', '')
                time_attr = grp.attrs.get('time', '')
                raw_name  = grp.attrs.get('raw_name', '')

                # ── datasets ──────────────────────────────────────────────
                best_ch          = int(grp['ch'][()])
                peak_time        = float(grp['peak_time'][()])
                start_peak       = float(grp['precise_start_peak'][()])
                end_peak         = float(grp['precise_end_peak'][()])
                precise_duration = float(grp['precise_duration'][()])
                cue_level        = float(grp['cue_level'][()])
                f0               = float(grp['f0'][()])

                ch_involved = sorted(set(
                    np.asarray(grp['channels_involved'][()]).astype(int).tolist()
                ))

            except KeyError as e:
                print(f"[WARN] {gname}: missing field {e} — skipping")
                continue

            # ── print info ────────────────────────────────────────────────
            print(f"{'─'*70}")
            print(f"  [{i}/{n_total}]  {gname}")
            print(f"  date             : {date_attr}")
            print(f"  time             : {time_attr}")
            print(f"  raw_name         : {raw_name}")
            print(f"  ch (best)        : {best_ch}")
            print(f"  peak_time        : {peak_time:.3f} s")
            print(f"  start/end_peak   : {start_peak:.3f} – {end_peak:.3f} s")
            print(f"  precise_duration : {precise_duration*1000:.1f} ms")
            print(f"  cue_level        : {cue_level:.4f}")
            if np.isnan(f0):
                print(f"  f0               : NaN / not estimated")
            else:
                print(f"  f0               : {f0:.1f} Hz")
            print(f"  channels_inv     : {ch_involved}")

            # ── load only the necessary chunk ───────────────────────────
            audio_path = os.path.join(audio_base, raw_name)
            if not os.path.exists(audio_path):
                print(f"  ⚠️  file not found: {audio_path} — skipping spectrogram\n")
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
                print(f"  ⚠️  error reading audio: {e} — skipping spectrogram\n")
                continue

            # ── use only the strong channel (best_ch) ───────────────────

            segment = chunk[:, best_ch]

            # ── spectrogram ──────────────────────────────────────────────
            freqs, times, S_db = compute_spectrogram(segment, sr)
            times_abs = times + t_start

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#1a1a1a')
            ax.set_facecolor('#1a1a1a')

            vmax = np.nanmax(S_db)
            vmin = vmax - 80
            ax.pcolormesh(times_abs, freqs, S_db,
                            shading='gouraud', cmap='inferno',
                            norm=Normalize(vmin=vmin, vmax=vmax))

            # reference lines: precise start/end and peak
            for t, ls, lbl in [
                (start_peak, '--', 'start'),
                (peak_time,  '-',  'peak'),
                (end_peak,   '--', 'end'),
            ]:
                ax.axvline(t, color='cyan', linewidth=1.0,
                            linestyle=ls, alpha=0.7, label=lbl)

            # f0 line (only if estimated)
            if not np.isnan(f0):
                ax.axhline(f0, color='lime', linewidth=1.0,
                            linestyle=':', alpha=0.8,
                            label=f'f0 {f0:.0f} Hz')

            ax.set_xlabel('Time (s)', color='white')
            ax.set_ylabel('Frequency (Hz)', color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

            title_f0 = "NaN" if np.isnan(f0) else f"{f0:.0f} Hz"
            ax.set_title(
                f"[{i}/{n_total}]  ch={best_ch}  f0={title_f0}  "
                f"cue={cue_level:.3f}  dur={precise_duration*1000:.0f} ms  "
                f"chs={ch_involved}",
                color='white', fontsize=9
            )
            ax.legend(fontsize=7, facecolor='#333', labelcolor='white',
                        loc='upper right')
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{int(v)} Hz"))

            # playback
            try:
                print(f"  playback ch={best_ch}  segment: {len(segment)/sr:.3f} s")
                sd.play(segment, sr)
                sd.wait()
            except Exception as e:
                print(f"  ⚠️  error during audio playback: {e}")

            plt.tight_layout()
            plt.show()   # blocks here — closing the window moves to the next channel/event
            plt.close(fig)

    print(f"\n[DONE] All {n_total} events inspected.")


def filter_h5_animal2vec(input_path: str, output_path: str) -> None:
    """
    Copy into the output HDF5 only the groups with non-NaN f0,
    cue_level >= 0.4, and channels_involved > 1.
    """
    total         = 0
    passed        = 0
    skipped_nan   = 0
    skipped_cue   = 0
    skipped_miss  = 0
    skipped_ch_involved    = 0

    with h5py.File(input_path, 'r') as fin, h5py.File(output_path, 'w') as fout:
        group_names = list(fin.keys())
        n_total = len(group_names)
        print(f"[INFO] Groups found in input: {n_total}")
        print(f"[INFO] Starting filter (f0 non-NaN, cue >= 0.4, channels > 1)...\n")

        for i, group_name in enumerate(group_names, start=1):
            total += 1
            grp_in = fin[group_name]

            try:
                f0        = float(grp_in['f0'][()])
                cue_level = float(grp_in['cue_level'][()])
                channels_involved = np.asarray(grp_in['channels_involved'][()]).astype(int).tolist()
            except KeyError as e:
                print(f"[WARN] {group_name}: missing field {e} — skip")
                skipped_miss += 1
                continue

            if np.isnan(f0):
                skipped_nan += 1
                continue

            if cue_level < 0.9:
                skipped_cue += 1
                continue

            # if len(channels_involved) <= 1:
            #     skipped_ch_involved += 1
            #     continue

            # ── full copy of the group ──────────────────────────────────
            grp_out = fout.create_group(group_name)
            for attr_key, attr_val in grp_in.attrs.items():
                grp_out.attrs[attr_key] = attr_val
            for ds_name in grp_in:
                grp_in.copy(ds_name, grp_out)

            passed += 1

            if i % 500 == 0:
                print(f"[{i:>6}/{n_total}] processed — passed so far: {passed}")

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  FILTER  —  {os.path.basename(input_path)}")
    print(sep)
    print(f"  Total candidates       : {total:>6}")
    print(f"  Passed                 : {passed:>6}  ({100*passed/total:.1f}%)")
    print(f"  Discarded (f0 NaN)     : {skipped_nan:>6}  ({100*skipped_nan/total:.1f}%)")
    print(f"  Discarded (cue < 0.4)  : {skipped_cue:>6}  ({100*skipped_cue/total:.1f}%)")
    print(f"  Discarded (channels<=1): {skipped_ch_involved:>6}  ({100*skipped_ch_involved/total:.1f}%)")
    print(f"  Skipped (missing field): {skipped_miss:>6}  ({100*skipped_miss/total:.1f}%)")
    print(sep)
    print(f"  Output written to      : {output_path}")
    print(sep + "\n")


def cue_and_channels_stats(db_path: str, verbose: bool = True) -> dict:
    """
    Read all groups in the HDF5 file and compute:
      - full statistics on cue_level (mean, median, std, percentiles,
        heuristic threshold for filtering, how many events survive
        increasing thresholds)
      - distribution of channels_involved
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"File not found: {db_path}")

    cue_values = []
    ch_counts  = []

    with h5py.File(db_path, 'r') as f:
        for gname in f.keys():
            grp = f[gname]

            # cue_level
            if 'cue_level' in grp:
                val = float(grp['cue_level'][()])
                if not np.isnan(val):
                    cue_values.append(val)

            # channels_involved
            if 'channels_involved' in grp:
                raw = grp['channels_involved'][()]
                arr = np.asarray(raw).astype(int)
                n = int(arr.shape[0]) if arr.ndim > 0 else int(arr)
                ch_counts.append(len(sorted(set(arr.tolist()))) if arr.ndim > 0 else 1)

    if not cue_values:
        print("  ⚠️  No cue_level values found.")
        return {}

    cue = np.array(cue_values)
    ch  = np.array(ch_counts) if ch_counts else np.array([])

    # ── cue statistics ──────────────────────────────────────────────────
    mean   = float(np.mean(cue))
    median = float(np.median(cue))
    std    = float(np.std(cue))
    cmin   = float(np.min(cue))
    cmax   = float(np.max(cue))
    p25    = float(np.percentile(cue, 25))
    p75    = float(np.percentile(cue, 75))
    p90    = float(np.percentile(cue, 90))
    p95    = float(np.percentile(cue, 95))
    p99    = float(np.percentile(cue, 99))

    # how many events pass increasing thresholds
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    survival   = {t: int(np.sum(cue >= t)) for t in thresholds}

    if verbose:
        sep = "=" * 60
        n   = len(cue)

        print(f"\n{sep}")
        print(f"  CUE LEVEL STATS  —  {os.path.basename(db_path)}")
        print(sep)
        print(f"  Total events:               {n:>6}")
        print(f"  Min / Max:                  {cmin:>7.4f}  /  {cmax:.4f}")
        print(f"  Mean:                       {mean:>10.4f}")
        print(f"  Median:                     {median:>10.4f}")
        print(f"  Std dev:                    {std:>10.4f}")
        print(f"  25th / 75th percentile:     {p25:>7.4f}  /  {p75:.4f}")
        print(f"  90th percentile:            {p90:>10.4f}")
        print(f"  95th percentile:            {p95:>10.4f}")
        print(f"  99th percentile:            {p99:>10.4f}")
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  Survival at increasing thresholds:")
        for t, count in survival.items():
            bar_len = int(40 * count / n)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"    cue >= {t:.1f}  →  {count:>6} / {n}  ({100*count/n:5.1f}%)  {bar}")

        print(f"\n{sep}")
        print(f"  CHANNELS INVOLVED  —  {os.path.basename(db_path)}")
        print(sep)

        if len(ch) > 0:
            total_ch = len(ch)
            print(f"  Total events:               {total_ch:>6}")
            print(f"  Mean channels per event:    {np.mean(ch):>10.2f}")
            print(f"  Max channels:                {int(np.max(ch)):>6}")
            print(f"  ─────────────────────────────────────────────────────")
            for k in sorted(set(ch.tolist())):
                n_k = int(np.sum(ch == k))
                bar_len = int(40 * n_k / total_ch)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(f"    ch == {k:<2}  →  {n_k:>6} / {total_ch}  ({100*n_k/total_ch:5.1f}%)  {bar}")
        else:
            print("  ⚠️  No channels_involved data found.")

        print(sep + "\n")

    return dict(
        total=len(cue),
        mean=mean, median=median, std=std,
        min=cmin, max=cmax,
        p25=p25, p75=p75, p90=p90, p95=p95, p99=p99,
        survival=survival,
        cue_values=cue,
        ch_counts=ch,
    )


def sound_validation_animal2vec(db_path: str,
                                 audio_base_path: str,
                                 listen_verbose: bool = True,
                                 padding_sec: float = PADDING_SEC) -> None:
    """
    Interactive manual validation for the animal2vec database.

    State logic for each group:
      - 'sound_validated' absent    → not yet validated → run validation
      - 'sound_validated' == None   → started but not completed → run validation
      - 'sound_validated' != None   → already validated (True/False) → skip
    """
    true_validated = 0
    false_validated = 0

    with h5py.File(db_path, 'r+') as f:
        group_names = sorted(f.keys())
        n_total = len(group_names)
        print(f"[INFO] {n_total} events in the database\n")

        for i, gname in enumerate(group_names, start=1):
            grp = f[gname]

            # ── check validation state ────────────────────────────────────
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

            # ── read fields ────────────────────────────────────────────────
            try:
                date_attr        = grp.attrs.get('date', '')
                time_attr        = grp.attrs.get('time', '')
                raw_name         = grp.attrs.get('raw_name', '')
                ch               = int(grp['ch'][()])
                peak_time        = float(grp['peak_time'][()])
                start_peak       = float(grp['precise_start_peak'][()])
                end_peak         = float(grp['precise_end_peak'][()])
                precise_duration = float(grp['precise_duration'][()])
                cue_level        = float(grp['cue_level'][()])
                f0               = float(grp['f0'][()])
                ch_involved      = sorted(set(
                    np.asarray(grp['channels_involved'][()]).astype(int).tolist()
                ))
            except KeyError as e:
                print(f"[WARN] {gname}: missing field {e} — skip\n")
                continue

            # ── print info ────────────────────────────────────────────────
            print(f"\n{'─'*70}")
            print(f"  [{i}/{n_total}]  {gname}")
            print(f"  date             : {date_attr}  {time_attr}")
            print(f"  raw_name         : {raw_name}")
            print(f"  ch (best)        : {ch}")
            print(f"  peak_time        : {peak_time:.3f} s")
            print(f"  start/end_peak   : {start_peak:.3f} – {end_peak:.3f} s")
            print(f"  precise_duration : {precise_duration*1000:.1f} ms")
            print(f"  cue_level        : {cue_level:.4f}")
            if np.isnan(f0):
                print(f"  f0               : NaN / not estimated")
            else:
                print(f"  f0               : {f0:.1f} Hz")
            print(f"  channels_inv     : {ch_involved}")

            # ── load audio chunk (strong channel only) ──────────────────
            audio_path = os.path.join(audio_base_path, raw_name)
            segment = None
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

            # ── spectrogram (strong channel only) ────────────────────────
            if segment is not None and sr > 0:
                freqs, times, S_db = compute_spectrogram(segment, sr)
                times_abs = times + t_start

                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#1a1a1a')
                ax.set_facecolor('#1a1a1a')

                vmax = np.nanmax(S_db)
                vmin = vmax - 80
                ax.pcolormesh(times_abs, freqs, S_db,
                              shading='gouraud', cmap='inferno',
                              norm=Normalize(vmin=vmin, vmax=vmax))

                for t, ls, lbl in [
                    (start_peak, '--', 'start'),
                    (peak_time,  '-',  'peak'),
                    (end_peak,   '--', 'end'),
                ]:
                    ax.axvline(t, color='cyan', linewidth=1.0,
                               linestyle=ls, alpha=0.7, label=lbl)

                if not np.isnan(f0):
                    ax.axhline(f0, color='lime', linewidth=1.0,
                               linestyle=':', alpha=0.8, label=f'f0 {f0:.0f} Hz')

                ax.set_xlabel('Time (s)', color='white')
                ax.set_ylabel('Frequency (Hz)', color='white')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#444')

                title_f0 = "NaN" if np.isnan(f0) else f"{f0:.0f} Hz"
                ax.set_title(
                    f"[{i}/{n_total}]  ch={ch} (strong)  f0={title_f0}  "
                    f"cue={cue_level:.3f}  dur={precise_duration*1000:.0f} ms",
                    color='white', fontsize=9,
                )
                ax.legend(fontsize=7, facecolor='#333', labelcolor='white',
                          loc='upper right')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda v, _: f"{int(v)} Hz"))

                if listen_verbose:
                    try:
                        print(f"  ▶ playback segment : {len(segment) / sr:.3f} s")
                        sd.play(segment, sr)
                        sd.wait()
                    except Exception as e:
                        print(f"  ⚠️  error during audio playback: {e}")

                plt.tight_layout()
                plt.show()
                plt.close(fig)

            # ── user input ─────────────────────────────────────────────────
            while True:
                answer = input(
                    "\n  Sound validation passed? [Y / N / exit] > "
                ).strip().lower()

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
                    print(f"\n[EXIT] Exiting after {i-1} validated events.\n")
                    return

                else:
                    print("  Invalid answer — type Y, N, or exit.")

    print(f"\n[DONE] All {n_total} events processed.")
    print(f"  Validated True : {true_validated}/{n_total} ({100*true_validated/n_total:.1f}%)")
    print(f"  Validated False: {false_validated}/{n_total} ({100*false_validated/n_total:.1f}%)")


def analyze_sound_validated_candidates_animal2vec(db_path: str,
                                                   audio_base_path: str,
                                                   padding_sec: float = PADDING_SEC) -> None:
    """
    Show spectrogram and play audio for every candidate with
    sound_validated == True (animal2vec database).

    For each event: strong channel first, then all other ch_involved.
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

            # ── read fields ────────────────────────────────────────────────
            try:
                date_attr        = grp.attrs.get('date', '')
                time_attr        = grp.attrs.get('time', '')
                raw_name         = grp.attrs.get('raw_name', '')
                ch_strong        = int(grp['ch'][()])
                peak_time        = float(grp['peak_time'][()])
                start_peak       = float(grp['precise_start_peak'][()])
                end_peak         = float(grp['precise_end_peak'][()])
                precise_duration = float(grp['precise_duration'][()])
                cue_level        = float(grp['cue_level'][()])
                f0               = float(grp['f0'][()])
                ch_involved      = sorted(set(
                    np.asarray(grp['channels_involved'][()]).astype(int).tolist()
                ))
            except KeyError as e:
                print(f"[WARN] {gname}: missing field {e} — skip\n")
                continue

            # ── print info ────────────────────────────────────────────────
            print(f"\n{'═'*70}")
            print(f"  [{i}/{n_valid}]  {gname}  ✅ sound_validated")
            print(f"{'─'*70}")
            print(f"  date             : {date_attr}  {time_attr}")
            print(f"  raw_name         : {raw_name}")
            print(f"  ch (strong)      : {ch_strong}")
            print(f"  channels_inv     : {ch_involved}")
            print(f"  peak_time        : {peak_time:.3f} s")
            print(f"  start/end_peak   : {start_peak:.3f} – {end_peak:.3f} s")
            print(f"  precise_duration : {precise_duration*1000:.1f} ms")
            print(f"  cue_level        : {cue_level:.4f}")
            if np.isnan(f0):
                print(f"  f0               : NaN / not estimated")
            else:
                print(f"  f0               : {f0:.1f} Hz")

            # ── load WAV chunk ────────────────────────────────────────────
            audio_path = os.path.join(audio_base_path, raw_name)
            if not os.path.exists(audio_path):
                print(f"  ⚠️  file not found: {audio_path} — skipping audio\n")
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
                print(f"  ⚠️  error reading audio: {e} — skipping audio\n")
                continue

            # ── strong channel first, then the others ───────────────────
            other_channels   = [c for c in ch_involved if c != ch_strong]
            channels_to_show = [ch_strong]

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

                # ── spectrogram ──────────────────────────────────────────
                freqs, times, S_db = compute_spectrogram(segment, sr)
                times_abs = times + t_start

                fig, ax = plt.subplots(figsize=(15, 7))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')

                vmax = np.nanmax(S_db)
                vmin = vmax - 80
                ax.pcolormesh(times_abs, freqs, S_db,
                              shading='gouraud', cmap='inferno',
                              norm=Normalize(vmin=vmin, vmax=vmax))

                for t, ls, lbl in [
                    (start_peak, '--', 'start'),
                    (peak_time,  '-',  'peak'),
                    (end_peak,   '--', 'end'),
                ]:
                    ax.axvline(t, color='cyan', linewidth=1.0,
                               linestyle=ls, alpha=0.7, label=lbl)

                if not np.isnan(f0):
                    ax.axhline(f0, color='lime', linewidth=1.0,
                               linestyle=':', alpha=0.8, label=f'f0 {f0:.0f} Hz')

                border_color = '#FFD700' if is_strong else '#888888'
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2.0 if is_strong else 1.0)

                ax.set_xlabel('Time (s)', color='black', fontsize=35)
                ax.set_ylabel('Frequency (Hz)', color='black', fontsize=35)
                ax.tick_params(colors='black', labelsize=30)

                title_f0 = "NaN" if np.isnan(f0) else f"{f0:.0f} Hz"
                # ax.set_title(
                #     f"[{i}/{n_valid}]  {ch_label}  |  "
                #     f"f0={title_f0}  cue={cue_level:.3f}  "
                #     f"dur={precise_duration*1000:.0f} ms  "
                #     f"[{ch_idx+1}/{len(channels_to_show)}]",
                #     color='white', fontsize=9,
                # )
                ax.legend(fontsize=25, facecolor='#333', labelcolor='white',
                          loc='upper right')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda v, _: f"{int(v)} Hz"))

                try:
                    print(f"  ▶ [{ch_label}]  playback: {len(segment) / sr:.3f} s")
                    sd.play(segment, sr)
                    sd.wait()
                except Exception as e:
                    print(f"  ⚠️  error during audio playback ch={ch}: {e}")

                plt.tight_layout()
                plt.show()
                plt.close(fig)

    print(f"\n[DONE] {n_valid} candidates analyzed.")


def reset_sound_validation_animal2vec(db_path: str) -> None:
    """
    Reset the manual validation state for all groups in the database.

    If the 'sound_validated' dataset exists, it is deleted (and would be
    recreated with value "None"). This returns all events to the
    'not validated' state.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"File not found: {db_path}")

    total = 0
    reset_count = 0
    created_count = 0

    with h5py.File(db_path, 'r+') as f:
        group_names = sorted(f.keys())
        n_total = len(group_names)
        print(f"[INFO] {n_total} groups found in the database")
        print(f"[INFO] Resetting 'sound_validated'...\n")

        for i, gname in enumerate(group_names, start=1):
            total += 1
            grp = f[gname]

            if 'sound_validated' in grp:
                del grp['sound_validated']
                # grp.create_dataset('sound_validated', data="None")
                reset_count += 1
            else:
                # grp.create_dataset('sound_validated', data="None")
                created_count += 1

            if i % 500 == 0:
                print(f"[{i:>6}/{n_total}] processed")

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  RESET SOUND VALIDATION — {os.path.basename(db_path)}")
    print(sep)
    print(f"  Total groups           : {total:>6}")
    print(f"  Resets performed        : {reset_count:>6}")
    print(f"  Created from scratch    : {created_count:>6}")
    print(f"  Output updated          : {db_path}")
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and validate animal2vec candidates from an HDF5 database."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── reset_validation ───────────────────────────────────────────────────
    p_reset = subparsers.add_parser(
        "reset_validation",
        help="Reset sound_validated for all groups in the database",
    )
    p_reset.add_argument("db_path", help="Path to the .h5 database (modified in-place)")

    # ── filter ────────────────────────────────────────────────────────────
    p_filter = subparsers.add_parser(
        "filter",
        help="Filter candidates (f0 non-NaN, cue >= 0.4, channels > 1)",
    )
    p_filter.add_argument("input_path",  help="Input .h5 database")
    p_filter.add_argument("output_path", help="Filtered output .h5 database")

    # ── inspect ───────────────────────────────────────────────────────────
    p_insp = subparsers.add_parser(
        "inspect",
        help="Inspect the database (strong channel, all events)",
    )
    p_insp.add_argument("db_path",         help="Path to the .h5 database")
    p_insp.add_argument("audio_base_path", help="Base folder for the WAV files")

    # ── stats ─────────────────────────────────────────────────────────────
    p_stats = subparsers.add_parser(
        "stats",
        help="cue_level and channels_involved statistics",
    )
    p_stats.add_argument("db_path", help="Path to the .h5 database")

    # ── validate ──────────────────────────────────────────────────────────
    p_val = subparsers.add_parser(
        "validate",
        help="Interactive manual validation (spectrogram + audio)",
    )
    p_val.add_argument("db_path",         help="Path to the .h5 database (modified in-place)")
    p_val.add_argument("audio_base_path", help="Base folder for the WAV files")
    p_val.add_argument(
        "--no-listen", dest="listen", action="store_false",
        help="Disable audio playback (spectrogram only)",
    )

    # ── analyze ───────────────────────────────────────────────────────────
    p_ana = subparsers.add_parser(
        "analyze_sound_validated",
        help="Show spectrogram and audio for candidates with sound_validated=True",
    )
    p_ana.add_argument("db_path",         help="Path to the .h5 database")
    p_ana.add_argument("audio_base_path", help="Base folder for the WAV files")

    # ── dispatch ──────────────────────────────────────────────────────────
    args = parser.parse_args()

    if args.command == "filter":
        filter_h5_animal2vec(args.input_path, args.output_path)

    elif args.command == "reset_validation":
        reset_sound_validation_animal2vec(args.db_path)

    elif args.command == "inspect":
        inspect_animal2vec_db(args.db_path, audio_base=args.audio_base_path)

    elif args.command == "stats":
        cue_and_channels_stats(args.db_path)

    elif args.command == "validate":
        sound_validation_animal2vec(
            db_path=args.db_path,
            audio_base_path=args.audio_base_path,
            listen_verbose=args.listen,
        )

    elif args.command == "analyze_sound_validated":
        analyze_sound_validated_candidates_animal2vec(
            db_path=args.db_path,
            audio_base_path=args.audio_base_path,
        )


if __name__ == "__main__":
    main()


# ── usage examples ────────────────────────────────────────────────────────

# python animal2vec-made-database_analysis.py validate Z:\postprocessing_results_animal2vec\2026-05-27\results_2026-05-27.h5 Z:\recordings2026\audio\2026-05-27
# python animal2vec-made-database_analysis.py filter   candidates.h5 filtered.h5
# python animal2vec-made-database_analysis.py inspect  filtered.h5   E:/soundofbees/2025-09-15
# python animal2vec-made-database_analysis.py stats    filtered.h5
# python animal2vec-made-database_analysis.py validate filtered.h5   E:/soundofbees/2025-09-15
# python animal2vec-made-database_analysis.py validate filtered.h5   E:/soundofbees/2025-09-15 --no-listen
# python animal2vec-made-database_analysis.py analyze_sound_validated filtered.h5 E:/soundofbees/2025-09-15