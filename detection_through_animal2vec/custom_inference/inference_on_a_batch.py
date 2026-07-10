"""Batch inference + aggregation + clustering for animal2vec stop-signal detection.
Includes f0 analysis (PitchDetector) on every candidate for outlier filtering.

Usage (example):
  python inference_on_a_batch_v2.py \
      --input-folder  /abyss/home/timon-nas/recordings2026/audio/2026-05-26 \
      --output-folder /abyss/home/timon-nas/postprocessing_results_animal2vec/2026-05-26 \
      --model-path    /abyss/home/mattia-montanari/runs/bee-finetune-checkpoint50/fold0/checkpoints/checkpoint_best.pt \
      --sample-rate 24000 --device cuda \
      --metric-threshold 0.70 --sigma-s 0.6 \
      --unique-values "['stop_signal_candidate']" \
      --overwrite-previous-preds True

Pipeline for each multichannel file:
    1. Split 32 channels -> temporary mono WAVs under /local (removed immediately after per-channel inference)
    2. Run animal2vec inference on each channel + f0 on each candidate -> csv_temp/<stem>_ch_<N>.csv
    3. aggregate_csvs()    -> temp.h5  (one HDF5 group per candidate, with f0)
    4. analyze_temp_file() -> results_<date>.h5  (temporal clustering + f0 outlier filtering, best per cluster)
    5. Clean up temp.h5
"""

import os
import re
import sys
import logging
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import h5py
import shutil
import gc
import psutil

# ---------------------------------------------------------------------------
# sys.path must be set BEFORE any fairseq/nn import.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_classes_root = Path(__file__).resolve().parents[1]
if str(_classes_root) not in sys.path:
    sys.path.insert(0, str(_classes_root))

import animal2vec_inference as a2v
from nn.utils import chunk_and_normalize
from classes.pitch_detector import PitchDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOLERANCE_SEC = 1.2
DEBUG_MAX_FILES = None
F0_OUTLIER_K = 2.5
MAD_FLOOR_HZ = 15.0

PITCH_PARAMS = dict(
    length_queue=5, hz_threshold=25, threshold_increment=1.3,
    padding_start_ms=5, padding_end_ms=25, freq_min=200, freq_max=600,
)

# ---------------------------------------------------------------------------
# Global verbosity and logger -- configured by main() after argparse.
# ---------------------------------------------------------------------------
VERBOSE = False
logger = logging.getLogger("inference")


def _setup_logger(output_folder: Path) -> None:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = output_folder / f"run_{ts}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    file_level = logging.DEBUG if VERBOSE else logging.ERROR
    console_level = logging.DEBUG if VERBOSE else logging.INFO

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    def _exc_hook(exc_type, exc_value, exc_tb):
        logger.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _exc_hook

    if VERBOSE:
        logger.info(f"Verbose mode ON | full log: {log_path}")
    else:
        logger.info(f"Verbose mode OFF | error log only: {log_path}")


def vprint(*args, **kwargs):
    logger.debug(" ".join(str(a) for a in args))


def iprint(*args, **kwargs):
    logger.info(" ".join(str(a) for a in args))


def event_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if VERBOSE:
        logger.info(msg)
    else:
        logger.debug(msg)


# ---------------------------------------------------------------------------
# Helpers audio
# ---------------------------------------------------------------------------
def list_wav_files(folder: Path) -> List[Path]:
    exts = ("*.wav", "*.WAV", "*.flac", "*.FLAC")
    files = []
    for e in exts:
        files.extend(sorted(folder.rglob(e)))
    return sorted(list(dict.fromkeys(files)))


def replace_ending(st: str, target_ending: str = ".csv") -> str:
    supported = ["WAV", "AIFF", "AIFC", "FLAC", "OGG", "MP3", "MAT"]
    st_ = st
    for ft in supported:
        regex_ft = "." + "".join(["[{}{}]".format(x.lower(), x.upper()) for x in ft])
        st_ = re.sub(regex_ft, target_ending, st_)
    return st_


# ---------------------------------------------------------------------------
# Helpers filename parsing
# ---------------------------------------------------------------------------
def parse_filename(filename: str) -> Dict[str, str]:
    pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}_\d{2}_\d{2}\.\d+)Z'
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")
    return {
        "date": match.group(1),
        "time": match.group(2).replace("_", ":"),
    }


def _channel_from_csv(csv_filename: str) -> int:
    m = re.search(r'_ch_(\d+)\.csv$', csv_filename)
    if not m:
        raise ValueError(f"Cannot extract channel from '{csv_filename}'")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Helpers timedelta parsing
# ---------------------------------------------------------------------------
def _td_to_sec(td_str: str) -> float:
    m = re.match(r'(\d+):(\d{2}):(\d{2}\.?\d*)', td_str.strip())
    if not m:
        raise ValueError(f"Cannot parse timedelta: '{td_str}'")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------
def _scalar_or_nan(val) -> np.ndarray:
    return np.array(val) if val is not None else np.array(np.nan)


def _array_or_empty(val) -> np.ndarray:
    return np.array(val) if val is not None else np.array([])


def _print_h5_summary(hdf5_path: str, label: str = "") -> None:
    if not VERBOSE:
        return
    if not os.path.exists(hdf5_path):
        vprint(f"[H5 summary] file not found: {hdf5_path}")
        return
    with h5py.File(hdf5_path, 'r') as f:
        keys = list(f.keys())
    if not keys:
        vprint(f"[H5 summary {label}] empty (0 groups)")
        return
    vprint(f"[H5 summary {label}] {len(keys)} gruppi in {os.path.basename(hdf5_path)}:")
    header = f"  {'group_name':<70} {'ch':>4}  {'peak_time':>10}  {'cue_level':>10}  {'f0':>8}  {'channels_involved':<20}"
    vprint(header)
    vprint("  " + "-" * (len(header) - 2))
    with h5py.File(hdf5_path, 'r') as f:
        for gname in sorted(f.keys()):
            grp = f[gname]
            ch = int(grp['ch'][()]) if 'ch' in grp else -1
            pt = float(grp['peak_time'][()]) if 'peak_time' in grp else float('nan')
            cue = float(grp['cue_level'][()]) if 'cue_level' in grp else float('nan')
            f0 = float(grp['f0'][()]) if 'f0' in grp else float('nan')
            ch_str = str(grp['channels_involved'][()].tolist()) if 'channels_involved' in grp else "N/A"
            pt_str = f"{pt:>10.3f}" if not np.isnan(pt) else f"{'nan':>10}"
            cue_str = f"{cue:>10.4f}" if not np.isnan(cue) else f"{'nan':>10}"
            f0_str = f"{f0:>8.1f}" if not np.isnan(f0) else f"{'nan':>8}"
            vprint(f"  {gname:<70} {ch:>4}  {pt_str}  {cue_str}  {f0_str}  {ch_str:<20}")


def _write_candidate_to_h5(hdf5_path: str, group_name: str,
                           date: str, time: str, raw_name: str,
                           ch: int, peak_time: float,
                           precise_start_peak: float, precise_end_peak: float,
                           precise_duration: float, cue_level: float,
                           f0: Optional[float]) -> None:
    with h5py.File(hdf5_path, 'a') as hf:
        if group_name in hf:
            vprint(f"  ⚠️  {group_name} already exists, skipping")
            return
        grp = hf.create_group(group_name)
        grp.attrs['date'] = date
        grp.attrs['time'] = time
        grp.attrs['raw_name'] = raw_name
        for name, val in dict(
            ch=ch, peak_time=peak_time,
            precise_start_peak=precise_start_peak,
            precise_end_peak=precise_end_peak,
            precise_duration=precise_duration,
            cue_level=cue_level, f0=f0,
        ).items():
            grp.create_dataset(name, data=_scalar_or_nan(val))
        grp.create_dataset('channels_involved', data=_array_or_empty(None))


# ---------------------------------------------------------------------------
# f0 estimation su un segmento numpy (in-memory, zero I/O)
# ---------------------------------------------------------------------------
def _estimate_f0(wav_np: np.ndarray, sr: int,
                 start_s: float, end_s: float) -> Optional[float]:
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    segment = wav_np[start_sample:end_sample]
    if len(segment) < 64:
        return None
    try:
        pitch_detector = PitchDetector(segment, sr)
        f0, _, _, _, whoop_pitch_info = pitch_detector.estimate_f0(**PITCH_PARAMS)
        if whoop_pitch_info is None:
            return None
        return float(f0) if f0 is not None else None
    except Exception as e:
        vprint(f"  ⚠️  f0 estimation error: {e}")
        return None


# ---------------------------------------------------------------------------
# aggregate_csvs: CSV -> temp.h5
# ---------------------------------------------------------------------------
def aggregate_csvs(csv_folder: Path, multich_stem: str,
                   temp_hdf5_path: str,
                   wav_data: Optional[np.ndarray],
                   wav_sr: Optional[int]) -> int:
    pattern = re.compile(rf'^{re.escape(multich_stem)}_ch_(\d+)\.csv$')
    csv_files = sorted(f for f in os.listdir(str(csv_folder)) if pattern.match(f))

    vprint(f"[aggregate_csvs] CSV trovati per '{multich_stem}':")
    if not csv_files:
        logger.warning("  No CSV files found in aggregate_csvs")
        return 0
    for cf in csv_files:
        vprint(f"  - {cf}")

    if os.path.exists(temp_hdf5_path):
        os.remove(temp_hdf5_path)
    with h5py.File(temp_hdf5_path, 'w'):
        pass
    vprint("[aggregate_csvs] temp.h5 created empty")

    dt = parse_filename(multich_stem)
    raw_name_wav = multich_stem + '.wav'
    total = 0

    for csv_file in csv_files:
        ch = _channel_from_csv(csv_file)
        csv_path = csv_folder / csv_file
        rows = []
        with open(str(csv_path), 'r') as fh:
            fh.readline()
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 6:
                    rows.append(parts)

        vprint(f"  CH {ch:02d} | {csv_file} | {len(rows)} rows")

        for parts in rows:
            start_s = _td_to_sec(parts[1])
            duration_s = _td_to_sec(parts[2])
            end_s = start_s + duration_s
            peak_time = start_s + duration_s / 2.0
            cue_level = float(parts[5].strip())

            f0 = None
            if wav_data is not None and wav_sr is not None:
                ch_signal = wav_data[:, ch] if wav_data.ndim == 2 else wav_data
                f0 = _estimate_f0(ch_signal, wav_sr, start_s, end_s)

            f0_str = f"{f0:.1f} Hz" if f0 is not None else "None"
            group_name = f"{multich_stem}_ch_{ch}_peak_{peak_time:.6f}"
            vprint(
                f"    → ch={ch:02d}  start={start_s:.3f}s  dur={duration_s:.3f}s  "
                f"peak={peak_time:.3f}s  cue={cue_level:.4f}  f0={f0_str}"
            )

            _write_candidate_to_h5(
                hdf5_path=temp_hdf5_path,
                group_name=group_name,
                date=dt['date'], time=dt['time'], raw_name=raw_name_wav,
                ch=ch, peak_time=peak_time,
                precise_start_peak=start_s, precise_end_peak=end_s,
                precise_duration=duration_s, cue_level=cue_level,
                f0=f0,
            )
            total += 1

    vprint(f"[aggregate_csvs] Totale candidati scritti in temp.h5: {total}")
    _print_h5_summary(temp_hdf5_path, label="after aggregate_csvs")
    return total


# ---------------------------------------------------------------------------
# Filtraggio outlier f0 con MAD
# ---------------------------------------------------------------------------
def _filter_f0_outliers(cluster: list, hdf5_path: str,
                        k: float = F0_OUTLIER_K) -> list:
    f0_values = []
    with h5py.File(hdf5_path, 'r') as hf:
        for c in cluster:
            grp = hf[c['name']]
            f0 = float(grp['f0'][()]) if 'f0' in grp else np.nan
            f0_values.append(f0)

    f0_arr = np.array(f0_values)
    valid_mask = ~np.isnan(f0_arr)

    vprint(f"  [f0 filter]  {valid_mask.sum()}/{len(cluster)} valori f0 validi  (k={k})")
    for c, f0 in zip(cluster, f0_values):
        vprint(f"    ch={c['ch']:02d}  peak={c['peak_time']:.3f}s  f0={'NaN' if np.isnan(f0) else f'{f0:.1f} Hz'}")

    if valid_mask.sum() < 2:
        vprint("  [f0 filter]  < 2 valori validi → filtro saltato")
        return cluster

    valid_f0 = f0_arr[valid_mask]
    median_f0 = np.median(valid_f0)
    mad_raw = np.median(np.abs(valid_f0 - median_f0))
    mad = max(mad_raw, MAD_FLOOR_HZ)
    threshold = k * mad

    vprint(f"  [f0 filter]  median={median_f0:.1f} Hz  MAD_raw={mad_raw:.1f}  MAD_eff={mad:.1f}  threshold=±{threshold:.1f} Hz")

    filtered = []
    for c, f0 in zip(cluster, f0_values):
        if np.isnan(f0) or abs(f0 - median_f0) > threshold:
            reason = "NaN" if np.isnan(f0) else f"|{f0:.1f}-{median_f0:.1f}|={abs(f0-median_f0):.1f} > {threshold:.1f}"
            vprint(f"    ✗ REMOVED  ch={c['ch']:02d}  ({reason})")
            with h5py.File(hdf5_path, 'a') as hf:
                if c['name'] in hf:
                    del hf[c['name']]
        else:
            vprint(f"    ✓ OK       ch={c['ch']:02d}  f0={f0:.1f} Hz")
            filtered.append(c)

    if not filtered:
        logger.warning("  [f0 filter] tutti rimossi → ripristino cluster originale")
        return cluster

    vprint(f"  [f0 filter]  {len(filtered)}/{len(cluster)} surviving members")
    return filtered


# ---------------------------------------------------------------------------
# analyze_temp_file: clustering + filtro f0 -> results_<date>.h5
# ---------------------------------------------------------------------------
def analyze_temp_file(temp_hdf5_path: str, results_hdf5_path: str) -> None:
    vprint(f"[analyze_temp_file] Starting clustering (tolerance={TOLERANCE_SEC}s)")

    cluster_idx = 0
    while True:
        with h5py.File(temp_hdf5_path, 'r') as hf:
            group_names = list(hf.keys())
        if not group_names:
            break

        ref_name = group_names[0]
        with h5py.File(temp_hdf5_path, 'r') as hf:
            ref_peak = float(hf[ref_name]['peak_time'][()])

        cluster = []
        with h5py.File(temp_hdf5_path, 'r') as hf:
            for name in group_names:
                grp = hf[name]
                pt        = float(grp['peak_time'][()])
                cue_level = float(grp['cue_level'][()]) if 'cue_level' in grp else np.nan
                ch        = int(grp['ch'][()]) if 'ch' in grp else -1
                if abs(pt - ref_peak) <= TOLERANCE_SEC:
                    cluster.append({'name': name, 'peak_time': pt,
                                    'cue_level': cue_level, 'ch': ch})

        cluster_idx += 1
        vprint(f"[cluster #{cluster_idx}]  ref_peak={ref_peak:.3f}s  ->  {len(cluster)} member(s)")

        cluster = _filter_f0_outliers(cluster, temp_hdf5_path)

        if not cluster:
            logger.warning("  empty cluster after f0 filter, skipping")
            continue

        channels_involved = sorted(c['ch'] for c in cluster)
        best = max(cluster, key=lambda c: c['cue_level'] if not np.isnan(c['cue_level']) else -np.inf)

        # Read the best candidate's f0 for the final filter.
        with h5py.File(temp_hdf5_path, 'r') as hf:
            best_grp  = hf[best['name']]
            best_f0   = float(best_grp['f0'][()]) if 'f0' in best_grp else np.nan
            best_cue  = float(best_grp['cue_level'][()]) if 'cue_level' in best_grp else np.nan

        # Final filter: f0 must be valid and cue must be >= 0.3.
        if np.isnan(best_f0) or best_cue < 0.3:
            reason = f"f0=NaN" if np.isnan(best_f0) else f"cue={best_cue:.4f} < 0.3"
            event_print(f"✗ rejected  t={ref_peak:.3f}s  chs={channels_involved}  "
                        f"best_ch={best['ch']}  {reason}")
            with h5py.File(temp_hdf5_path, 'a') as hf:
                for c in cluster:
                    if c['name'] in hf:
                        del hf[c['name']]
            continue

        f0_s = f"{best_f0:.1f} Hz"
        event_print(f"✓ event  t={ref_peak:.3f}s  chs={channels_involved}  "
                    f"best_ch={best['ch']}  cue={best_cue:.4f}  f0={f0_s}")

        # Copy the surviving candidate into the final results HDF5 file.
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
            else:
                vprint(f"  ⚠️  {result_name} already in results.h5, skipping")

        with h5py.File(temp_hdf5_path, 'a') as hf:
            for c in cluster:
                if c['name'] in hf:
                    del hf[c['name']]

        with h5py.File(temp_hdf5_path, 'r') as hf:
            remaining = len(hf.keys())
        vprint(f"  remaining temp.h5 groups: {remaining}")

    vprint(f"[analyze_temp_file] exhausted after {cluster_idx} cluster(s)")
    _print_h5_summary(results_hdf5_path, label="results.h5 AFTER analyze_temp_file")

# ---------------------------------------------------------------------------
# Core inference (in-memory, model already loaded)
# ---------------------------------------------------------------------------
def run_inference_in_memory(model, device: str, file_path: Path,
                            csv_out_dir: Path, inf_args,
                            wav_np: Optional[np.ndarray] = None) -> None:
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader

    unique_labels = eval(inf_args.unique_values)
    n_classes = len(unique_labels)

    dataset = a2v.FileFolderDataset(
        str(file_path),
        sample_rate=inf_args.sample_rate,
        channel_info=inf_args.channel_info if inf_args.channel_info else None,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    header = ["Name", "Start", "Duration", "Time Format", "Type", "Description", "F0"]

    method_dict = {
        "sigma_s": inf_args.sigma_s,
        "metric_threshold": inf_args.metric_threshold,
        "maxfilt_s": getattr(inf_args, "maxfilt_s", 0.1),
        "max_duration_s": getattr(inf_args, "max_duration_s", 0.5),
        "lowP": getattr(inf_args, "lowP", 0.125),
    }

    with torch.inference_mode():
        model.eval()
        for samples in dataloader:
            if not isinstance(samples, dict):
                continue

            current_file = samples["filename"][0]
            fname = replace_ending(os.path.basename(current_file), ".csv")
            file_out = csv_out_dir / fname

            if not inf_args.overwrite_previous_preds and file_out.exists():
                vprint(f"  Skipping (already exists): {file_out.name}")
                continue

            wav = samples["source"].squeeze()
            batched_wav = chunk_and_normalize(
                wav,
                segment_length=inf_args.segment_length,
                sample_rate=inf_args.sample_rate,
                normalize=False,
                max_batch_size=inf_args.batch_size,
            )

            all_time_intervals = [[] for _ in range(n_classes)]
            all_likelihoods = [[] for _ in range(n_classes)]

            for bi, batch in enumerate(batched_wav):
                if not torch.is_tensor(batch):
                    batch = torch.stack(batch)
                elif batch.dim() == 1:
                    batch = batch.view(1, -1)

                if inf_args.normalize:
                    batch = torch.stack([torch.nn.functional.layer_norm(x, x.shape) for x in batch])

                net_output = model(source=batch.to(device))

                if "linear_eval_projection" in net_output:
                    probs = torch.sigmoid(net_output["linear_eval_projection"].clone().detach().cpu()).float()
                elif "encoder_out" in net_output:
                    probs = torch.sigmoid(net_output["encoder_out"].clone().detach().cpu()).float()
                else:
                    logger.warning(f"  no logits key for batch {bi}, skipping")
                    continue

                fused = model.fuse_predict(
                    batch.size(-1), probs,
                    method_dict=method_dict,
                    method=inf_args.method,
                    multiplier=bi,
                    bs=inf_args.batch_size,
                )

                for sub_batch_ivs, sub_batch_lks in zip(fused[0], fused[2]):
                    for ci in range(n_classes):
                        if ci < len(sub_batch_ivs):
                            all_time_intervals[ci].extend(sub_batch_ivs[ci])
                        if ci < len(sub_batch_lks):
                            all_likelihoods[ci].extend(sub_batch_lks[ci])

            wav_for_f0 = wav_np if wav_np is not None else wav.numpy()

            rows = []
            for ci, label in enumerate(unique_labels):
                for iv, lk in zip(all_time_intervals[ci], all_likelihoods[ci]):
                    if len(iv) < 2:
                        continue
                    start_s = float(iv[0])
                    dur_s = float(iv[1] - iv[0])
                    if lk < inf_args.metric_threshold:
                        continue
                    if dur_s < inf_args.min_pred_length:
                        continue

                    f0 = _estimate_f0(wav_for_f0, inf_args.sample_rate, start_s, start_s + dur_s)
                    f0_str = f"{f0:.4f}" if f0 is not None else "nan"

                    rows.append([
                        label,
                        str(timedelta(seconds=start_s)),
                        str(timedelta(seconds=dur_s)),
                        "decimal", "", f"{lk:.4f}", f0_str,
                    ])

            df = pd.DataFrame(rows, columns=header)
            df.to_csv(str(file_out), sep='\t', index=False)
            vprint(f"  Written: {file_out.name}  ({len(df)} detections)")


# ---------------------------------------------------------------------------
# Noisy moments
# ---------------------------------------------------------------------------
def load_noisy_moments(csv_path: str) -> list:
    try:
        import zoneinfo
    except ImportError:
        from backports import zoneinfo

    from datetime import datetime, timezone
    TZ_KONSTANZ = zoneinfo.ZoneInfo("Europe/Berlin")

    noisy = []
    if not os.path.exists(csv_path):
        logger.warning(f"noisy_moments CSV not found: {csv_path}, no filter applied.")
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
            end_local = datetime.strptime(f"{date_str} {end_str}", "%Y-%m-%d %H:%M")
            start_utc = start_local.replace(tzinfo=TZ_KONSTANZ).astimezone(timezone.utc)
            end_utc = end_local.replace(tzinfo=TZ_KONSTANZ).astimezone(timezone.utc)
            noisy.append((start_utc, end_utc))

    iprint(f"Loaded {len(noisy)} noisy window(s) from {csv_path}")
    return noisy


def _is_noisy(audio_filename: str, noisy_windows: list) -> bool:
    from datetime import datetime, timezone
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
# Orchestration for a single multichannel file
# ---------------------------------------------------------------------------
def process_multich_file(fp: Path, tmpdir: Path, csv_temp: Path,
                         output_folder: Path, model, device: str,
                         inf_args, noisy_windows: list = []) -> None:
    if _is_noisy(fp.name, noisy_windows):
        iprint(f"⏭️  skipped (noisy window): {fp.name}")
        return

    try:
        info = sf.info(str(fp))
    except Exception as e:
        logger.warning(f"Skipping (unreadable): {e}")
        return

    n_ch = info.channels or 1
    iprint(f"{n_ch} ch  |  {info.duration:.1f}s  |  {info.samplerate} Hz")

    wav_data, wav_sr = sf.read(str(fp), always_2d=True)

    vprint(f"Splitting {n_ch} channels into tmpdir ...")
    mono_files = []
    if n_ch > 1:
        for ch in range(n_ch):
            out_name = f"{fp.stem}_ch_{ch:02d}{fp.suffix}"
            out_path = tmpdir / out_name
            sf.write(str(out_path), wav_data[:, ch], wav_sr)
            mono_files.append((out_path, ch))
    else:
        mono_files = [(fp, 0)]
    vprint(f"Split complete: {len(mono_files)} mono files")

    vprint(f"Inference ({n_ch} channels) ...")
    for mf, ch in mono_files:
        vprint(f"  -> {mf.name}")
        ch_signal = wav_data[:, ch] if wav_data.ndim == 2 else wav_data
        try:
            run_inference_in_memory(model, device, mf, csv_temp, inf_args, wav_np=ch_signal)
        except Exception as e:
            logger.error(f"ERROR inference ch={ch} file={fp.name}: {e}")
            logger.error(traceback.format_exc())
        finally:
            if mf != fp:
                mf.unlink(missing_ok=True)

    multich_stem = fp.stem
    dt = parse_filename(multich_stem)
    temp_h5 = str(output_folder / "temp.h5")
    results_h5 = str(output_folder / f"results_{dt['date']}.h5")

    n_candidates = aggregate_csvs(csv_temp, multich_stem, temp_h5, wav_data=wav_data, wav_sr=wav_sr)

    if n_candidates > 0:
        analyze_temp_file(temp_h5, results_h5)

    shutil.rmtree(csv_temp)
    csv_temp.mkdir(parents=True, exist_ok=True)
    vprint(f"[cleanup] csv_temp recreated empty")

    if os.path.exists(temp_h5):
        os.remove(temp_h5)
        vprint("temp.h5 removed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global VERBOSE

    parser = argparse.ArgumentParser(
        description="animal2vec batch inference + f0 + clustering for stop-signal detection"
    )
    parser.add_argument("--input-folder", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--sample-rate", default=24000, type=int)
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--unique-values", default="['stop_signal_candidate']", type=str)
    parser.add_argument("--method", default="avg", choices=["avg", "max", "canny"])
    parser.add_argument("--metric-threshold", default=0.70, type=float)
    parser.add_argument("--sigma-s", default=0.6, type=float)
    parser.add_argument("--overwrite-previous-preds", default=True, type=lambda v: v.lower() in ("true", "1", "yes"))
    parser.add_argument("--normalize", default=True, type=lambda v: v.lower() in ("true", "1", "yes"))
    parser.add_argument("--segment-length", default=10.0, type=float)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--min-pred-length", default=0.05, type=float)
    parser.add_argument("--channel-info", default="", type=str)
    parser.add_argument("--noisy-moments-csv", default="", type=str)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    VERBOSE = args.verbose

    if not args.input_folder.exists():
        raise SystemExit(f"Input folder not found: {args.input_folder}")
    if not args.model_path.exists():
        raise SystemExit(f"Model checkpoint not found: {args.model_path}")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    _setup_logger(args.output_folder)

    csv_temp = args.output_folder / "csv_temp"
    csv_temp.mkdir(parents=True, exist_ok=True)

    tmpdir = Path("/local/tmp_inference")
    tmpdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"tmpdir mono WAV: {tmpdir}  (su /local, non ephemeral-storage)")

    import torch
    from fairseq import checkpoint_utils

    logger.info(f"Loading model from {args.model_path} ...")
    try:
        models, _ = checkpoint_utils.load_model_ensemble([str(args.model_path)])
    except Exception as e:
        logger.critical(f"FATAL: unable to load the model: {e}")
        logger.critical(traceback.format_exc())
        raise SystemExit(1)
    logger.info("Model loaded")

    device_str = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    if args.device == "cuda" and device_str == "cpu":
        logger.warning("CUDA requested but not available -- falling back to CPU")
    logger.info(f"Moving model to {device_str} ...")
    model = models[0].to(device_str)
    model.eval()
    logger.info("Model ready")

    all_files = list_wav_files(args.input_folder)
    if not all_files:
        raise SystemExit(f"No WAV/FLAC files found in {args.input_folder}")

    if DEBUG_MAX_FILES is not None:
        files_to_process = all_files[:DEBUG_MAX_FILES]
        logger.info(f"[DEBUG] Processing {len(files_to_process)}/{len(all_files)} file(s).")
    else:
        files_to_process = all_files
        logger.info(f"Files to process: {len(files_to_process)}")

    noisy_windows = load_noisy_moments(args.noisy_moments_csv) if args.noisy_moments_csv else []

    try:
        for idx, fp in enumerate(files_to_process, 1):
            iprint("")
            iprint(f"[{idx}/{len(files_to_process)}] {fp.name}")
            iprint("=" * 70)
            try:
                process_multich_file(fp, tmpdir, csv_temp, args.output_folder, model, device_str, args, noisy_windows=noisy_windows)
            except Exception as e:
                logger.error(f"ERROR on {fp.name}: {e}")
                logger.error(traceback.format_exc())
            finally:
                gc.collect()
                torch.cuda.empty_cache()

                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                ram_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
                logger.info(f"  [mem] GPU alloc={allocated:.0f}MB  reserved={reserved:.0f}MB  RAM={ram_gb:.2f}GB")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.info(f"tmpdir {tmpdir} removed")

    logger.info("=" * 70)
    logger.info("DONE")


if __name__ == "__main__":
    main()