"""Batch inference, aggregation, and clustering for animal2vec stop-signal detection.
Multi-GPU with a persistent worker pool: the weights are loaded ONCE per GPU
at startup and remain in memory for the whole process lifetime.
The channels of each multichannel file are processed in parallel across the available GPUs.
This is the main difference from inference_on_a_batch.py, which runs the same
pipeline without multithreading and processes the files in a single worker.
It was kept as an experimental path, but it was not used as the final solution
because it did not show a meaningful speedup.

Usage (example):
  python inference_on_a_batch_v3_multigpu.py \
      --input-folder  /abyss/home/timon-nas/recordings2026/audio/2026-05-26 \
      --output-folder /abyss/home/timon-nas/postprocessing_results_animal2vec/2026-05-26 \
      --model-path    /abyss/home/mattia-montanari/runs/bee-finetune-checkpoint50/fold0/checkpoints/checkpoint_best.pt \
      --sample-rate 24000 --device cuda \
      --metric-threshold 0.70 --sigma-s 0.6 \
      --unique-values "[\'stop_signal_candidate\']" \
      --overwrite-previous-preds True

Pipeline:
    1. Startup: spawn N workers (one per GPU), each loading the weights on its own device
    2. For each multichannel file:
             a. Split 32 channels -> temporary mono WAVs
             b. Send 32 jobs to the job_queue (round-robin across GPUs)
             c. Wait for all 32 results from the result_queue  <- barrier
       d. aggregate_csvs()    → temp.h5
       e. analyze_temp_file() → results_<date>.h5
             f. Clean up csv_temp + temp.h5
    3. Shutdown: send the STOP signal to all workers
"""

import os
import re
import sys
import argparse
import tempfile
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Dict, List
import multiprocessing as mp

import numpy as np
import soundfile as sf
import h5py
import shutil


# ---------------------------------------------------------------------------
# sys.path must be set BEFORE any fairseq/nn import.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOLERANCE_SEC   = 1.2
DEBUG_MAX_FILES = 1   # None = all files
_STOP_SIGNAL    = "__STOP__"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def list_wav_files(folder: Path) -> List[Path]:
    exts = ("*.wav", "*.WAV", "*.flac", "*.FLAC")
    files = []
    for e in exts:
        files.extend(sorted(folder.rglob(e)))
    return sorted(list(dict.fromkeys(files)))


def split_channels_and_write(tmpdir: Path, src_path: Path) -> List[Path]:
    data, sr = sf.read(str(src_path), always_2d=True)
    out_paths = []
    for ch in range(data.shape[1]):
        out_name = f"{src_path.stem}_ch_{ch:02d}{src_path.suffix}"
        out_path = tmpdir / out_name
        sf.write(str(out_path), data[:, ch], sr)
        out_paths.append(out_path)
    return out_paths


def replace_ending(st: str, target_ending: str = ".csv") -> str:
    supported = ["WAV", "AIFF", "AIFC", "FLAC", "OGG", "MP3", "MAT"]
    st_ = st
    for ft in supported:
        regex_ft = "." + "".join(["[{}{}]".format(x.lower(), x.upper()) for x in ft])
        st_ = re.sub(regex_ft, target_ending, st_)
    return st_


# ---------------------------------------------------------------------------
# Filename parsing helpers
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
# Timedelta parsing helpers
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
    if not os.path.exists(hdf5_path):
        print(f"  [H5 summary] file not found: {hdf5_path}")
        return
    with h5py.File(hdf5_path, 'r') as f:
        keys = list(f.keys())
    if not keys:
        print(f"  [H5 summary{f' {label}' if label else ''}] empty (0 groups)")
        return
    print(f"  [H5 summary{f' {label}' if label else ''}] {len(keys)} groups in {os.path.basename(hdf5_path)}:")
    header = f"    {'group_name':<70} {'ch':>4}  {'peak_time':>10}  {'cue_level':>10}  {'channels_involved':<20}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    with h5py.File(hdf5_path, 'r') as f:
        for gname in sorted(f.keys()):
            grp   = f[gname]
            ch    = int(grp['ch'][()]) if 'ch' in grp else -1
            pt    = float(grp['peak_time'][()]) if 'peak_time' in grp else float('nan')
            cue   = float(grp['cue_level'][()]) if 'cue_level' in grp else float('nan')
            if 'channels_involved' in grp:
                arr    = grp['channels_involved'][()]
                ch_str = str(arr.tolist()) if arr.size > 0 else "[]"
            else:
                ch_str = "N/A"
            pt_str  = f"{pt:>10.3f}" if not np.isnan(pt)  else f"{'nan':>10}"
            cue_str = f"{cue:>10.4f}" if not np.isnan(cue) else f"{'nan':>10}"
            print(f"    {gname:<70} {ch:>4}  {pt_str}  {cue_str}  {ch_str:<20}")


def _write_candidate_to_h5(hdf5_path, group_name, date, time, raw_name,
                            ch, peak_time, precise_start_peak,
                            precise_end_peak, precise_duration, cue_level):
    with h5py.File(hdf5_path, 'a') as f:
        if group_name in f:
            print(f"    ⚠️  {group_name} already exists, skipping")
            return
        grp = f.create_group(group_name)
        grp.attrs['date']     = date
        grp.attrs['time']     = time
        grp.attrs['raw_name'] = raw_name
        for name, val in dict(
            ch=ch, peak_time=peak_time,
            precise_start_peak=precise_start_peak,
            precise_end_peak=precise_end_peak,
            precise_duration=precise_duration,
            cue_level=cue_level,
        ).items():
            grp.create_dataset(name, data=_scalar_or_nan(val))
        grp.create_dataset('channels_involved', data=_array_or_empty(None))


# ---------------------------------------------------------------------------
# aggregate_csvs + analyze_temp_file  (invariati)
# ---------------------------------------------------------------------------
def aggregate_csvs(csv_folder: Path, multich_stem: str, temp_hdf5_path: str) -> int:
    pattern   = re.compile(rf'^{re.escape(multich_stem)}_ch_(\d+)\.csv$')
    csv_files = sorted(f for f in os.listdir(str(csv_folder)) if pattern.match(f))

    print(f"\n  [aggregate_csvs] CSV files found for '{multich_stem}':")
    if not csv_files:
        print("    ⚠️  No CSV files found")
        return 0
    for cf in csv_files:
        print(f"    - {cf}")

    if os.path.exists(temp_hdf5_path):
        os.remove(temp_hdf5_path)
    with h5py.File(temp_hdf5_path, 'w'):
        pass
    print("  [aggregate_csvs] temp.h5 created empty")

    dt           = parse_filename(multich_stem)
    raw_name_wav = multich_stem + '.wav'
    total        = 0

    for csv_file in csv_files:
        ch       = _channel_from_csv(csv_file)
        csv_path = csv_folder / csv_file
        rows     = []
        with open(str(csv_path), 'r') as fh:
            fh.readline()
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 6:
                    rows.append(parts)

        print(f"    CH {ch:02d} | {csv_file} | {len(rows)} rows")
        for parts in rows:
            start_s    = _td_to_sec(parts[1])
            duration_s = _td_to_sec(parts[2])
            end_s      = start_s + duration_s
            peak_time  = start_s + duration_s / 2.0
            cue_level  = float(parts[5].strip())
            group_name = f"{multich_stem}_ch_{ch}_peak_{peak_time:.6f}"
            print(f"      -> group: ch={ch:02d}  start={start_s:.3f}s  dur={duration_s:.3f}s  "
                  f"peak={peak_time:.3f}s  cue={cue_level:.4f}")
            _write_candidate_to_h5(
                temp_hdf5_path, group_name,
                dt['date'], dt['time'], raw_name_wav,
                ch, peak_time, start_s, end_s, duration_s, cue_level,
            )
            total += 1

    print(f"\n  [aggregate_csvs] Total candidates written to temp.h5: {total}")
    _print_h5_summary(temp_hdf5_path, label="after aggregate_csvs")
    return total


def analyze_temp_file(temp_hdf5_path: str, results_hdf5_path: str) -> None:
    print(f"\n  [analyze_temp_file] Starting clustering (tolerance={TOLERANCE_SEC}s)")

    n_before = 0
    if os.path.exists(results_hdf5_path):
        with h5py.File(results_hdf5_path, 'r') as f:
            n_before = len(f.keys())
    print(f"  [analyze_temp_file] results.h5 before: {n_before} existing events")

    cluster_idx = 0
    while True:
        with h5py.File(temp_hdf5_path, 'r') as f:
            group_names = list(f.keys())
        if not group_names:
            break

        ref_name = group_names[0]
        with h5py.File(temp_hdf5_path, 'r') as f:
            ref_peak = float(f[ref_name]['peak_time'][()])

        cluster = []
        with h5py.File(temp_hdf5_path, 'r') as f:
            for name in group_names:
                grp = f[name]
                pt  = float(grp['peak_time'][()])
                cue = float(grp['cue_level'][()]) if 'cue_level' in grp else np.nan
                ch  = int(grp['ch'][()]) if 'ch' in grp else -1
                if abs(pt - ref_peak) <= TOLERANCE_SEC:
                    cluster.append({'name': name, 'peak_time': pt, 'cue_level': cue, 'ch': ch})

        channels_involved = sorted(c['ch'] for c in cluster)
        best = max(cluster, key=lambda c: c['cue_level'] if not np.isnan(c['cue_level']) else -np.inf)

        cluster_idx += 1
        print(f"\n  [cluster #{cluster_idx}]  ref_peak={ref_peak:.3f}s  -> {len(cluster)} member(s)")
        for c in sorted(cluster, key=lambda x: x['ch']):
            marker = " ← BEST" if c['name'] == best['name'] else ""
            print(f"    ch={c['ch']:02d}  peak={c['peak_time']:.3f}s  cue={c['cue_level']:.4f}{marker}")
        print(f"    channels_involved = {channels_involved}")
        print(f"    -> writing to results.h5: {best['name']}")

        with h5py.File(temp_hdf5_path, 'r') as src, \
             h5py.File(results_hdf5_path, 'a') as dst:
            rname = best['name']
            if rname not in dst:
                src.copy(rname, dst, name=rname)
                if 'channels_involved' in dst[rname]:
                    del dst[rname]['channels_involved']
                dst[rname].create_dataset(
                    'channels_involved',
                    data=np.array(channels_involved, dtype=np.int32),
                )
            else:
                print(f"    ⚠️  {rname} already in results.h5, skipping")

        with h5py.File(temp_hdf5_path, 'a') as f:
            for c in cluster:
                if c['name'] in f:
                    del f[c['name']]

        with h5py.File(temp_hdf5_path, 'r') as f:
            remaining = len(f.keys())
        print(f"    remaining temp.h5 after cluster: {remaining}")

    print(f"\n  [analyze_temp_file] temp.h5 exhausted after {cluster_idx} cluster(s)")
    _print_h5_summary(results_hdf5_path, label="results.h5 AFTER analyze_temp_file")


# ---------------------------------------------------------------------------
# PERSISTENT WORKER -- runs in a dedicated subprocess for the whole lifetime
#                      of the program, with the model loaded once.
# ---------------------------------------------------------------------------
def _persistent_worker(worker_id: int, device_id: str,
                        model_path: str, repo_root: str,
                        job_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """
        Persistent subprocess. Loop:
            1. Pull a job from job_queue
            2. If job == _STOP_SIGNAL -> exit
            3. Otherwise run inference and push the result into result_queue
        The model is loaded ONCE at the start of this function.
    """
        # Heavy imports inside the subprocess (spawn: no inherited CUDA state).
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    from fairseq import checkpoint_utils

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import animal2vec_inference as a2v
    from nn.utils import chunk_and_normalize

    print(f"[worker {worker_id}] Starting on {device_id}, loading model...", flush=True)
    models, _ = checkpoint_utils.load_model_ensemble([model_path])
    model = models[0].to(device_id)
    model.eval()
    print(f"[worker {worker_id}] Model ready on {device_id} ✓", flush=True)

    # Notify the main process that this worker is ready.
    result_queue.put({"type": "ready", "worker_id": worker_id})

    # Infinite job-processing loop.
    while True:
        job = job_queue.get()

        # Termination signal.
        if job == _STOP_SIGNAL:
            print(f"[worker {worker_id}] Received STOP, exiting.", flush=True)
            break

        file_path   = Path(job["file_path"])
        csv_out_dir = Path(job["csv_out_dir"])
        inf_args    = job["inf_args"]
        job_id      = job["job_id"]

        print(f"  [worker {worker_id} | {device_id}] job_id={job_id}  file={file_path.name}", flush=True)

        try:
            unique_labels = eval(inf_args["unique_values"])
            n_classes     = len(unique_labels)

            dataset    = a2v.FileFolderDataset(
                str(file_path),
                sample_rate=inf_args["sample_rate"],
                channel_info=inf_args["channel_info"] if inf_args["channel_info"] else None,
            )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            csv_out_dir.mkdir(parents=True, exist_ok=True)
            header = ["Name", "Start", "Duration", "Time Format", "Type", "Description"]

            method_dict = {
                "sigma_s":          inf_args["sigma_s"],
                "metric_threshold": inf_args["metric_threshold"],
                "maxfilt_s":        inf_args.get("maxfilt_s", 0.1),
                "max_duration_s":   inf_args.get("max_duration_s", 0.5),
                "lowP":             inf_args.get("lowP", 0.125),
            }

            fname    = replace_ending(file_path.name, ".csv")
            file_out = csv_out_dir / fname

            if not inf_args["overwrite_previous_preds"] and file_out.exists():
                print(f"    [worker {worker_id}] Skipping (exists): {file_out.name}", flush=True)
                result_queue.put({"type": "done", "job_id": job_id,
                                   "csv_path": str(file_out), "error": None})
                continue

            all_time_intervals = [[] for _ in range(n_classes)]
            all_likelihoods    = [[] for _ in range(n_classes)]

            with torch.inference_mode():
                for samples in dataloader:
                    if not isinstance(samples, dict):
                        continue
                    wav = samples["source"].squeeze()
                    batched_wav = chunk_and_normalize(
                        wav,
                        segment_length=inf_args["segment_length"],
                        sample_rate=inf_args["sample_rate"],
                        normalize=False,
                        max_batch_size=inf_args["batch_size"],
                    )
                    for bi, batch in enumerate(batched_wav):
                        if not torch.is_tensor(batch):
                            batch = torch.stack(batch)
                        elif batch.dim() == 1:
                            batch = batch.view(1, -1)
                        if inf_args["normalize"]:
                            batch = torch.stack(
                                [torch.nn.functional.layer_norm(x, x.shape) for x in batch]
                            )
                        net_output = model(source=batch.to(device_id))
                        if "linear_eval_projection" in net_output:
                            probs = torch.sigmoid(net_output["linear_eval_projection"].clone().detach().cpu()).float()
                        elif "encoder_out" in net_output:
                            probs = torch.sigmoid(net_output["encoder_out"].clone().detach().cpu()).float()
                        else:
                            print(f"    [worker {worker_id}] WARNING: no logits key for batch {bi}", flush=True)
                            continue
                        fused = model.fuse_predict(
                            batch.size(-1), probs,
                            method_dict=method_dict,
                            method=inf_args["method"],
                            multiplier=bi,
                            bs=inf_args["batch_size"],
                        )
                        for sub_batch_ivs, sub_batch_lks in zip(fused[0], fused[2]):
                            for ci in range(n_classes):
                                if ci < len(sub_batch_ivs):
                                    all_time_intervals[ci].extend(sub_batch_ivs[ci])
                                if ci < len(sub_batch_lks):
                                    all_likelihoods[ci].extend(sub_batch_lks[ci])

            rows = []
            for ci, label in enumerate(unique_labels):
                for iv, lk in zip(all_time_intervals[ci], all_likelihoods[ci]):
                    if len(iv) < 2:
                        continue
                    start_s = float(iv[0])
                    dur_s   = float(iv[1] - iv[0])
                    if lk < inf_args["metric_threshold"] or dur_s < inf_args["min_pred_length"]:
                        continue
                    rows.append([
                        label,
                        str(timedelta(seconds=start_s)),
                        str(timedelta(seconds=dur_s)),
                        "decimal", "", f"{lk:.4f}",
                    ])

            df = pd.DataFrame(rows, columns=header)
            df.to_csv(str(file_out), sep="\t", index=False)
            print(f"    [worker {worker_id} | {device_id}] Written: {file_out.name}  ({len(df)} detections)", flush=True)

            result_queue.put({"type": "done", "job_id": job_id,
                               "csv_path": str(file_out), "error": None})

        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [worker {worker_id}] ERROR job_id={job_id}: {e}\n{tb}", flush=True)
            result_queue.put({"type": "done", "job_id": job_id,
                               "csv_path": None, "error": str(e)})


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def detect_devices(requested_device: str) -> List[str]:
    import torch
    if requested_device == "cpu":
        print("[device] CPU mode -- 1 worker")
        return ["cpu"]
    if not torch.cuda.is_available():
        print("[device] CUDA requested but not available -- falling back to CPU")
        return ["cpu"]
    n = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n)]
    print(f"[device] Detected {n} GPU(s):")
    for i, d in enumerate(devices):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory // (1024**3)
        print(f"  {d}  →  {name}  ({mem} GB)")
    return devices


# ---------------------------------------------------------------------------
# Orchestration for a single multichannel file
# ---------------------------------------------------------------------------
def process_multich_file(fp: Path, tmpdir: Path, csv_temp: Path,
                         output_folder: Path,
                         job_queue: mp.Queue, result_queue: mp.Queue,
                         n_workers: int, inf_args_dict: dict) -> None:
    try:
        info = sf.info(str(fp))
    except Exception as e:
        print(f"  ⚠️  Skipping (unreadable): {e}")
        return

    n_ch = info.channels or 1
    print(f"  {n_ch} ch  |  {info.duration:.1f}s  |  {info.samplerate} Hz")
    print(f"  Splitting {n_ch} channels into tmpdir ...")
    mono_files = split_channels_and_write(tmpdir, fp) if n_ch > 1 else [fp]
    print(f"  Split complete: {len(mono_files)} mono files")

    # Send all jobs -- workers pull them as soon as they become free.
    print(f"\n  Sending {len(mono_files)} jobs to the pool ({n_workers} active worker(s)) ...")
    for i, mf in enumerate(mono_files):
        job_queue.put({
            "job_id":      i,
            "file_path":   str(mf),
            "csv_out_dir": str(csv_temp),
            "inf_args":    inf_args_dict,
        })
    print(f"  All jobs sent, waiting for completion ...")

    # Barrier: wait for exactly one result per mono file.
    completed = 0
    failed    = 0
    for _ in range(len(mono_files)):
        res = result_queue.get()
        if res["type"] != "done":
            continue
        completed += 1
        if res["error"]:
            failed += 1
            print(f"  ⚠️  job_id={res['job_id']} FAILED: {res['error']}")
        else:
            csv_name = Path(res["csv_path"]).name if res["csv_path"] else "(no output)"
            print(f"  ✓ job_id={res['job_id']:02d} -> {csv_name}")

    print(f"\n  Inference completed: {completed}/{len(mono_files)} OK, {failed} failed")

    multich_stem = fp.stem
    dt           = parse_filename(multich_stem)
    temp_h5      = str(output_folder / "temp.h5")
    results_h5   = str(output_folder / f"results_{dt['date']}.h5")

    print(f"\n  {'='*60}")
    print(f"  AGGREGATION  ->  {multich_stem}")
    print(f"  {'='*60}")
    n_candidates = aggregate_csvs(csv_temp, multich_stem, temp_h5)

    if n_candidates == 0:
        print("  No candidates in temp.h5, skipping clustering.")
    else:
        print(f"\n  {'='*60}")
        print(f"  CLUSTERING  ->  {os.path.basename(results_h5)}")
        print(f"  {'='*60}")
        analyze_temp_file(temp_h5, results_h5)
        shutil.rmtree(csv_temp)
        csv_temp.mkdir(parents=True, exist_ok=True)
        print(f"  [cleanup] csv_temp recreated empty: {csv_temp}")

    if os.path.exists(temp_h5):
        os.remove(temp_h5)
        print("\n  temp.h5 removed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="animal2vec batch inference + clustering (persistent multi-GPU pool)"
    )
    parser.add_argument("--input-folder",             required=True,  type=Path)
    parser.add_argument("--output-folder",            required=True,  type=Path)
    parser.add_argument("--model-path",               required=True,  type=Path)
    parser.add_argument("--sample-rate",              default=24000,  type=int)
    parser.add_argument("--device",                   default="cpu",  choices=["cuda", "cpu"])
    parser.add_argument("--unique-values",            default="['stop_signal_candidate']", type=str)
    parser.add_argument("--method",                   default="avg",  choices=["avg", "max", "canny"])
    parser.add_argument("--metric-threshold",         default=0.70,   type=float)
    parser.add_argument("--sigma-s",                  default=0.6,    type=float)
    parser.add_argument("--overwrite-previous-preds", default=True,
                        type=lambda v: v.lower() in ("true", "1", "yes"))
    parser.add_argument("--normalize",                default=True,
                        type=lambda v: v.lower() in ("true", "1", "yes"))
    parser.add_argument("--segment-length",           default=10.0,   type=float)
    parser.add_argument("--batch-size",               default=16,     type=int)
    parser.add_argument("--min-pred-length",          default=0.05,   type=float)
    parser.add_argument("--channel-info",             default="",     type=str)
    args = parser.parse_args()

    if not args.input_folder.exists():
        raise SystemExit(f"Input folder not found: {args.input_folder}")
    if not args.model_path.exists():
        raise SystemExit(f"Model checkpoint not found: {args.model_path}")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    csv_temp = args.output_folder / "csv_temp"
    csv_temp.mkdir(parents=True, exist_ok=True)

    # Detect GPUs.
    devices  = detect_devices(args.device)
    n_workers = len(devices)

    inf_args_dict = {
        "unique_values":            args.unique_values,
        "sample_rate":              args.sample_rate,
        "channel_info":             args.channel_info,
        "sigma_s":                  args.sigma_s,
        "metric_threshold":         args.metric_threshold,
        "overwrite_previous_preds": args.overwrite_previous_preds,
        "normalize":                args.normalize,
        "segment_length":           args.segment_length,
        "batch_size":               args.batch_size,
        "min_pred_length":          args.min_pred_length,
        "method":                   args.method,
    }

    # Spawn persistent workers with mp_context="spawn" (required for CUDA).
    ctx          = mp.get_context("spawn")
    job_queue    = ctx.Queue()
    result_queue = ctx.Queue()
    workers      = []

    print(f"\nStarting {n_workers} persistent worker(s) ...")
    for wid, device_id in enumerate(devices):
        p = ctx.Process(
            target=_persistent_worker,
            args=(wid, device_id, str(args.model_path),
                  str(_repo_root), job_queue, result_queue),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Wait until all workers are ready (model loaded).
    ready_count = 0
    print("Waiting for all workers to load the model ...")
    while ready_count < n_workers:
        msg = result_queue.get()
        if msg.get("type") == "ready":
            ready_count += 1
            print(f"  worker {msg['worker_id']} ready  ({ready_count}/{n_workers})")
    print(f"All workers ready. Starting processing.\n")

    all_files = list_wav_files(args.input_folder)
    if not all_files:
        raise SystemExit(f"No WAV/FLAC files found in {args.input_folder}")

    if DEBUG_MAX_FILES is not None:
        files_to_process = all_files[:DEBUG_MAX_FILES]
        print(f"[DEBUG] Processing {len(files_to_process)}/{len(all_files)} file(s).")
    else:
        files_to_process = all_files
        print(f"Files to process: {len(files_to_process)}")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for idx, fp in enumerate(files_to_process, 1):
            print(f"\n[{idx}/{len(files_to_process)}] {fp.name}")
            print("=" * 80)
            try:
                process_multich_file(
                    fp, tmpdir, csv_temp, args.output_folder,
                    job_queue, result_queue, n_workers, inf_args_dict,
                )
            except Exception as e:
                print(f"  ⚠️  ERROR on {fp.name}: {e}")
                traceback.print_exc()

    # Send the stop signal to all workers.
    print("\nStopping workers ...")
    for _ in workers:
        job_queue.put(_STOP_SIGNAL)
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            print(f"  Worker {p.pid} did not exit, forcing kill")
            p.kill()

    print("\n" + "=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()