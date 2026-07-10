#!/usr/bin/env python3
"""
This script synchronizes audio and video around audio events stored in an HDF5 database.
For each event it builds the matching audio clip, extracts the corresponding video clips,
optionally muxes them together, and crops the result around the calibrated microphone.

Usage:
    python sync_audio_video.py --db PATH --audio-folder PATH --video-folder PATH [OPTIONS]
"""

import argparse
import bisect
import logging
import re
import traceback
from datetime import datetime, timezone
import os
import subprocess
import h5py
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classes.microphone_surface_calibrator import MicrophoneSurfaceCalibrator


TMP_DIR       = "tmp"
AUDIO_TS_FMT  = "%Y-%m-%dT%H:%M:%S.%fZ"
VIDEO_TS_FMT  = "%Y%m%dT%H%M%S.%fZ"


# -----------------------
# Logging
# -----------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("sync_events")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s  %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

log: logging.Logger = None


# -----------------------
# HDF5 helpers
# -----------------------

def _read_scalar(grp, key, cast=None, default=None):
    if key in grp:
        value = grp[key][()]
    elif key in grp.attrs:
        value = grp.attrs.get(key, default)
    else:
        return default
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    if cast is not None and value is not None:
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default
    return value


# -----------------------
# Filename parsers
# -----------------------

def parse_fileaudio_name(filename: str) -> float:
    stem = os.path.basename(filename).rsplit(".", 1)[0]
    ts   = stem.replace("_", ":", 2)
    dt   = datetime.strptime(ts, AUDIO_TS_FMT).replace(tzinfo=timezone.utc)
    return dt.timestamp()

def parse_filevideo_name(filename: str) -> Tuple[float, float]:
    base = os.path.basename(filename)
    m    = re.search(r"_(\d{8}T\d{6}\.\d+(?:\.\d+)?Z)--(\d{8}T\d{6}\.\d+(?:\.\d+)?Z)", base)
    if not m:
        raise ValueError(f"Cannot parse video timestamps from: {filename}")
    def clean(ts):
        return re.sub(r"(\.\d{1,6})\.\d+Z$", r"\1Z", ts)
    start_dt = datetime.strptime(clean(m.group(1)), VIDEO_TS_FMT).replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(clean(m.group(2)), VIDEO_TS_FMT).replace(tzinfo=timezone.utc)
    return start_dt.timestamp(), end_dt.timestamp()


# -----------------------
# Window
# -----------------------

def compute_window(absolute_peak_time: float, window_size: float) -> Tuple[float, float]:
    half = window_size / 2
    return absolute_peak_time - half, absolute_peak_time + half


# -----------------------
# Index builders
# -----------------------

def build_audio_index(audio_folder: str) -> list:
    index = []
    for f in os.listdir(audio_folder):
        if not f.endswith(".wav"):
            continue
        try:
            index.append((parse_fileaudio_name(f), f))
        except ValueError:
            pass
    index.sort(key=lambda x: x[0])
    return index

def build_video_index(video_folder: str, cam: int) -> list:
    index = []
    for f in os.listdir(video_folder):
        if not (f.endswith(".mp4") and f.startswith(f"cam-{cam}_")):
            continue
        try:
            s, e = parse_filevideo_name(f)
            index.append((s, e, f))
        except ValueError:
            pass
    index.sort(key=lambda x: x[0])
    return index


# -----------------------
# Calibration bootstrap
# -----------------------

def ensure_calibration(calibration_path: str, video_folder: str,
                        num_mics_per_cam: int = 16) -> MicrophoneSurfaceCalibrator:
    cal = MicrophoneSurfaceCalibrator()
    if Path(calibration_path).exists():
        log.info(f"Loading calibration from {calibration_path}")
        cal.load_calibration(calibration_path)
    else:
        log.info(f"Calibration not found -- starting interactive calibration")
        cal.calibrate(video_folder=video_folder,
                      num_mics_per_cam=num_mics_per_cam,
                      output_path=calibration_path)
        log.info(f"Calibration saved to {calibration_path}")
    return cal


# -----------------------
# Audio loader
# -----------------------

def merge_audiofiles(audio_a: np.ndarray, audio_b: np.ndarray) -> np.ndarray:
    return np.concatenate([audio_a, audio_b], axis=0)

def load_audioclip(raw_name: str, window_start: float, window_end: float,
                   audio_index: list, audio_start_time: float,
                   base_folder: str, ch: int) -> Optional[str]:
    try:
        audiofile, sr  = sf.read(os.path.join(base_folder, raw_name), always_2d=True)
        audio_end_time = audio_start_time + len(audiofile) / sr

        log.debug(f"Audio bounds: [{datetime.utcfromtimestamp(audio_start_time).isoformat()}Z -> "
          f"{datetime.utcfromtimestamp(audio_end_time).isoformat()}Z]  ({raw_name})")

        current_idx = next(
            (i for i, (ts, name) in enumerate(audio_index) if name == raw_name), None
        )

        effective_audio = audiofile
        effective_start = audio_start_time

        if window_start < audio_start_time:
            if current_idx is None or current_idx == 0:
                log.warning(f"{raw_name}: window extends to the left, no previous file -- clip truncated")
            else:
                prev_ts, prev_name = audio_index[current_idx - 1]
                prev_audio, prev_sr = sf.read(os.path.join(base_folder, prev_name), always_2d=True)
                if prev_sr != sr:
                    log.warning(f"sample rate mismatch ({prev_sr} vs {sr}) -- skip left merge")
                else:
                    log.info(f"Merging with previous file: {prev_name}")
                    effective_audio = merge_audiofiles(prev_audio, audiofile)
                    effective_start = prev_ts

        if window_end > audio_end_time:
            if current_idx is None or current_idx >= len(audio_index) - 1:
                log.warning(f"{raw_name}: window extends to the right, no next file -- clip truncated")
            else:
                next_ts, next_name = audio_index[current_idx + 1]
                next_audio, next_sr = sf.read(os.path.join(base_folder, next_name), always_2d=True)
                if next_sr != sr:
                    log.warning(f"sample rate mismatch ({next_sr} vs {sr}) -- skip right merge")
                else:
                    log.info(f"Merging with next file: {next_name}")
                    effective_audio = merge_audiofiles(effective_audio, next_audio)

        start_sample = max(0, int((window_start - effective_start) * sr))
        end_sample   = min(len(effective_audio), int((window_end - effective_start) * sr))

        if end_sample <= start_sample:
            raise ValueError(f"window fuori range: start={start_sample} end={end_sample}")

        clip = effective_audio[start_sample:end_sample, ch]
        output_path = os.path.join(TMP_DIR, "audio_clip.wav")
        sf.write(output_path, clip, sr, subtype="PCM_16")
        log.info(f"Audio clip: {len(clip)/sr:.3f} s  ch={ch+1}  -> {output_path}")
        return output_path

    except Exception as e:
        log.error(f"load_audioclip FAILED for {raw_name}: {e}\n{traceback.format_exc()}")
        return None


# -----------------------
# Video loader
# -----------------------

def merge_videofiles(vid_path_a: str, vid_path_b: str, output_path: str) -> None:
    concat_list = os.path.join(TMP_DIR, "concat_list.txt")
    with open(concat_list, "w") as f:
        f.write(f"file '{os.path.abspath(vid_path_a)}'\n")
        f.write(f"file '{os.path.abspath(vid_path_b)}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", concat_list, "-c", "copy", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr[-2000:]}")

def _extract_videoclip(vid_path: str, seek: float, duration: float,
                       output_path: str, cam: int) -> str:
    cmd = ["ffmpeg", "-y",
           "-ss", f"{seek:.6f}", "-i", vid_path,
           "-t", f"{duration:.6f}",
           "-c:v", "libx264", "-crf", "18", "-preset", "fast",
           "-an", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed:\n{result.stderr[-2000:]}")
    log.info(f"Video cam{cam} clip: {duration:.3f} s  -> {output_path}")
    return output_path

def load_videoclip(cam: int, window_start: float, window_end: float,
                   video_index: list, video_folder: str) -> Optional[str]:
    try:
        starts = [s for s, e, n in video_index]
        idx    = bisect.bisect_right(starts, window_start) - 1

        if idx < 0 or idx >= len(video_index):
            raise ValueError(f"no video file covers window_start={window_start}")

        vid_start, vid_end, vid_name = video_index[idx]
        vid_path = os.path.join(video_folder, vid_name)

        log.debug(f"cam{cam} bounds: [{datetime.utcfromtimestamp(vid_start).isoformat()}Z -> "
          f"{datetime.utcfromtimestamp(vid_end).isoformat()}Z]  ({vid_name})")

        effective_path  = vid_path
        effective_start = vid_start

        if window_start < vid_start:
            if idx == 0:
                log.warning(f"cam{cam}: window extends to the left, no previous file -- clip truncated")
            else:
                prev_start, _, prev_name = video_index[idx - 1]
                merged = os.path.join(TMP_DIR, f"cam{cam}_merged_left.mp4")
                log.info(f"cam{cam}: merging with previous file: {prev_name}")
                merge_videofiles(os.path.join(video_folder, prev_name), vid_path, merged)
                effective_path  = merged
                effective_start = prev_start

        if window_end > vid_end:
            if idx >= len(video_index) - 1:
                log.warning(f"cam{cam}: window extends to the right, no next file -- clip truncated")
            else:
                _, _, next_name = video_index[idx + 1]
                merged = os.path.join(TMP_DIR, f"cam{cam}_merged_right.mp4")
                log.info(f"cam{cam}: merging with next file: {next_name}")
                merge_videofiles(effective_path, os.path.join(video_folder, next_name), merged)
                effective_path = merged

        output_path = os.path.join(TMP_DIR, f"cam{cam}_clip.mp4")
        return _extract_videoclip(effective_path,
                                   seek=window_start - effective_start,
                                   duration=window_end - window_start,
                                   output_path=output_path, cam=cam)
    except Exception as e:
        log.error(f"load_videoclip cam{cam} FAILED: {e}\n{traceback.format_exc()}")
        return None


# -----------------------
# Mux audio + video
# -----------------------

def mux_audio_video(video_path: str, audio_path: str, output_path: str) -> None:
    cmd = ["ffmpeg", "-y",
           "-i", video_path, "-i", audio_path,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-map", "0:v:0", "-map", "1:a:0",
           "-shortest", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr[-2000:]}")
    log.info(f"Muxed -> {output_path}")


# -----------------------
# Zoom crop
# -----------------------

def zoom_on_mic(synced_clip_path: str, ch_1based: int,
                calibrator: MicrophoneSurfaceCalibrator,
                output_path: str, zoom_size_px: int = 300) -> None:
    cx, cy = calibrator.get_pixel_coords(ch_1based)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         synced_clip_path],
        capture_output=True, text=True
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{probe.stderr[-1000:]}")
    w_orig, h_orig = map(int, probe.stdout.strip().split(","))

    half   = zoom_size_px // 2
    x1     = max(0, cx - half);  y1 = max(0, cy - half)
    x2     = min(w_orig, cx + half); y2 = min(h_orig, cy + half)
    crop_w = x2 - x1;  crop_h = y2 - y1

    vf = f"crop={crop_w}:{crop_h}:{x1}:{y1},scale={w_orig}:{h_orig}"
    cmd = ["ffmpeg", "-y", "-i", synced_clip_path,
           "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "fast",
           "-c:a", "copy", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg zoom failed:\n{result.stderr[-2000:]}")
    log.info(f"Zoomed -> {output_path}  mic={ch_1based} ({cx},{cy}) crop={crop_w}x{crop_h}")


# -----------------------
# CLI
# -----------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract audio+video clips for each event in an HDF5 database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db",               required=True,  metavar="PATH")
    p.add_argument("--audio-folder",     required=True,  metavar="PATH")
    p.add_argument("--video-folder",     required=True,  metavar="PATH")
    p.add_argument("--output-dir",       default="output", metavar="PATH")
    p.add_argument("--calibration",      default="calibration.json", metavar="PATH")
    p.add_argument("--num-mics-per-cam", type=int,   default=16)
    p.add_argument("--window-size",      type=float, default=10.0)
    p.add_argument("--zoom-size",        type=int,   default=1200)
    p.add_argument("--fps",              type=float, default=15.0)
    p.add_argument("--log-file",         default="sync_events.log")
    return p


# -----------------------
# Main
# -----------------------

def main() -> None:
    global log
    args = build_parser().parse_args()
    log  = setup_logger(args.log_file)

    os.makedirs(TMP_DIR,         exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    calibrator  = ensure_calibration(args.calibration, args.video_folder, args.num_mics_per_cam)
    audio_index = build_audio_index(args.audio_folder)
    cam0_index  = build_video_index(args.video_folder, cam=0)
    cam1_index  = build_video_index(args.video_folder, cam=1)
    log.info(f"Audio: {len(audio_index)} files  |  Cam0: {len(cam0_index)}  |  Cam1: {len(cam1_index)}")

    with h5py.File(args.db, "r") as f:
        event_ids = list(f.keys())
        log.info(f"Events: {len(event_ids)}")

        for event_id in event_ids:
            grp = f[event_id]
            print(f"\n{'='*50}\n{event_id}")

            raw_name          = grp.attrs.get("raw_name", "")
            ch                = _read_scalar(grp, "ch",        cast=int,   default=0)   # 0-based
            peak_time         = _read_scalar(grp, "peak_time", cast=float, default=None)
            channels_involved = np.array(grp["channels_involved"]) if "channels_involved" in grp else np.array([])

            ch_1based = ch + 1  # used by the calibrator and logs (1-based)
            log.info(f"raw_name={raw_name}  ch={ch_1based}  peak_time={peak_time}  "
                     f"channels_involved={( channels_involved + 1).tolist()}")

            audio_start_time         = parse_fileaudio_name(raw_name)
            absolute_peak            = audio_start_time + peak_time
            window_start, window_end = compute_window(absolute_peak, args.window_size)
            log.info(f"window: [{datetime.utcfromtimestamp(window_start).isoformat()}Z -> "
                     f"{datetime.utcfromtimestamp(window_end).isoformat()}Z]")

            # Step 1: extract the raw audio and video clips around the event window.
            audio_clip = load_audioclip(raw_name, window_start, window_end,
                                        audio_index, audio_start_time,
                                        args.audio_folder, ch)          # 0-based
            cam0_clip  = load_videoclip(0, window_start, window_end, cam0_index, args.video_folder)
            cam1_clip  = load_videoclip(1, window_start, window_end, cam1_index, args.video_folder)

            # Step 2: mux the audio clip into each available video clip.
            synced_cam0 = synced_cam1 = None

            for cam_id, vid_clip in ((0, cam0_clip), (1, cam1_clip)):
                if audio_clip and vid_clip:
                    try:
                        synced = os.path.join(args.output_dir, f"{event_id}_cam{cam_id}_sync.mp4")
                        mux_audio_video(vid_clip, audio_clip, synced)
                        if cam_id == 0:
                            synced_cam0 = synced
                        else:
                            synced_cam1 = synced
                    except Exception as e:
                        log.error(f"mux cam{cam_id} FAILED: {e}\n{traceback.format_exc()}")
                else:
                    log.warning(f"Skipping mux cam{cam_id}: "
                                f"audio={'OK' if audio_clip else 'NONE'}  "
                                f"vid={'OK' if vid_clip else 'NONE'}")

            # Step 3: crop the synchronized clip around the calibrated microphone location.
            try:
                cam_for_ch = calibrator.get_cam_for_mic(ch_1based)  # 1-based
            except KeyError:
                log.warning(f"ch={ch_1based} not in calibration -- skipping zoom")
                cam_for_ch = None

            if cam_for_ch is not None:
                synced_for_zoom = synced_cam0 if cam_for_ch == 0 else synced_cam1
                if synced_for_zoom:
                    try:
                        zoom_on_mic(synced_for_zoom, ch_1based, calibrator,
                                    os.path.join(args.output_dir, f"{event_id}_zoomed.mp4"),
                                    zoom_size_px=args.zoom_size)
                    except Exception as e:
                        log.error(f"zoom FAILED: {e}\n{traceback.format_exc()}")
                else:
                    log.warning(f"Skipping zoom: synced clip for cam{cam_for_ch} is NONE")

    log.info("Done.")


if __name__ == "__main__":
    main()