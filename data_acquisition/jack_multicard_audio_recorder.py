#!/usr/bin/env python3
"""
Chunked multichannel audio recorder built on jack_capture.
It records only the selected JACK ports, writes each chunk into a daily folder,
and stops cleanly on SIGINT or SIGTERM while removing partial files.

Usage:
    python3 jack_multicard_audio_recorder.py -d <chunk_seconds> [-t <total_seconds>] [-o <output_dir>] [--channels N] [--ports PORT [PORT ...]]
"""

import argparse
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# Ports of interest for each board (only those with a connected microphone)
DEFAULT_PORTS = [
    "system:capture_1",
    "system:capture_3",
    "system:capture_5",
    "system:capture_7",
    "beeSlave1:capture_1",
    "beeSlave1:capture_3",
    "beeSlave1:capture_5",
    "beeSlave1:capture_7",
    "beeSlave2:capture_1",
    "beeSlave2:capture_3",
    "beeSlave2:capture_5",
    "beeSlave2:capture_7",
    "beeSlave3:capture_1",
    "beeSlave3:capture_3",
    "beeSlave3:capture_5",
    "beeSlave3:capture_7",
    "beeSlave4:capture_7",
    "beeSlave4:capture_5",
    "beeSlave4:capture_3",
    "beeSlave4:capture_1",
    "beeSlave5:capture_7",
    "beeSlave5:capture_5",
    "beeSlave5:capture_3",
    "beeSlave5:capture_1",
    "beeSlave6:capture_7",
    "beeSlave6:capture_5",
    "beeSlave6:capture_3",
    "beeSlave6:capture_1",
    "beeSlave7:capture_7",
    "beeSlave7:capture_5",
    "beeSlave7:capture_3",
    "beeSlave7:capture_1",
]
DEFAULT_CHANNELS = len(DEFAULT_PORTS)


# ── Signal handling ──────────────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Core ─────────────────────────────────────────────────────────────────────

def run_jack_capture_once(duration, output_dir, channels, ports):
    global _shutdown_requested

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H_%M_%S.%fZ")
    daily_dir = output_dir / now.strftime("%Y-%m-%d")
    daily_dir.mkdir(parents=True, exist_ok=True)

    filename_pattern = str(daily_dir / f"{ts}.wav")

    cmd = ["jack_capture", "--channels", str(channels)]
    for port in ports:
        cmd += ["--port", port]
    cmd += ["-f", "wav", "-b", "24", "-d", str(duration), "-fn", filename_pattern]

    print(f"[jack_capture] Running: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd)

    # Poll until the capture finishes or a stop signal is received.
    while proc.poll() is None:
        if _shutdown_requested:
            print("[jack_capture] Stop signal received, terminating the process...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            break
        time.sleep(0.5)

    # Remove partial files if the capture was interrupted or failed.
    if _shutdown_requested or proc.returncode != 0:
        partial_files = list(daily_dir.glob(f"{ts}*.wav"))
        for f in partial_files:
            try:
                f.unlink()
                print(f"[jack_capture] Removed partial file: {f}")
            except FileNotFoundError:
                pass

        if _shutdown_requested:
            print("[jack_capture] Incomplete chunk removed, exiting cleanly.")
            sys.exit(0)
        else:
            print(f"[jack_capture] Error (exit code {proc.returncode})", file=sys.stderr)
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Chunked recording using jack_capture from 4 devices (selected channels only)"
    )
    parser.add_argument("-d", "--duration", type=float, required=True,
                        help="Duration of each chunk in seconds")
    parser.add_argument("-t", "--total-duration", type=float,
                        help="Total duration in seconds (default: infinite until Ctrl+C)")
    parser.add_argument("-o", "--output-dir", type=str, default="jack_chunks",
                        help="Output directory for WAV chunks")
    parser.add_argument("-c", "--channels", type=int, default=None,
                        help=f"Total channels to record. Default: {DEFAULT_CHANNELS} (= len(DEFAULT_PORTS))")
    parser.add_argument("--ports", type=str, nargs="+", default=None,
                        help="List of JACK ports to record. Default: DEFAULT_PORTS defined in script")

    args = parser.parse_args()

    ports    = args.ports    if args.ports    is not None else DEFAULT_PORTS
    channels = args.channels if args.channels is not None else len(ports)

    total_duration = float("inf") if args.total_duration is None else args.total_duration

    print("IMPORTANT: JACK and zita-a2j for all slaves must already be running.")
    print(f"Ports ({len(ports)}):")
    for p in ports:
        print(f"  {p}")
    print(f"Channels:       {channels}")
    print(f"Chunk duration: {args.duration} s")
    print(f"Total duration: {'infinite' if total_duration == float('inf') else total_duration} s")
    print(f"Output dir:     {args.output_dir}")
    print("#" * 80)

    start_wall = time.time()

    while True:
        if _shutdown_requested:
            print("[main] Stop signal received before starting the next chunk, exiting.")
            sys.exit(0)

        if total_duration != float("inf"):
            if time.time() - start_wall >= total_duration:
                print("[main] Total duration reached, stopping.")
                break

        ok = run_jack_capture_once(args.duration, args.output_dir, channels, ports)
        if not ok:
            print("[main] jack_capture failed, stopping.")
            break


if __name__ == "__main__":
    main()