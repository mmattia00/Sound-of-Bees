#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SYNC_WAV = "/tank/mls_1hour_mono.wav"
#SYNC_WAV = "/tank/sync_pulse_60sec_mono.wav"
SYNC_CLIENT_NAME = "jack-play"  # nome default del client jack-play



def run_jack_capture_once(duration, output_dir, channels):
    """
    Lancia jack_capture per 'duration' secondi, salvando un file multicanale
    (system:capture_* + slave_in:capture_*).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H_%M_%S.%fZ")
    filename_pattern = str(output_dir / f"{ts}.wav")

    cmd = [
        "jack_capture",
        "--channels", str(channels),
        "--port", "system:capture_*",
        "--port", "beeSlave1:capture_*",
        "--port", "beeSlave2:capture_*",
        "-f", "wav",
        "-b", "24",
        "-d", str(duration),
        "-fn", filename_pattern,
    ]
    print(f"[jack_capture] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[jack_capture] Error (exit code {e.returncode})", file=sys.stderr)
        return False

    return True


def start_jack_play_and_route():
    cmd = ["jack-play", SYNC_WAV]
    print(f"[jack-play] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)

    print("[jack-play] Waiting for JACK client ports to appear...")
    ports_out1 = []
    for _ in range(50):
        try:
            out = subprocess.check_output(["jack_lsp"], text=True)
        except subprocess.CalledProcessError:
            time.sleep(0.1)
            continue

        ports_out1 = [
            line.strip()
            for line in out.splitlines()
            if line.strip().startswith("jack-play-") and line.strip().endswith(":out_1")
        ]
        if ports_out1:
            break
        time.sleep(0.1)

    if not ports_out1:
        print("[jack-play] Warning: no jack-play-* out_1 ports found, routing may fail.", file=sys.stderr)
    else:
        print("[jack-play] Found ports:", ", ".join(ports_out1))

    def safe_connect(src, dst):
        try:
            print(f"[jack-play] Connecting {src} -> {dst}")
            subprocess.run(["jack_connect", src, dst], check=True)
        except subprocess.CalledProcessError:
            print(f"[jack-play] Could not connect {src} -> {dst}", file=sys.stderr)

    # collega tutti i jack-play-*:out_1 solo al master
    for src in ports_out1:
        safe_connect(src, "system:playback_10")

    return proc


def stop_jack_play(proc):
    """
    Ferma jack-play se il processo è ancora vivo.
    """
    if proc is None:
        return
    if proc.poll() is None:
        print("[jack-play] Stopping jack-play process...")
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[jack-play] Forcing kill...")
                proc.kill()
        except Exception as e:
            print(f"[jack-play] Error while stopping jack-play: {e}", file=sys.stderr)
    else:
        print("[jack-play] Process already finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Chunked recording using jack_capture (20ch: system + slave_in) + sync via jack-play"
    )
    parser.add_argument(
        "-d", "--duration", type=float, required=True,
        help="Duration of each chunk in seconds"
    )
    parser.add_argument(
        "-t", "--total-duration", type=float,
        help="Total duration in seconds (default: infinite until Ctrl+C)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="jack_chunks",
        help="Output directory for WAV chunks"
    )
    parser.add_argument(
        "-c", "--channels", type=int, default=30,
        help="Number of channels to record (default: 30 = 10 system + 10 beeSlave1 + 10 beeSlave2)"
    )

    args = parser.parse_args()

    if args.total_duration is None:
        total_duration = float("inf")
    else:
        total_duration = args.total_duration

    print("IMPORTANT: JACK and alsa_in (slave_in) must already be running.")
    print("Using ports: system:capture_* + beeSlave1:capture_* + beeSlave2:capture_*")
    print(f"Chunk duration: {args.duration} s")
    print(f"Total duration: {'infinite' if total_duration == float('inf') else total_duration} s")
    print(f"Output dir: {args.output_dir}")
    print(f"Sync file (mono): {SYNC_WAV}")
    print("#" * 80)

    # Avvia jack-play sul segnale di sync e fai routing
    jack_play_proc = start_jack_play_and_route()

    start_wall = time.time()

    try:
        while True:
            if total_duration != float("inf"):
                elapsed = time.time() - start_wall
                if elapsed >= total_duration:
                    print("[main] Total duration reached, stopping.")
                    break

            ok = run_jack_capture_once(args.duration, args.output_dir, args.channels)
            if not ok:
                print("[main] jack_capture failed, stopping.")
                break

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user, exiting.")
    finally:
        # ferma jack-play
        stop_jack_play(jack_play_proc)


if __name__ == "__main__":
    main()