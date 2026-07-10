#!/usr/bin/env python3
"""
This script transfers daily audio folders to the timon server via rclone.
It keeps track of the last successfully sent folder so repeated runs are incremental.

Usage:
    python3 push_audio_to_timon.py [--verbose] <audio_folder> [rclone_remote]
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STATE_FILENAME   = ".last_sent_state"
RCLONE_ERROR_LOG = "rclone_errors.log"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=level,
        stream=sys.stdout,
    )
    logging.Formatter.converter = time.gmtime


def read_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            logging.warning("State file is corrupted, starting from scratch.")
    return {"last_sent_day": None}


def write_state(state_file: Path, last_sent_day: str) -> None:
    state = {
        "last_sent_day": last_sent_day,
        "last_sent_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    state_file.write_text(json.dumps(state, indent=2))
    logging.debug("State updated: %s", state)


def rclone_copy_day(
    src_dir: Path,
    remote_dest: str,
    err_log: Path,
) -> bool:
    """
    Transfer the entire daily folder with rclone copy --transfers 24.
    Idempotent: files already present on the remote side are skipped.
    Returns True only if rclone exits with rc=0.
    Streams output to the terminal in real time and writes errors to err_log.
    """
    # rclone is configured for high-throughput transfers with bounded retries and timeouts.
    cmd = [
        "rclone", "copy",
        str(src_dir),
        remote_dest,
        "--transfers", "24",
        "--create-empty-src-dirs",
        "--progress",
        "--log-file", str(err_log),
        "--log-level", "ERROR",
        "--timeout", "30s",
        "--contimeout", "15s",
        "--retries", "3",
        "--low-level-retries", "3",
        "--retries-sleep", "5s",
    ]
    logging.debug("rclone: %s", " ".join(cmd))
    logging.info("rclone: invio cartella %s ...", src_dir.name)
    start = time.monotonic()

    try:
        result = subprocess.run(cmd)
        elapsed = time.monotonic() - start

        if result.returncode != 0:
            logging.error("rclone: FAIL %s (rc=%d, %.1fs) -- details in %s",
                          src_dir.name, result.returncode, elapsed, err_log)
            return False

        logging.info("rclone: OK %s (%.1fs)", src_dir.name, elapsed)
        return True

    except KeyboardInterrupt:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with err_log.open("a") as f:
            f.write(f"---- {ts} ERROR day={src_dir.name} rc=-2 ----\n")
            f.write("Interrupted by user (SIGINT)\n\n")
        logging.error("rclone: interrupted on %s -- details in %s", src_dir.name, err_log)
        raise

    except FileNotFoundError:
        logging.error("rclone not found in PATH")
        sys.exit(1)


def get_today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def run_scan(
    audio_folder: Path,
    rclone_remote: str,
    state_file: Path,
    err_log: Path,
) -> None:
    state = read_state(state_file)
    last_sent_day = state.get("last_sent_day")
    today = get_today_utc()

    logging.info("=== SCAN START === last_sent_day=%s today=%s",
                 last_sent_day or "[none]", today)

    # Process day folders in chronological order so the state file can stop on the last successful day.
    day_dirs = sorted(
        d for d in audio_folder.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not day_dirs:
        logging.info("No folders found.")
        return

    for day_dir in day_dirs:
        day = day_dir.name

        # Skip folders that were already sent in a previous run.
        if last_sent_day and day <= last_sent_day:
            logging.debug("Skip %s (already sent)", day)
            continue

        # Skip empty folders that do not contain WAV files.
        wav_files = list(day_dir.glob("*.wav"))
        if not wav_files:
            logging.info("Skip %s (no .wav files)", day)
            continue

        logging.info("Invio %s (%d file) → %s/%s",
                     day, len(wav_files), rclone_remote, day)

        ok = rclone_copy_day(day_dir, f"{rclone_remote}/{day}", err_log)

        if ok:
            # Persist progress only after a successful remote copy.
            write_state(state_file, day)
            last_sent_day = day
        else:
            logging.error(
                "Transfer failed for %s -- it will be retried on the next run. "
                "Details in %s", day, err_log
            )
            return  # stop here: do not continue with later folders

    logging.info("=== SCAN END ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push daily audio to timon via rclone"
    )
    parser.add_argument("audio_folder", type=Path)
    parser.add_argument("rclone_remote", nargs="?", default="timon:/audio")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    audio_folder: Path = args.audio_folder.resolve()
    if not audio_folder.is_dir():
        logging.error("Directory not found: %s", audio_folder)
        sys.exit(1)

    state_file = audio_folder / STATE_FILENAME
    err_log    = audio_folder / RCLONE_ERROR_LOG

    logging.info("audio_folder : %s", audio_folder)
    logging.info("rclone_remote: %s", args.rclone_remote)

    try:
        run_scan(audio_folder, args.rclone_remote, state_file, err_log)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()