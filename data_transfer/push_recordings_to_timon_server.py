#!/usr/bin/env python3
"""
This script transfers daily recording folders (audio or video) to timon via rclone.
It keeps track of the last successfully sent folder so repeated runs are incremental.
It was made in order to use the same script for both audio and video recordings,
at the end it has been used this rclone-based script to transfer only audio and 
so not used in the final acquisition pipeline.

Usage:
    python3 push_recordings_to_timon.py [--verbose] [--media audio|video] [--cam cam-0] <base_folder> [rclone_remote]
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RCLONE_REMOTE = "timon:/A26-03-0300/recordings2026"
STATE_FILENAME        = ".last_sent_state"
RCLONE_ERROR_LOG      = "rclone_errors.log"
TRANSFER_LOG          = "transfer.log"


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


def folder_size_human(path: Path) -> str:
    """Compute the total folder size and return it in human-readable units."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} PB"


def rclone_copy_day(
    src_dir: Path,
    remote_dest: str,
    err_log: Path,
    transfer_log: Path,
) -> bool:
    # rclone is configured for high-throughput transfers and size-only comparisons.
    cmd = [
        "rclone", "copy",
        str(src_dir),
        remote_dest,
        "--transfers", "24",
        "--create-empty-src-dirs",
        "--size-only",          # ← confronta solo dimensione, ignora timestamp
        "--log-file", str(err_log),
        "--log-level", "INFO",
        "--stats-one-line",
        "--timeout", "30s",
        "--contimeout", "15s",
        "--retries", "3",
        "--low-level-retries", "3",
        "--retries-sleep", "5s",
    ]

    size = folder_size_human(src_dir)
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logging.debug("rclone: %s", " ".join(cmd))
    logging.info("rclone: invio cartella %s (%s) ...", src_dir.name, size)
    start = time.monotonic()

    try:
        result = subprocess.run(cmd)
        elapsed = time.monotonic() - start
        completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if result.returncode != 0:
            logging.error("rclone: FAIL %s (rc=%d, %.1fs)", src_dir.name, result.returncode, elapsed)
            with transfer_log.open("a") as f:
                f.write(
                    f"transfer of folder {src_dir.name} ({size}) not completed,"
                    f" check error log ({err_log.name})\n"
                )
            return False

        with transfer_log.open("a") as f:
            f.write(
                f"transfer of folder {src_dir.name} ({size})"
                f" started at {started_at}"
                f" completed at {completed_at}"
                f" tot time {elapsed:.1f}s\n"
            )
        logging.info("rclone: OK %s (%.1fs)", src_dir.name, elapsed)
        return True

    except KeyboardInterrupt:
        with transfer_log.open("a") as f:
            f.write(
                f"transfer of folder {src_dir.name} ({size}) not completed,"
                f" check error log ({err_log.name})\n"
            )
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with err_log.open("a") as f:
            f.write(f"---- {ts} ERROR day={src_dir.name} rc=-2 ----\n")
            f.write("Interrupted by user (SIGINT)\n\n")
        logging.error("rclone: interrupted on %s", src_dir.name)
        raise

    except FileNotFoundError:
        logging.error("rclone not found in PATH")
        sys.exit(1)


def get_today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def run_scan(
    scan_root: Path,
    rclone_remote: str,
    state_file: Path,
    err_log: Path,
    transfer_log: Path,
) -> None:
    state = read_state(state_file)
    last_sent_day = state.get("last_sent_day")
    today = get_today_utc()

    logging.info("=== SCAN START === scan_root=%s last_sent_day=%s today=%s",
                 scan_root, last_sent_day or "[none]", today)

    # Walk folders in chronological order so the state file can stop at the last successful folder.
    day_dirs = sorted(
        d for d in scan_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not day_dirs:
        logging.info("No folders found in %s.", scan_root)
        return

    for day_dir in day_dirs:
        day = day_dir.name

        # Skip folders already sent in a previous run.
        # if last_sent_day and day <= last_sent_day:  # normal behavior
        if last_sent_day and day <= last_sent_day and day != today:  # benchmark/test mode

            logging.debug("Skip %s (already sent)", day)
            continue


        # Skip empty folders that do not contain any file.
        any_file = next(day_dir.rglob("*"), None)
        if any_file is None:
            logging.info("Skip %s (empty folder)", day)
            continue

        logging.info("Invio %s → %s/%s", day, rclone_remote, day)

        ok = rclone_copy_day(day_dir, f"{rclone_remote}/{day}", err_log, transfer_log)

        if ok:
            # Persist progress only after a successful transfer.
            write_state(state_file, day)
            last_sent_day = day
        else:
            logging.error(
                "Transfer failed for %s -- it will be retried on the next run. "
                "Details in %s", day, err_log
            )
            return

    logging.info("=== SCAN END ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push daily recordings to timon via rclone"
    )
    parser.add_argument("base_folder", type=Path,
                        help="Base folder (e.g. /tank/recordings2026)")
    parser.add_argument("rclone_remote", nargs="?", default=DEFAULT_RCLONE_REMOTE)
    parser.add_argument("--media", choices=("audio", "video"), default="audio")
    parser.add_argument("--cam", help="Camera name, required with --media=video (e.g. cam-0)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    base_folder: Path = args.base_folder.expanduser().resolve()
    if not base_folder.is_dir():
        logging.error("Directory not found: %s", base_folder)
        sys.exit(1)

    rclone_remote = args.rclone_remote.rstrip("/")

    if args.media == "audio":
        scan_root = base_folder / "audio"
        rclone_remote = rclone_remote + "/audio"
    else:
        if not args.cam:
            logging.error("--cam is required with --media=video (e.g. --cam cam-0)")
            sys.exit(1)
        scan_root = base_folder / "video" / "out" / args.cam
        rclone_remote = rclone_remote + "/video/" + args.cam

    if not scan_root.is_dir():
        logging.error("Source folder not found: %s", scan_root)
        sys.exit(1)

    state_file   = scan_root / STATE_FILENAME
    err_log      = scan_root / RCLONE_ERROR_LOG
    transfer_log = scan_root / TRANSFER_LOG

    logging.info("scan_root    : %s", scan_root)
    logging.info("rclone_remote: %s", rclone_remote)
    logging.info("state_file   : %s", state_file)
    logging.info("transfer_log : %s", transfer_log)
    logging.info("rclone_errors: %s", err_log)

    try:
        run_scan(scan_root, rclone_remote, state_file, err_log, transfer_log)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()