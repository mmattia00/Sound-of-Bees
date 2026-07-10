#!/usr/bin/env python3
"""Convert a CSV label table into an animal2vec-compatible pickle.

Usage:
    python from_csv_to_pickle.py --csv input_labels.csv --out stop_signal_labels.pkl

The input CSV must contain one row per event with a filename column and
onset/offset columns in seconds. The script normalizes the table, removes
invalid rows, and writes a pickle with the canonical animal2vec fields.
"""
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a CSV label table into an animal2vec-compatible pickle."
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=str,
        help="Input CSV path (e.g. pickle_file_preparation.csv)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output pickle path (e.g. stop_signal_labels.pkl)",
    )
    parser.add_argument(
        "--class-name",
        default="stop_signal_candidate",
        type=str,
        help="Value to write in the Name column.",
    )
    parser.add_argument(
        "--focal",
        action="store_true",
        help="If set, write Focal=True for all rows (default: False).",
    )
    parser.add_argument(
        "--sep",
        default=",",
        type=str,
        help="CSV separator (default: ,)",
    )
    parser.add_argument(
        "--filename-col",
        default="filename",
        type=str,
        help="Column name containing audio file basenames.",
    )
    parser.add_argument(
        "--onset-col",
        default="onset_sec",
        type=str,
        help="Column name containing event onset in seconds.",
    )
    parser.add_argument(
        "--offset-col",
        default="offset_sec",
        type=str,
        help="Column name containing event offset in seconds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=args.sep)

    # Normalize header names and drop accidental trailing empty columns.
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    required = [args.filename_col, args.onset_col, args.offset_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. Available columns: {df.columns.tolist()}"
        )

    # Basic cleaning for robust conversion from user-provided CSV.
    # Keep only rows with a valid filename and a strict onset/offset order.
    df[args.filename_col] = df[args.filename_col].astype(str).str.strip()
    df[args.onset_col] = pd.to_numeric(df[args.onset_col], errors="coerce")
    df[args.offset_col] = pd.to_numeric(df[args.offset_col], errors="coerce")

    before_drop = len(df)
    df = df.dropna(subset=[args.filename_col, args.onset_col, args.offset_col])
    df = df[df[args.offset_col] > df[args.onset_col]].copy()
    after_drop = len(df)

    labels = pd.DataFrame(
        {
            "Name": args.class_name,
            "AudioFile": df[args.filename_col],
            "StartRelative": pd.to_timedelta(df[args.onset_col], unit="s"),
            "EndRelative": pd.to_timedelta(df[args.offset_col], unit="s"),
            "Focal": bool(args.focal),
        }
    )

    # Ensure the destination exists before serializing the final label table.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_pickle(out_path)

    print("Input CSV:", str(csv_path))
    print("Output pickle:", str(out_path))
    print("Rows read:", before_drop)
    print("Rows kept:", after_drop)
    print("Unique audio files:", labels["AudioFile"].nunique())
    print("Preview:")
    print(labels.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

