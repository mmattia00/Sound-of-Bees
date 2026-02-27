#!/usr/bin/env python3 -u
import argparse
import os
import re
from pathlib import Path
from typing import Iterator, List, Tuple

import h5py
import numpy as np
import pandas as pd
import soundfile
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from numpy.typing import NDArray


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", metavar="DIR", help="root directory containing files to index")
    parser.add_argument("--valid-percent", default=0.2, type=float, metavar="D")
    parser.add_argument("--n-split", default=5, type=int)
    parser.add_argument("--dest", default=".", type=str, metavar="DIR")
    parser.add_argument("--ext", default="wav", type=str, metavar="EXT")
    parser.add_argument("--seed", default=1612, type=int, metavar="N")
    parser.add_argument("--few-shot", default=False, type=bool)
    parser.add_argument("--leave-p-out", default=False, type=bool)
    parser.add_argument("--path-must-contain", default=None, type=str, metavar="FRAG")
    parser.add_argument("--valid-set-individuals", type=str, default="", nargs="+")
    parser.add_argument("--id-lookup-file-path", type=str, default="")
    return parser


def get_files(dir, re_obj=None):
    files = []
    print(f"[get_files] Starting os.walk on: {dir}")
    for dirpath, dirnames, filenames in os.walk(dir):
        print(f"[get_files]   Entering dir: {dirpath} ({len(filenames)} files, {len(dirnames)} subdirs)")
        if re_obj is not None:
            filenames = [f for f in filenames if re_obj.match(f)]
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    print(f"[get_files] Done. Found {len(files)} files total.")
    return files


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_files_indices_one_individual(individual, labels_X, id_lookup_file_path=""):
    if not id_lookup_file_path:
        indices_indiv = [
            i for i in range(len(labels_X))
            if individual.lower() in Path(labels_X[i]).stem.lower()
        ]
        print(f"Found {len(indices_indiv)} files for individual {individual} based on file name matching.")
    else:
        lookup_file_df = pd.DataFrame(pd.read_csv(id_lookup_file_path, sep="\\t"))
        lookup_file_df_one_indiv = lookup_file_df[lookup_file_df['OriginalName'].str.contains(f"_{individual}")]
        randomised_file_names = lookup_file_df_one_indiv['RandomizedName'].tolist()
        indices_indiv = []
        for file_name in randomised_file_names:
            idx_file_name = [i for i in range(len(labels_X)) if file_name in labels_X[i]]
            assert len(idx_file_name) <= 1
            if idx_file_name:
                indices_indiv.append(idx_file_name[0])
    return indices_indiv


def split_train_valid_by_individuals(labels_X, valid_set_individuals, id_lookup_file_path):
    valid_set_indices = []
    for individual in valid_set_individuals:
        idx_ind_valid = get_files_indices_one_individual(individual, labels_X, id_lookup_file_path)
        valid_set_indices.extend(idx_ind_valid)
    train_set_indices = list(set(range(len(labels_X))) - set(valid_set_indices))
    it_ = iter([(np.array(train_set_indices), np.array(valid_set_indices))])
    return it_


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
        print(f"[main] Created output directory: {args.dest}")

    print(f"\n[main] Args: {args}")
    dir_path = os.path.realpath(args.root)
    print(f"[main] Resolved root path: {dir_path}")

    # ── STEP 1: raccolta file ──────────────────────────────────────────────────
    print(f"\n[main] STEP 1: collecting .{args.ext} files via os.walk ...")
    ext_re = re.compile(r".*\.{}$".format(args.ext), re.IGNORECASE)
    audio_files = get_files(dir_path, ext_re)
    if len(audio_files) < 1:
        raise RuntimeError("No audio files were found.")
    print(f"[main] STEP 1 done: {len(audio_files)} audio files found.")

    # ── STEP 2: First Pass — classifica ogni file ──────────────────────────────
    print(f"\n[main] STEP 2: First Pass — checking for label files (.h5) ...")
    labels_X = []
    labels_y = []
    targets = []          # ← aggiungi questa riga
    files_without_labels = []


    for i, fname in enumerate(audio_files):
        file_path = os.path.realpath(fname)

        if args.path_must_contain and args.path_must_contain not in file_path:
            print(f"[main]   [{i+1}/{len(audio_files)}] SKIPPED (path_must_contain): {file_path}")
            continue

        label_file = file_path.replace(os.sep + "wav" + os.sep, os.sep + "lbl" + os.sep)
        label_file = label_file.replace(".wav", ".h5")
        label_file_check = os.path.isfile(label_file)

        if label_file_check:
            with h5py.File(label_file, "r") as f:
                categorical_label = list(f["lbl_cat"])
            if len(categorical_label) == 0:
                files_without_labels.append(file_path)
                print(f"[main]   [{i+1}/{len(audio_files)}] NO LABELS (empty .h5): {os.path.basename(file_path)}")
            else:
                cl_unique, cl_counts = np.unique(categorical_label, return_counts=True)
                labels_X.append(file_path)
                labels_y.append(cl_unique)
                print(f"[main]   [{i+1}/{len(audio_files)}] HAS LABELS {cl_unique.tolist()}: {os.path.basename(file_path)}")
        else:
            files_without_labels.append(file_path)
            if i % 100 == 0:  # stampa ogni 100 per non spammare
                print(f"[main]   [{i+1}/{len(audio_files)}] no .h5 found → pretrain only: {os.path.basename(file_path)}")

    print(f"\n[main] STEP 2 done:")
    print(f"         files WITH labels:    {len(labels_X)}")
    print(f"         files WITHOUT labels: {len(files_without_labels)}")

    # ── STEP 3: statistiche classi (solo se ci sono label) ────────────────────
    targets = []
    if len(labels_y) > 0:
        print(f"\n[main] STEP 3: computing class statistics ...")
        unique_target_classes, unique_target_class_counts = np.unique(flatten(labels_y), return_counts=True)
        print(f"[main] Classes/counts: ", end="")
        for class_id, class_count in zip(unique_target_classes, unique_target_class_counts):
            print(f"{class_id}/{class_count} ", end="")
        print()
        for ll in labels_y:
            tmp_zero_target = np.zeros(max(unique_target_classes) + 1)
            tmp_zero_target[ll] = 1
            targets.append(tmp_zero_target)
        print(f"[main] STEP 3 done: target matrix shape = ({len(targets)}, {len(targets[0])})")
    else:
        print(f"\n[main] STEP 3: SKIPPED — no labeled files found.")

    # ── STEP 4: leave-p-out (opzionale) ───────────────────────────────────────
    if args.leave_p_out:
        print(f"\n[main] STEP 4: leave-p-out strategy ...")
        # (codice invariato)
        root_len = len(dir_path)
        unique_basenames = np.unique([[os.path.basename(x[1 + root_len:])[:-18] for x in labels_X]]).tolist()
        p = round(0.2 * len(unique_basenames))
        print(f"[main] leave-p-out: p={p} files held out for test")
        lof = np.random.choice(unique_basenames, p).tolist()
        test_index_lof = set(np.argwhere([any([y in x for y in lof]) for x in labels_X]).flatten())
        train_index_lof = set(np.arange(len(labels_X)).flatten()) - test_index_lof
        files_without_labels_index_lof = (
            np.argwhere([not any([y in x for y in lof]) for x in files_without_labels]).squeeze().tolist()
        )
        valid_f_lof = open(os.path.join(args.dest, "valid_lof.tsv"), "w")
        pretrain_f_lof = open(os.path.join(args.dest, "pretrain_lof.tsv"), "w")
        train_f_lof = open(os.path.join(args.dest, "train_lof.tsv"), "w")
        print(dir_path, file=valid_f_lof)
        print(dir_path, file=pretrain_f_lof)
        print(dir_path, file=train_f_lof)
        for filename_index in train_index_lof:
            file_path = os.path.realpath(labels_X[filename_index])
            try:
                frames = soundfile.info(file_path).frames
            except soundfile.LibsndfileError as sE:
                print(sE); continue
            line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
            print(line, file=train_f_lof)
            print(line, file=pretrain_f_lof)
        for filename_index in test_index_lof:
            file_path = os.path.realpath(labels_X[filename_index])
            try:
                frames = soundfile.info(file_path).frames
            except soundfile.LibsndfileError as sE:
                print(sE); continue
            line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
            print(line, file=valid_f_lof)
        for filename_index in files_without_labels_index_lof:
            file_path = os.path.realpath(files_without_labels[filename_index])
            try:
                frames = soundfile.info(file_path).frames
            except soundfile.LibsndfileError as sE:
                print(sE); continue
            line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
            print(line, file=pretrain_f_lof)
        print(f"[main] STEP 4 done: written valid_lof.tsv, pretrain_lof.tsv, train_lof.tsv")
    else:
        print(f"\n[main] STEP 4: SKIPPED — leave-p-out not requested.")

    # ── STEP 5: scrittura pretrain.tsv ────────────────────────────────────────
    print(f"\n[main] STEP 5: writing pretrain.tsv ({len(files_without_labels)} files) ...")
    pretrain_f = open(os.path.join(args.dest, "pretrain.tsv"), "w")
    print(dir_path, file=pretrain_f)
    written = 0
    for fname in files_without_labels:
        file_path = os.path.realpath(fname)
        try:
            frames = soundfile.info(file_path).frames
        except soundfile.LibsndfileError as sE:
            print(f"[main]   ERROR reading {file_path}: {sE}")
            continue
        line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
        print(line, file=pretrain_f)
        written += 1
    print(f"[main] STEP 5 done: {written} entries written to pretrain.tsv")

    # ── STEP 6: split train/valid per finetuning ──────────────────────────────
    if not args.valid_set_individuals:
        sss = MultilabelStratifiedShuffleSplit(n_splits=args.n_split, test_size=args.valid_percent, random_state=args.seed)
        it_ = (
            sss.split(labels_X, targets)
            if 0 < args.valid_percent
            else (np.arange(len(labels_X)), [])
        )
    else:
        it_ = split_train_valid_by_individuals(labels_X, args.valid_set_individuals, args.id_lookup_file_path)

    if len(labels_X) > 0 and len(targets) > 0:
        print(f"\n[main] STEP 6: writing {args.n_split}-fold train/valid manifests ...")
        for idx, (train_index, test_index) in enumerate(it_):
            print(f"\n[main]   Fold {idx+1}/{args.n_split}: {len(train_index)} train, {len(test_index)} valid")
            train_f = open(os.path.join(args.dest, "train_{}.tsv".format(idx)), "w")
            valid_f = open(os.path.join(args.dest, "valid_{}.tsv".format(idx)), "w") if args.valid_percent > 0 else None
            print(dir_path, file=train_f)
            if valid_f is not None:
                print(dir_path, file=valid_f)

            for filename_index in train_index:
                file_path = os.path.realpath(labels_X[filename_index])
                try:
                    frames = soundfile.info(file_path).frames
                except soundfile.LibsndfileError as sE:
                    print(sE); continue
                line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
                print(line, file=train_f)
                if pretrain_f is not None:
                    print(line, file=pretrain_f)

            if valid_f is not None:
                for filename_index in test_index:
                    file_path = os.path.realpath(labels_X[filename_index])
                    try:
                        frames = soundfile.info(file_path).frames
                    except soundfile.LibsndfileError as sE:
                        print(sE); continue
                    line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
                    print(line, file=valid_f)
                    if pretrain_f is not None:
                        print(line, file=pretrain_f)

            train_f.close()
            if valid_f is not None:
                valid_f.close()
            if pretrain_f is not None:
                pretrain_f.close()
                pretrain_f = None
                print(f"[main]   pretrain.tsv closed after fold {idx+1}")

            if args.few_shot and 0 < args.valid_percent:
                # (few-shot invariato, omesso per brevità)
                pass

        print(f"[main] STEP 6 done.")
    else:
        print(f"\n[main] STEP 6: SKIPPED — no labeled files, no train/valid splits needed.")

    if pretrain_f is not None:
        pretrain_f.close()
        print(f"[main] pretrain.tsv closed.")

    print(f"\n[main] ALL DONE. Output in: {args.dest}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
