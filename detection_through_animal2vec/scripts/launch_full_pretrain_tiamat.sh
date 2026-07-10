#!/bin/bash
# ==============================================================================
# launch_full_pretrain_tiamat.sh
#
# DESCRIPTION:
#   Copies the full dataset to fast local storage, copies the manifest, and
#   launches pretraining on the copied data.
#
# USAGE:
#   ./launch_full_pretrain_tiamat.sh
#
#   No arguments required. All paths and configuration are hardcoded below.
#   Run from any directory; the script uses absolute paths internally.
#
#   NOTE: Steps 0-4 (dataset/manifest copy and disk space checks) are
#   currently commented out in this script — they were presumably already
#   run once and are kept here for reference / re-use if the data needs to
#   be re-synced. Only Step 5 (pretraining) is active by default.
#
# REQUIREMENTS:
#   - Python environment with Hydra, PyTorch, and animal2vec dependencies
#   - If Steps 0-4 are re-enabled: read access to the source dataset and
#     write access + sufficient free space on /local
#
# OUTPUT:
#   - Pretraining checkpoints/logs saved under:
#     /abyss/home/mattia-montanari/runs/bee-pretrain-tiamat/
#
# NOTES:
#   - The script stops immediately on any error (set -e).
#   - rsync steps (when enabled) use `|| true` so a partial rsync failure
#     does not abort the whole script.
# ==============================================================================

set -e

# ==============================================================================
# VARIABLES
# ==============================================================================

SRC_WAV="/abyss/home/dataset_preprocessed/SoundOfBees_10s_2026-03-05/wav/24000Hz/byss_home_dataset_raw"
SRC_LBL="/abyss/home/dataset_preprocessed/SoundOfBees_10s_2026-03-05/lbl"
SRC_MANIFEST="/abyss/home/dataset_preprocessed/SoundOfBees_10s_2026-03-05/manifest"

DST_BASE="/local/SoundOfBees_10s_full"
DST_WAV="$DST_BASE/wav/24000Hz/byss_home_dataset_raw"
DST_LBL="$DST_BASE/lbl"
DST_MANIFEST="$DST_BASE/manifest"

TRAIN_SCRIPT="/abyss/home/timon-nas/Sound-of-Bees/detection_through_deep_learning/animal2vec/animal2vec_train.py"
CONFIG_DIR="/abyss/home/timon-nas/Sound-of-Bees/detection_through_deep_learning/animal2vec/configs/bees"


# # ==============================================================================
# # STEP 0: Create destination folders
# # ==============================================================================

# echo ""
# echo "======================================================"
# echo " STEP 0: Creating destination folders"
# echo "======================================================"

# mkdir -p "$DST_WAV"
# mkdir -p "$DST_LBL"
# mkdir -p "$DST_MANIFEST"

# echo "[OK] Folders created:"
# echo "     WAV      -> $DST_WAV"
# echo "     LBL      -> $DST_LBL"
# echo "     MANIFEST -> $DST_MANIFEST"


# # ==============================================================================
# # STEP 1: Copy ALL wav files to fast storage
# # ==============================================================================

# echo ""
# echo "======================================================"
# echo " STEP 1: Copying wav files to fast storage (rsync)"
# echo "======================================================"
# echo "[INFO] Source:      $SRC_WAV"
# echo "[INFO] Destination: $DST_WAV"
# echo "[INFO] rsync will copy only missing or modified files"

# rsync -avh \
#     --progress \
#     --ignore-errors \
#     --partial \
#     "$SRC_WAV/" \
#     "$DST_WAV/" || true

# echo "[OK] wav transfer completed"
# echo "[INFO] wav files at destination:"
# find "$DST_WAV" -name "*.wav" | wc -l


# # ==============================================================================
# # STEP 2: Copy lbl folder
# # ==============================================================================

# echo ""
# echo "======================================================"
# echo " STEP 2: Copying lbl to fast storage (rsync)"
# echo "======================================================"

# rsync -avh \
#     --ignore-errors \
#     "$SRC_LBL/" \
#     "$DST_LBL/" || true

# echo "[OK] lbl transfer completed"


# # ==============================================================================
# # STEP 3: Copy pre-existing manifest
# # ==============================================================================

# echo ""
# echo "======================================================"
# echo " STEP 3: Copying manifest"
# echo "======================================================"
# echo "[INFO] Copying manifest from: $SRC_MANIFEST"

# rsync -avh \
#     --ignore-errors \
#     "$SRC_MANIFEST/" \
#     "$DST_MANIFEST/" || true

# echo "[OK] Manifest copied:"
# ls -lh "$DST_MANIFEST/"


# # ==============================================================================
# # STEP 4: Check disk space
# # ==============================================================================

# echo ""
# echo "======================================================"
# echo " STEP 4: Checking disk space"
# echo "======================================================"

# echo "[INFO] Space used in /local:"
# du -sh /local/*

# echo "[INFO] Remaining free space:"
# df -h /local | tail -1


# ==============================================================================
# STEP 5: Pretraining
# ==============================================================================
# Set up environment variables, prepare the Hydra run directory, then launch
# the pretraining job on the (previously copied) fast-storage dataset.

echo ""
echo "======================================================"
echo " STEP 5: Starting pretraining"
echo "======================================================"

# --- Environment variables for Hydra, caches, and temp directory ---
export HYDRA_FULL_ERROR=1
export HF_HOME=/abyss/home/mattia-montanari/.cache/huggingface
export TORCH_HOME=/abyss/home/mattia-montanari/.cache/torch
export TMPDIR=/abyss/home/mattia-montanari/tmp

mkdir -p "$TMPDIR"

HYDRA_RUN_DIR="/abyss/home/mattia-montanari/runs/bee-pretrain-tiamat"
mkdir -p "$HYDRA_RUN_DIR"

echo "[INFO] Starting training with config: $CONFIG_DIR/full_pretrain_tiamat"
echo "[INFO] Hydra run dir: $HYDRA_RUN_DIR"
echo "[INFO] Start timestamp: $(date)"

python "$TRAIN_SCRIPT" \
    hydra.run.dir="$HYDRA_RUN_DIR" \
    --config-dir="$CONFIG_DIR" \
    --config-name=full_pretrain_tiamat

echo ""
echo "======================================================"
echo " DONE - Training completed: $(date)"
echo "======================================================"