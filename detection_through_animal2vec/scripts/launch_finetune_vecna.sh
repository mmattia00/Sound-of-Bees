#!/bin/bash
# ==============================================================================
# launch_finetune_vecna.sh
#
# DESCRIPTION:
#   Runs finetuning starting from a checkpoint produced during pretraining.
#
# USAGE:
#   ./launch_finetune_vecna.sh
#
#   No arguments required. All paths and configuration are hardcoded below.
#   Run from any directory; the script uses absolute paths internally.
#
# REQUIREMENTS:
#   - Python environment with Hydra, PyTorch, and animal2vec dependencies
#   - Access to the manifest directory and the pretraining checkpoint file
#
# OUTPUT:
#   - Finetuned checkpoints saved under:
#     /abyss/home/mattia-montanari/runs/bee-finetune-last-checkpoint-no-augmentation/checkpoints/
# ==============================================================================

set -e

# ==============================================================================
# VARIABLES
# ==============================================================================

TRAIN_SCRIPT="/abyss/home/timon-nas/Sound-of-Bees/detection_through_deep_learning/animal2vec/animal2vec_train.py"
CONFIG_DIR="/abyss/home/timon-nas/Sound-of-Bees/detection_through_deep_learning/animal2vec/configs/bees"

# ==============================================================================
# STEP 1: Dataset checks
# ==============================================================================
# Verify that the manifest folder and the pretraining checkpoint exist before
# starting finetuning, to fail fast instead of mid-run.

echo ""
echo "======================================================"
echo " STEP 1: Checking dataset and manifest"
echo "======================================================"

MANIFEST_ROOT="/local/labelled_dataset_no_augmentation/preprocessed_audios/Bees_10s_2026-06-15/manifest"
CHECKPOINT_PATH="/local/checkpoint_last.pt" # checkpoint from pretraining, used as the starting point for finetuning

if [ ! -d "$MANIFEST_ROOT" ]; then
    echo "[ERROR] Manifest folder not found: $MANIFEST_ROOT"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "[OK] Manifest folder found: $MANIFEST_ROOT"
echo "[OK] Checkpoint found: $CHECKPOINT_PATH"
echo "[INFO] Available manifest files:"
ls -lh "$MANIFEST_ROOT/"

# ==============================================================================
# STEP 2: Finetuning
# ==============================================================================
# Set up environment variables, prepare a clean run directory, then launch
# the finetuning job via Hydra.

echo ""
echo "======================================================"
echo " STEP 2: Starting finetuning"
echo "======================================================"

# --- Environment variables for Hydra, caches, and temp directory ---
export HYDRA_FULL_ERROR=1
export HF_HOME=/abyss/home/mattia-montanari/.cache/huggingface
export TORCH_HOME=/abyss/home/mattia-montanari/.cache/torch
export TMPDIR=/abyss/home/mattia-montanari/tmp

mkdir -p "$TMPDIR"

echo "[INFO] Start timestamp: $(date)"

HYDRA_RUN_DIR="/abyss/home/mattia-montanari/runs/bee-finetune-last-checkpoint-no-augmentation"

# Reset the run directory to ensure a clean training run.
rm -rf "$HYDRA_RUN_DIR"
mkdir -p "$HYDRA_RUN_DIR"

echo "[INFO] Checkpoint: $CHECKPOINT_PATH"

echo "======================================================"
echo " Fold ${FOLD}"
echo " Run dir: ${HYDRA_RUN_DIR}"
echo "======================================================"

python "$TRAIN_SCRIPT" \
--config-dir="$CONFIG_DIR" \
--config-name=finetune_last_chekpoint \
hydra.run.dir="$HYDRA_RUN_DIR" \
dataset.train_subset="train_0" \
dataset.valid_subset="valid_0" \
checkpoint.maximize_best_checkpoint_metric=true

# For now, always use only a single fold
# (FOLD is currently unset/unused; kept for future multi-fold support)

echo ""
echo "======================================================"
echo " DONE - Finetuning completed: $(date)"
echo "======================================================"