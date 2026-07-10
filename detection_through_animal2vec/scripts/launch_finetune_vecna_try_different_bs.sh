#!/bin/bash
# ==============================================================================
# launch_finetune_all_bs.sh
#
# DESCRIPTION:
#   Runs finetuning + evaluation for all batch sizes (excluding 280s, already
#   completed). Sequence: 80s → 160s → 560s → 1120s
#
# USAGE:
#   ./launch_finetune_all_bs.sh
#
#   No arguments required. All paths and configuration are hardcoded below.
#   Run from any directory; the script uses absolute paths internally.
#
# REQUIREMENTS:
#   - Python environment with Hydra, PyTorch, and animal2vec dependencies
#   - Access to the manifest directory and the last checkpoint file
#   - Sufficient disk space in RUNS_BASE and ESTIMATES_BASE for outputs
#
# OUTPUT:
#   - Trained checkpoints saved under: $RUNS_BASE/bee-finetune-last-checkpoint-no-augmentation_<bs_label>/checkpoints/
#   - Evaluation predictions saved under: $ESTIMATES_BASE/batch_size_<bs_label>/
# ==============================================================================

set -e

# --- Script paths ---
TRAIN_SCRIPT="/abyss/home/timon-nas/Sound-of-Bees/detection_through_deep_learning/animal2vec/animal2vec_train.py"
EVAL_SCRIPT="/abyss/home/timon-nas/Sound-of-Bees/detection_through_deep_learning/animal2vec/get_results_for_single_manifest_split.py"
CONFIG_DIR="/abyss/home/timon-nas/Sound-of-Bees/detection_through_deep_learning/animal2vec/configs/bees"

# --- Dataset and checkpoint paths ---
MANIFEST_ROOT="/local/labelled_dataset_no_augmentation/preprocessed_audios/Bees_10s_2026-06-15/manifest"
CHECKPOINT_PATH="/local/checkpoint_last.pt"

# --- Model / evaluation parameters ---
UNIQUE_LABELS="['stop_signal_candidate']"
CONV_LAYERS="[(127, 63, 1)] +[(512, 10, 5)] + [(512, 3, 4)] + [(512, 3, 3)] + [(512, 3, 2)] + [(512, 3, 1)] + [(512, 2, 1)] * 2"

# ==============================================================================
# STEP 1: Dataset checks
# ==============================================================================
# Verify that the manifest folder and the base checkpoint exist before
# starting any training run, to fail fast instead of mid-loop.

echo ""
echo "======================================================"
echo " STEP 1: Checking dataset and manifest"
echo "======================================================"

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
# STEP 2: Finetuning + evaluation loop
# ==============================================================================
# For each batch size configuration: run finetuning, then evaluate the best
# checkpoint produced by that run.

echo ""
echo "======================================================"
echo " STEP 2: Starting finetuning + evaluation loop"
echo "======================================================"

# --- Environment variables for Hydra, caches, and temp directory ---
export HYDRA_FULL_ERROR=1
export HF_HOME=/abyss/home/mattia-montanari/.cache/huggingface
export TORCH_HOME=/abyss/home/mattia-montanari/.cache/torch
export TMPDIR=/abyss/home/mattia-montanari/tmp
mkdir -p "$TMPDIR"

RUNS_BASE="/abyss/home/mattia-montanari/runs"
ESTIMATES_BASE="/abyss/home/timon-nas/a2v_estimates"

# List of (config_name:batch_size_label) pairs to process in sequence.
# Note: "chekpoint" (typo) matches the actual YAML config filenames.
declare -a CONFIGS=(
    "finetune_last_chekpoint_bs_80s:bs_80s"
    "finetune_last_chekpoint_bs_160s:bs_160s"
    "finetune_last_chekpoint_bs_560s:bs_560s"
    "finetune_last_chekpoint_bs_1120s:bs_1120s"
)

TOTAL=${#CONFIGS[@]}
CURRENT=0

for ENTRY in "${CONFIGS[@]}"; do
    # Split "config_name:batch_size_label" into its two components
    CONFIG_NAME="${ENTRY%%:*}"
    BS_LABEL="${ENTRY##*:}"
    CURRENT=$((CURRENT + 1))

    HYDRA_RUN_DIR="${RUNS_BASE}/bee-finetune-last-checkpoint-no-augmentation_${BS_LABEL}"
    BEST_CHECKPOINT="${HYDRA_RUN_DIR}/checkpoints/checkpoint_best.pt"
    EVAL_OUT_DIR="${ESTIMATES_BASE}/batch_size_${BS_LABEL}"

    echo ""
    echo "======================================================"
    echo " [${CURRENT}/${TOTAL}] Batch size: ${BS_LABEL}"
    echo " Config:    ${CONFIG_NAME}"
    echo " Train dir: ${HYDRA_RUN_DIR}"
    echo " Eval dir:  ${EVAL_OUT_DIR}"
    echo " Started:   $(date)"
    echo "======================================================"

    # --- Finetuning ---
    # Reset the run directory to ensure a clean training run each time.
    echo "[INFO] Starting finetuning..."
    rm -rf "$HYDRA_RUN_DIR"
    mkdir -p "$HYDRA_RUN_DIR"

    python "$TRAIN_SCRIPT" \
        --config-dir="$CONFIG_DIR" \
        --config-name="$CONFIG_NAME" \
        hydra.run.dir="$HYDRA_RUN_DIR" \
        dataset.train_subset="train_0" \
        dataset.valid_subset="valid_0" \
        checkpoint.maximize_best_checkpoint_metric=true

    echo "[OK] Finetuning completed: $(date)"

    # --- Checkpoint check ---
    # If the best checkpoint wasn't produced, skip evaluation for this batch
    # size but continue with the remaining configurations.
    if [ ! -f "$BEST_CHECKPOINT" ]; then
        echo "[ERROR] checkpoint_best.pt not found in ${HYDRA_RUN_DIR}/checkpoints/"
        echo "[SKIP]  Skipping evaluation for ${BS_LABEL}"
        continue
    fi

    # --- Evaluation ---
    # Run evaluation using the best checkpoint from this finetuning run.
    echo "[INFO] Starting evaluation with checkpoint_best.pt..."
    mkdir -p "$EVAL_OUT_DIR"

    python "$EVAL_SCRIPT" \
        --model_path "$BEST_CHECKPOINT" \
        --manifest_path "$MANIFEST_ROOT" \
        --unique_labels "$UNIQUE_LABELS" \
        --sample_rate 24000 \
        --conv_feature_layers "$CONV_LAYERS" \
        --out_path "$EVAL_OUT_DIR" \
        --export_predictions True \
        --min_label_size 0

    echo "[OK] Evaluation completed: $(date)"
    echo "[OK] Predictions saved to: ${EVAL_OUT_DIR}"
    echo "======================================================"

done

# ==============================================================================
# Final summary
# ==============================================================================
echo ""
echo "======================================================"
echo " DONE - All runs completed: $(date)"
echo ""
echo " Checkpoints:"
for ENTRY in "${CONFIGS[@]}"; do
    BS_LABEL="${ENTRY##*:}"
    echo "   ${RUNS_BASE}/bee-finetune-last-checkpoint-no-augmentation_${BS_LABEL}/checkpoints/"
done
echo ""
echo " Predictions h5:"
for ENTRY in "${CONFIGS[@]}"; do
    BS_LABEL="${ENTRY##*:}"
    echo "   ${ESTIMATES_BASE}/batch_size_${BS_LABEL}/"
done
echo "======================================================"