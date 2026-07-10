#!/bin/bash
# ==============================================================================
# launch_inference_postprocessing.sh
#
# DESCRIPTION:
#   Runs the animal2vec post-processing / inference pipeline over a batch of
#   recording days. For each day, it processes the raw audio recordings with
#   a finetuned checkpoint (bs_160s) and saves the resulting predictions.
#
# USAGE:
#   ./launch_inference_postprocessing.sh
#
#   No arguments required. The list of days to process is hardcoded in the
#   "days" array below — edit it to change which days get processed.
#
# REQUIREMENTS:
#   - Python environment with the animal2vec inference dependencies
#   - CUDA-capable GPU (device is hardcoded to "cuda")
#   - Access to the finetuned checkpoint, the raw recordings, and the
#     noisy-moments CSV file
#
# OUTPUT:
#   - Post-processed predictions saved under:
#     /abyss/home/timon-nas/postprocessing_results_animal2vec_sigma01_bs160s/<day>/
#
# NOTES:
#   - The script stops immediately on any error (set -e).
#   - A commented-out single-day test block is kept below for reference.
# ==============================================================================

set -e

printf "Importing modules...\n"

# Install timezone-related dependencies required by the pipeline
pip install backports.zoneinfo tzdata --quiet

printf '%s\n' "Modules imported successfully."

printf '%s\n' "Starting post-processing pipeline for animal2vec..."

printf 'Current working directory: %s\n' "$(pwd)"

printf '%s\n' "Changing working directory..."

# Move into the animal2vec project directory so relative script paths resolve correctly
cd /abyss/home/timon-nas/Sound-of-Bees/detection_through_deep_learning/animal2vec

printf 'Current working directory: %s\n' "$(pwd)"


# # printf '%s\n' "Trying the pipeline on one day from the previous season..."

# --- Reference: single-day test run (kept for reuse, currently disabled) ---
# in_folder="/abyss/home/dataset_raw/2025-09-15"
# results_folder="/abyss/home/timon-nas/postprocessing_results_animal2vec_bs_160s/2025-09-15"
# mkdir -p "$results_folder"

# python inference_on_a_batch.py --input-folder $in_folder --output-folder $results_folder --model-path /abyss/home/mattia-montanari/runs/bee-finetune-last-checkpoint-no-augmentation_bs_160s/checkpoints/checkpoint_best.pt --sample-rate 24000 --device cuda --metric-threshold 0.46 --sigma-s 0.1 --unique-values "['stop_signal_candidate']" --overwrite-previous-preds True --noisy-moments-csv /abyss/home/timon-nas/Sound-of-Bees/postprocessing/noisy_moments.csv


# # days=("2026-05-29" "2026-05-28" "2026-05-27")

# List of recording days to process in this batch run
days=("2026-05-30" "2026-05-31" "2026-06-01" "2026-06-02" "2026-06-03" "2026-06-04" "2026-06-05" "2026-06-06" "2026-06-07" "2026-06-08" "2026-06-09")

# Main loop: run inference + post-processing for each day independently
for day in "${days[@]}"; do
    printf '%s\n' "----------------------------------------------"
    printf '%s\n' "----------------------------------------------"
    printf '%s\n' "----------------------------------------------"
    printf 'Processing day %s...\n' "$day"
    printf '%s\n' "Starting inference post-processing pipeline..."

    results_folder="/abyss/home/timon-nas/postprocessing_results_animal2vec_sigma01_bs160s/$day"
    mkdir -p "$results_folder"

    # Run inference on the given day's audio using the bs_160s finetuned checkpoint
    python inference_on_a_batch.py --input-folder /abyss/home/timon-nas/recordings2026/audio/$day --output-folder /abyss/home/timon-nas/postprocessing_results_animal2vec_sigma01_bs160s/$day --model-path /abyss/home/mattia-montanari/runs/bee-finetune-last-checkpoint-no-augmentation_bs_160s/checkpoints/checkpoint_best.pt --sample-rate 24000 --device cuda --metric-threshold 0.46 --sigma-s 0.1 --unique-values "['stop_signal_candidate']" --overwrite-previous-preds True --noisy-moments-csv /abyss/home/timon-nas/Sound-of-Bees/postprocessing/noisy_moments.csv

    printf '%s\n' "Inference post-processing pipeline completed successfully."

    printf '%s\n' "----------------------------------------------"
done