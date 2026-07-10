#!/bin/bash
set -e

# Batch wrapper for the classical DSP post-processing pipeline.
# It's the bash script launched by the job scheduler on the ccu cluster to run the post-processing pipeline for multiple days. 
# It runs the multichannel candidate extraction, then synchronizes audio and video
# for each selected day and stores the outputs in the per-day results folder.

printf '%s\n' "Starting post-processing pipeline for classical DSP..."
printf '%s\n' "Changing working directory..."

cd ../abyss/home/timon-nas/Sound-of-Bees/postprocessing

printf 'Current working directory: %s\n' "$(pwd)"

# days=("2026-05-27" "2026-05-28" "2026-05-29" "2026-05-30" "2026-05-31")
# days=("2026-06-01" "2026-06-02" "2026-06-03" "2026-06-04")
# days=("2026-06-05" "2026-06-06" "2026-06-07" "2026-06-08" "2026-06-09")
# Select the days to process by editing this list.
# days=("2026-05-27" "2026-05-28" "2026-05-29" "2026-05-30" "2026-05-31")
# days=("2026-06-01" "2026-06-02" "2026-06-03" "2026-06-04")
# days=("2026-06-05" "2026-06-06" "2026-06-07" "2026-06-08" "2026-06-09")
days=("2026-06-10" "2026-06-11")

for day in "${days[@]}"; do
    printf '%s\n' "----------------------------------------------"
    printf '%s\n' "----------------------------------------------"
    printf '%s\n' "----------------------------------------------"
    printf 'Processing day %s...\n' "$day"
    printf '%s\n' "Starting audio post-processing pipeline..."

    # Start a new per-day processing block.
    results_folder="/abyss/home/timon-nas/postprocessing_results/$day"
    mkdir -p "$results_folder"

    # Skip the expensive post-processing step if the final HDF5 database already exists.
    if [ ! -f "$results_folder/results_$day.h5" ]; then
        printf 'Results file does not exist at %s. Proceeding with post-processing.\n' "$results_folder/results_$day.h5"
        python postprocessing_pipeline_classical_dsp_ccu.py \
            --input-folder "/abyss/home/timon-nas/recordings2026/audio/$day" \
            --results-folder "$results_folder" \
            --num-workers 32
        printf '%s\n' "Post-processing pipeline completed successfully."
    fi

    printf '%s\n' "----------------------------------------------"
    # Synchronize the processed audio database with the corresponding video folder.
    printf 'Starting audio-video synchronization for day %s...\n' "$day"
    python sync_audio_video.py \
        --db "/abyss/home/timon-nas/postprocessing_results/$day/results_$day.h5" \
        --audio-folder "/abyss/home/timon-nas/recordings2026/audio/$day/" \
        --video-folder "/abyss/home/timon-nas/recordings2026/video/$day/" \
        --output-dir "$results_folder"
done