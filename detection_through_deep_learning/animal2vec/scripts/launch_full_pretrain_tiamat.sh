#!/bin/bash
# ==============================================================================
# launch_full_pretrain_tiamat.sh
# Copia dataset completo su fast storage, copia manifest, lancia pretraining
# ==============================================================================

set -e

# ==============================================================================
# VARIABILI
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
# # STEP 0: Crea le cartelle di destinazione
# # ==============================================================================

# echo ""
# echo "======================================================"
# echo " STEP 0: Creazione cartelle di destinazione"
# echo "======================================================"

# mkdir -p "$DST_WAV"
# mkdir -p "$DST_LBL"
# mkdir -p "$DST_MANIFEST"

# echo "[OK] Cartelle create:"
# echo "     WAV      -> $DST_WAV"
# echo "     LBL      -> $DST_LBL"
# echo "     MANIFEST -> $DST_MANIFEST"

# # ==============================================================================
# # STEP 1: Copia TUTTI i wav su fast storage
# # ==============================================================================

# echo ""
# echo "======================================================"
# echo " STEP 1: Copia wav su fast storage (rsync)"
# echo "======================================================"
# echo "[INFO] Sorgente:     $SRC_WAV"
# echo "[INFO] Destinazione: $DST_WAV"
# echo "[INFO] rsync copierà solo i file mancanti o modificati"

# rsync -avh \
#     --progress \
#     --ignore-errors \
#     --partial \
#     "$SRC_WAV/" \
#     "$DST_WAV/" || true

# echo "[OK] Trasferimento wav completato"
# echo "[INFO] File wav in destinazione:"
# find "$DST_WAV" -name "*.wav" | wc -l

# # ==============================================================================
# # STEP 2: Copia cartella lbl
# # ==============================================================================

# echo ""
# echo "======================================================"
# echo " STEP 2: Copia lbl su fast storage (rsync)"
# echo "======================================================"

# rsync -avh \
#     --ignore-errors \
#     "$SRC_LBL/" \
#     "$DST_LBL/" || true

# echo "[OK] Trasferimento lbl completato"

# # ==============================================================================
# # STEP 3: Copia manifest pre-esistente
# # ==============================================================================

# echo ""
# echo "======================================================"
# echo " STEP 3: Copia manifest"
# echo "======================================================"
# echo "[INFO] Copio manifest da: $SRC_MANIFEST"

# rsync -avh \
#     --ignore-errors \
#     "$SRC_MANIFEST/" \
#     "$DST_MANIFEST/" || true

# echo "[OK] Manifest copiati:"
# ls -lh "$DST_MANIFEST/"

# # ==============================================================================
# # STEP 4: Verifica spazio
# # ==============================================================================

# echo ""
# echo "======================================================"
# echo " STEP 4: Verifica spazio"
# echo "======================================================"

# echo "[INFO] Spazio usato in /local:"
# du -sh /local/*

# echo "[INFO] Spazio libero rimanente:"
# df -h /local | tail -1

# ==============================================================================
# STEP 5: Pretraining
# ==============================================================================

echo ""
echo "======================================================"
echo " STEP 5: Avvio pretraining"
echo "======================================================"

export HYDRA_FULL_ERROR=1
export HF_HOME=/abyss/home/mattia-montanari/.cache/huggingface
export TORCH_HOME=/abyss/home/mattia-montanari/.cache/torch
export TMPDIR=/abyss/home/mattia-montanari/tmp

mkdir -p "$TMPDIR"

HYDRA_RUN_DIR="/abyss/home/mattia-montanari/runs/bee-pretrain-tiamat"
mkdir -p "$HYDRA_RUN_DIR"

echo "[INFO] Avvio training con config: $CONFIG_DIR/full_pretrain_tiamat"
echo "[INFO] Hydra run dir: $HYDRA_RUN_DIR"
echo "[INFO] Timestamp avvio: $(date)"

python "$TRAIN_SCRIPT" \
    hydra.run.dir="$HYDRA_RUN_DIR" \
    --config-dir="$CONFIG_DIR" \
    --config-name=full_pretrain_tiamat

echo ""
echo "======================================================"
echo " DONE - Training completato: $(date)"
echo "======================================================"