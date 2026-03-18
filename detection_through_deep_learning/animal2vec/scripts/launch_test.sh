#!/bin/bash
# ==============================================================================
# launch_test.sh
# Copia metà dataset su fast storage, crea manifest, lancia pretraining
# ==============================================================================

# "set -e" fa uscire lo script immediatamente se un comando fallisce con errore.
# Lo usiamo per bloccarci subito su errori gravi (es. mkdir fallisce).
# Per rsync invece gestiamo gli errori manualmente — vedi sotto.
set -e

# ==============================================================================
# VARIABILI — tutti i path in un posto solo, facile da modificare
# ==============================================================================

SRC_WAV="/abyss/home/dataset_preprocessed/SoundOfBees_10s_2026-03-05/wav/24000Hz/byss_home_dataset_raw"
SRC_LBL="/abyss/home/dataset_preprocessed/SoundOfBees_10s_2026-03-05/lbl"

DST_BASE="/local/SoundOfBees_10s_half"
DST_WAV="$DST_BASE/wav/24000Hz/byss_home_dataset_raw"
DST_LBL="$DST_BASE/lbl"
DST_MANIFEST="$DST_BASE/manifest"

MANIFEST_SCRIPT="/abyss/home/Sound-of-Bees/detection_through_deep_learning/animal2vec/scripts/animal2vec_manifest.py"
TRAIN_SCRIPT="/abyss/home/Sound-of-Bees/detection_through_deep_learning/animal2vec/animal2vec_train.py"
CONFIG_DIR="/abyss/home/Sound-of-Bees/detection_through_deep_learning/animal2vec/configs/bees"

FILE_LIST="/tmp/wav_half_list.txt"

# ==============================================================================
# STEP 0: Crea le cartelle di destinazione
# ==============================================================================

echo ""
echo "======================================================"
echo " STEP 0: Creazione cartelle di destinazione"
echo "======================================================"

# -p significa "crea anche le cartelle intermedie se non esistono"
# e non da errore se esistono già
mkdir -p "$DST_WAV"
mkdir -p "$DST_LBL"
mkdir -p "$DST_MANIFEST"

echo "[OK] Cartelle create:"
echo "     WAV  -> $DST_WAV"
echo "     LBL  -> $DST_LBL"
echo "     MANIFEST -> $DST_MANIFEST"

# ==============================================================================
# STEP 1: Conta i file wav e costruisce la lista della metà
# ==============================================================================

echo ""
echo "======================================================"
echo " STEP 1: Selezione metà file wav"
echo "======================================================"

# "find ... -name '*.wav'" trova tutti i file wav ricorsivamente
# "-printf '%P\n'" stampa il path RELATIVO rispetto alla dir sorgente (senza il prefix $SRC_WAV/)
#   questo è importante per rsync --files-from che vuole path relativi
# "wc -l" conta le righe = numero file
TOTAL=$(find "$SRC_WAV" -name "*.wav" -printf "%P\n" | wc -l)
HALF=$((TOTAL / 2))   # divisione intera in bash, es: 1001/2 = 500

echo "[INFO] File wav totali trovati: $TOTAL"
echo "[INFO] File che verranno copiati (metà): $HALF"

# Genera la lista dei primi HALF file e la salva in un file temporaneo
# "head -n $HALF" prende le prime $HALF righe
find "$SRC_WAV" -name "*.wav" -printf "%P\n" | head -n "$HALF" > "$FILE_LIST"

echo "[OK] Lista file salvata in: $FILE_LIST"

# ==============================================================================
# STEP 2: Copia metà dei wav su fast storage
# ==============================================================================

echo ""
echo "======================================================"
echo " STEP 2: Copia wav su fast storage (rsync)"
echo "======================================================"
echo "[INFO] Sorgente: $SRC_WAV"
echo "[INFO] Destinazione: $DST_WAV"
echo "[INFO] Inizio trasferimento..."

# rsync flags spiegati:
#   -a  = "archive mode": preserva permessi, timestamp, ricorsivo — equivale a -rlptgoD
#   -v  = verbose: stampa ogni file copiato
#   -h  = human readable: mostra dimensioni in KB/MB/GB
#   --progress        = mostra progresso per ogni file
#   --ignore-errors   = se un file fallisce, continua invece di fermarsi
#   --partial         = se il trasferimento si interrompe, mantieni il file parziale
#                       così al prossimo rsync riprende da dove si era fermato
#   --files-from      = legge la lista di file da copiare dal file specificato
#                       i path nella lista sono RELATIVI rispetto a $SRC_WAV/
#
# "|| true" alla fine: normalmente con "set -e" se rsync finisce con errore
# lo script si ferma. "|| true" dice "se rsync fallisce, considera comunque OK"
# utile perché --ignore-errors può far uscire rsync con codice non-zero
# anche se ha copiato tutto quello che poteva

rsync -avh \
    --progress \
    --ignore-errors \
    --files-from="$FILE_LIST" \
    "$SRC_WAV/" \
    "$DST_WAV/" || true

echo "[OK] Trasferimento wav completato"
echo "[INFO] File effettivamente copiati in destinazione:"
find "$DST_WAV" -name "*.wav" | wc -l

# ==============================================================================
# STEP 3: Copia tutta la cartella lbl
# ==============================================================================

echo ""
echo "======================================================"
echo " STEP 3: Copia lbl su fast storage (rsync)"
echo "======================================================"

# Nota il trailing slash su $SRC_LBL/ — in rsync è importante:
#   CON slash finale:    copia il CONTENUTO di lbl/ dentro $DST_LBL/
#   SENZA slash finale:  copia la CARTELLA lbl/ dentro $DST_LBL/ (crea $DST_LBL/lbl/)
rsync -avh \
    --ignore-errors \
    "$SRC_LBL/" \
    "$DST_LBL/" || true

echo "[OK] Trasferimento lbl completato"

# ==============================================================================
# STEP 4: Verifica spazio usato sul fast storage
# ==============================================================================

echo ""
echo "======================================================"
echo " STEP 4: Verifica spazio"
echo "======================================================"

echo "[INFO] Spazio usato in /local:"
du -sh /local/*

echo "[INFO] Spazio libero rimanente su /dev/md0:"
df -h /local | tail -1

# ==============================================================================
# STEP 5: Crea i manifest puntando al fast storage
# ==============================================================================

echo ""
echo "======================================================"
echo " STEP 5: Creazione manifest"
echo "======================================================"

# Controlla se i manifest esistono già — se sì, salta la creazione
if [ -f "$DST_MANIFEST/pretrain.tsv" ]; then
    echo "[SKIP] Manifest già esistenti, salto la creazione"
    ls -lh "$DST_MANIFEST/"
else
    echo "[INFO] Il manifest punterà ai file in: $DST_WAV"
    echo "[INFO] I manifest verranno salvati in: $DST_MANIFEST"

    python "$MANIFEST_SCRIPT" \
        "$DST_WAV" \
        --dest "$DST_MANIFEST" \
        --ext wav \
        --valid-percent 0.1 \
        --n-split 1

    echo "[OK] Manifest creati:"
    ls -lh "$DST_MANIFEST/"
fi

# ==============================================================================
# STEP 6: Pretraining
# ==============================================================================

echo ""
echo "======================================================"
echo " STEP 6: Avvio pretraining"
echo "======================================================"

# Variabili d'ambiente — spiegazione:
#   HF_HOME    = dove HuggingFace scarica/cacha i modelli
#   TORCH_HOME = dove PyTorch scarica pesi pretrainati
#   TMPDIR     = cartella per file temporanei di sistema
# Tutte su CephFS così sopravvivono al pod e non riempiono il container layer

export HYDRA_FULL_ERROR=1
export HF_HOME=/abyss/home/mattia-montanari/.cache/huggingface
export TORCH_HOME=/abyss/home/mattia-montanari/.cache/torch
export TMPDIR=/abyss/home/mattia-montanari/tmp

mkdir -p "$TMPDIR"
mkdir -p /abyss/home/checkpoints/bee-pretrain-test
mkdir -p /abyss/home/tb_logs/bee-pretrain-test

echo "[INFO] Avvio training con config: $CONFIG_DIR/test_pretrain"
echo "[INFO] Checkpoints -> /abyss/home/checkpoints/bee-pretrain-test"
echo "[INFO] TensorBoard -> /abyss/home/tb_logs/bee-pretrain-test"
echo "[INFO] Timestamp avvio: $(date)"

python "$TRAIN_SCRIPT" \
    --config-dir="$CONFIG_DIR" \
    --config-name=test_pretrain

echo ""
echo "======================================================"
echo " DONE - Training completato: $(date)"
echo "======================================================"