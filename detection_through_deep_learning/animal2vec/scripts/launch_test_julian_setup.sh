#!/bin/sh

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

# change directory to animal2vec repo
# animal2vec repo - sostituisci /path/to/code con:
CODE=/abyss/home/Sound-of-Bees/detection_through_deep_learning/animal2vec

# run the pretrain
python3 $CODE/animal2vec_train.py hydra.run.dir="outputs/my_pretrained_model" --config-dir=$CODE/configs/bees --config-name test_julian_setup.yaml

# Finetune
# python3 /path/to/code/animal2vec_train.py hydra.run.dir="outputs/my_finetuned_model" model.w2v_path="/path/to/code/outputs/my_pretrained_model/checkpoints/checkpoint_last.pt" --config-dir=/path/to/code/configs/ --config-name finetune_mixup_100
