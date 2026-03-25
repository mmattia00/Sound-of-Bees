#!/bin/sh

# Update apt
apt update
# Declare unattended
export DEBIAN_FRONTEND=noninteractive
# set max threads
export NUMEXPR_MAX_THREADS=128
# install base stuff from official repos
apt -y install git zsh curl nano htop nvtop python3-sklearn python3-pandas python3-sklearn-pandas python3-seaborn python3-matplotlib python3-pkgconfig ffmpeg libavcodec-dev libavformat-dev libavfilter-dev libavdevice-dev
# upgrade pip
python3 -m pip install --upgrade pip==24.0 setuptools==69.5.1 wheel==0.40.0
# change directory to torchaudio and build it
cd /abyss/home/ml/audio
rm -rf build
USE_FFMPEG=1 python setup.py develop
# change directory to animal2vec repo
# animal2vec repo - sostituisci /path/to/code con:
CODE=/abyss/home/Sound-of-Bees/detection_through_deep_learning/animal2vec
cd $CODE
# install minimum with pip
pip install tensorflow==2.12.0 tensorboardX==2.6.1 scikit-image==0.21.0 intervaltree==3.1.0 pyarrow==12.0.1 umap-learn==0.5.3 hydra-core==1.0.7 omegaconf==2.0.6 bitarray==2.7.5 sacrebleu==1.4.12 nvitop==1.1.2 tabulate==0.9.0 iterative-stratification
# install timm without dependencies, as otherwise torch and everything will be pulled
pip install -U --no-dependencies timm==0.9.2
# export an updated Pythonpath variable
export PYTHONPATH="${PYTHONPATH}:/abyss/home/ml/fairseq:/abyss/home/ml/audio"
# run the pretrain
python3 $CODE/animal2vec_train.py hydra.run.dir="outputs/my_pretrained_model" --config-dir=$CODE/configs/bees --config-name test_julian_setup.yaml

# Finetune
# python3 /path/to/code/animal2vec_train.py hydra.run.dir="outputs/my_finetuned_model" model.w2v_path="/path/to/code/outputs/my_pretrained_model/checkpoints/checkpoint_last.pt" --config-dir=/path/to/code/configs/ --config-name finetune_mixup_100
