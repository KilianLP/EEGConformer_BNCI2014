#!/bin/bash
#SBATCH --job-name=EEGFormer
#SBATCH --output=logs/eegconformer.out
#SBATCH --error=logs/eegconformer.err
#SBATCH --gres=gpu:rtx3090:1               
#SBATCH --partition=Brain       
#SBATCH --cpus-per-gpu=4            

export HOME="/Brain/private/m22jocal"               
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"

cd /Brain/private/m22jocal/EEGConformer_BNCI2014
source .venv/bin/activate
python train.py --attention simpleattention
