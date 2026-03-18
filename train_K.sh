#!/bin/bash
#SBATCH --job-name=EEGConformer
#SBATCH --output=logs/eeg_conformer.out
#SBATCH --error=logs/eeg_conformer.err
#SBATCH --gres=gpu:rtx3090:1               
#SBATCH --partition=Brain       
#SBATCH --cpus-per-gpu=4            
#SBATCH --qos=highbrain

export HOME="/Brain/private/k23preus"               
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"

echo "starting job on $(hostname) at $(date)"

cd /Brain/private/k23preus/EEGConformer_BNCI2014
source venv/bin/activate
python train.py --n-seeds 3
