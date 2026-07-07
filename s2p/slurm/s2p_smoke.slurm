#!/bin/bash
#SBATCH --job-name=s2p-smoke
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_s2p
mkdir -p logs results/s2p_pretrain_smoke
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
srun /home/infres/yinwang/anaconda3/envs/eeg2025/bin/python s2p/scripts/s2p_smoke.py --model $1 --n_subjects $2 --hours 50 --steps ${3:-30}
