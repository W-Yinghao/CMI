#!/bin/bash
#SBATCH --job-name=fsr-dump
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
srun /home/infres/yinwang/anaconda3/envs/eeg2025/bin/python scripts/cb_cbm_feature_dump.py --model $1 --dataset shu --device cuda --bs ${2:-64}
