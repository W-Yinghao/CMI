#!/bin/bash
#SBATCH --job-name=fsr-cbm-dump
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs
export PYTHONUNBUFFERED=1
srun /home/infres/yinwang/anaconda3/envs/eeg2025/bin/python scripts/cb_cbm_feature_dump.py --model cbramod --dataset shu --device cuda
