#!/bin/bash
#SBATCH --job-name=fsr-temporal
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs
export PYTHONUNBUFFERED=1
srun /home/infres/yinwang/anaconda3/envs/eeg2025/bin/python scripts/codebrain_temporal_sidecheck.py
