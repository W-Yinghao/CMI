#!/bin/bash
#SBATCH --job-name=fsr-cb-real
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
export PYTHONUNBUFFERED=1
srun "$PY" scripts/codebrain_real_forward.py
