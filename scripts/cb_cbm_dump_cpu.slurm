#!/bin/bash
#SBATCH --job-name=fsr-cbm-dump
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
srun /home/infres/yinwang/anaconda3/envs/eeg2025/bin/python scripts/cb_cbm_feature_dump.py --model cbramod --dataset shu --device cpu
