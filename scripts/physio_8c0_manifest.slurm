#!/bin/bash
#SBATCH --job-name=fsr-8c0-manifest
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs
export PYTHONUNBUFFERED=1
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python scripts/physio_8c0_manifest.py
