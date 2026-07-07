#!/bin/bash
#SBATCH --job-name=fsr-8c1-scaling
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python scripts/physio_8c1_scaling.py
echo "=== 8C-1 scaling done ==="
