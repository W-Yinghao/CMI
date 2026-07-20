#!/usr/bin/env bash
#SBATCH --job-name=c84l1c-canary
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00

set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

PYTHON=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python
REPO=/home/infres/yinwang/CMI_AAAI_oaci

cd "$REPO"
exec "$PYTHON" -m oaci.multidataset.c84l1_canary run-real "$@"
