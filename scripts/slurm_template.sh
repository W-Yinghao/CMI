#!/bin/bash
#SBATCH --job-name=cmi
#SBATCH --partition=V100             # V100 | V100-32GB | A100  (per project instruction)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00              # V100=2d max, A100=1d max
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

# Offline MOABB/MNE cache (read-only datalake).
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Usage: sbatch scripts/slurm_template.sh <module/script> [args...]
srun "$PY" "$@"
