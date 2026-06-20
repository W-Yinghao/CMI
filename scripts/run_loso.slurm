#!/bin/bash
#SBATCH --job-name=cmi-loso
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Usage:
#   sbatch scripts/run_loso.slurm <DATASET> <BACKBONE> "<CONFIGS:method:lam ...>" [EPOCHS] [TAG]
#   sbatch scripts/run_loso.slurm BNCI2014_001 EEGNet "erm:0 marginal:1 chain:1 lpc_uniform:1 lpc_prior:1" 300
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results

DATASET=${1:-BNCI2014_001}
BACKBONE=${2:-EEGNet}
CONFIGS=${3:-"erm:0 marginal:1 chain:1 lpc_uniform:1 lpc_prior:1"}
EPOCHS=${4:-300}
TAG=${5:-}
RESAMPLE=${6:-128}

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mpl-${SLURM_JOB_ID:-0}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "DATASET=$DATASET BACKBONE=$BACKBONE CONFIGS=$CONFIGS EPOCHS=$EPOCHS"

srun "$PY" -m cmi.run_loso \
    --dataset "$DATASET" --backbone "$BACKBONE" --configs $CONFIGS \
    --epochs "$EPOCHS" --resample "$RESAMPLE" \
    --out "results/${DATASET}_${BACKBONE}${TAG}.json"
