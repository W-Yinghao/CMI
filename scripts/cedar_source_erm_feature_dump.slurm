#!/usr/bin/env bash
#SBATCH --job-name=cedar01f-srcerm
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/cedar/feature_supply

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mpl-${SLURM_JOB_ID:-0}-${SLURM_ARRAY_TASK_ID:-0}

BACKBONES=(EEGNetMini EEGConformerMini)
BACKBONE=${BACKBONES[${SLURM_ARRAY_TASK_ID:-0}]}
OUT_DIR="results/cedar/feature_supply/cedar01f_bnci2014_001_seed0"

echo "CEDAR_01F Route C source-ERM feature dump"
echo "backbone=${BACKBONE}"
echo "out_dir=${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m cedar_eeg.runners.run_01f_source_erm_feature_dump \
  --dataset BNCI2014_001 \
  --backbones "${BACKBONE}" \
  --seed 0 \
  --out-dir "${OUT_DIR}" \
  --epochs "${CEDAR_01F_EPOCHS:-160}" \
  --bs "${CEDAR_01F_BS:-64}" \
  --device cuda
