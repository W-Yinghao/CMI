#!/bin/bash
#SBATCH --job-name=cmi-loso
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Thin passthrough wrapper: all args go straight to cmi.run_loso.
#   sbatch -p V100 scripts/run.slurm --dataset BNCI2014_001 --backbone EEGConformer \
#       --configs erm:0 lpc_prior:0.3 --tmin 0 --tmax 4 --resample 250 --out results/x.json
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mpl-${SLURM_JOB_ID:-0}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "ARGS: $*"
srun "$PY" -m cmi.run_loso "$@"
