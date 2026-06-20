#!/bin/bash
#SBATCH --job-name=cmi-xdata
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Generic runner: first arg = module, rest passed through.
#   sbatch -p A100 scripts/runmod.slurm cmi.run_cross_dataset --backbone EEGNet --configs erm:0 lpc_prior:0.3 ...
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mpl-${SLURM_JOB_ID:-0}
MOD=$1; shift
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "MODULE=$MOD ARGS: $*"
srun "$PY" -m "$MOD" "$@"
