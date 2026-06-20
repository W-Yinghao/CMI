#!/bin/bash
#SBATCH --job-name=cmi-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# CPU-only passthrough (for TSMNet / classical Riemannian / any non-GPU job).
#   sbatch -p cpu-high scripts/cpu.slurm cmi.run_loso --dataset BNCI2014_001 --backbone TSMNet ...
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mpl-${SLURM_JOB_ID:-0}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
MOD=$1; shift
echo "MODULE=$MOD ARGS: $*"
srun "$PY" -m "$MOD" "$@"
