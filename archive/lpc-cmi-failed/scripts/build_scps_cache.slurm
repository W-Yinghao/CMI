#!/bin/bash
#SBATCH --job-name=scz-cache
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Build a cross-dataset SCPS cache (CPU job; BIDS reading is slow).
#   sbatch -p CPU scripts/build_scps_cache.slurm SCZ ds003944,ds003947,ds004000,ds004367
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mpl-${SLURM_JOB_ID:-0}
echo "BUILD CACHE: $*"
srun "$PY" scripts/build_scps_cache.py "$@"
