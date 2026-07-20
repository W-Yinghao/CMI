#!/usr/bin/env bash
#SBATCH --job-name=c75-finalize
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c75-representation-construct/logs/%x_%j.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c75-representation-construct/logs/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/tmp/c75-pycache-${SLURM_JOB_ID}

mkdir -p /projects/EEG-foundation-model/yinghao/oaci-c75-representation-construct/logs
cd /home/infres/yinwang/CMI_AAAI_oaci

/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python \
  -m oaci.conditioned_ceiling_coverage.c75_finalize
