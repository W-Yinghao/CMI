#!/usr/bin/env bash
#SBATCH --job-name=c76-analyze
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c76-representation-association/logs/%x_%j.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c76-representation-association/logs/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/tmp/c76-pycache-${SLURM_JOB_ID}

mkdir -p /projects/EEG-foundation-model/yinghao/oaci-c76-representation-association/logs
cd /home/infres/yinwang/CMI_AAAI_oaci

/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python \
  -m oaci.conditioned_ceiling_coverage.c76_representation_association_orbit \
  analyze
