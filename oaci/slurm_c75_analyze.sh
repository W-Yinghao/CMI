#!/usr/bin/env bash
#SBATCH --job-name=c75-analyze
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c75-representation-construct/logs/%x_%j.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c75-representation-construct/logs/%x_%j.err

set -euo pipefail

# The analysis parallelizes blocked-null replicates across processes. Keep
# linear-algebra libraries single-threaded to avoid 48x nested oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/tmp/c75-pycache-${SLURM_JOB_ID}

mkdir -p /projects/EEG-foundation-model/yinghao/oaci-c75-representation-construct/logs
cd /home/infres/yinwang/CMI_AAAI_oaci

/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python \
  -m oaci.conditioned_ceiling_coverage.c75_representation_construct_validity \
  analyze
