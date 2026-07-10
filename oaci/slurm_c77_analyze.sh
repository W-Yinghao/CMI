#!/usr/bin/env bash
#SBATCH --job-name=c77-analyze
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs/%x_%j.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/tmp/c77-pycache-${SLURM_JOB_ID}
mkdir -p /projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs
cd /home/infres/yinwang/CMI_AAAI_oaci
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python \
  -m oaci.conditioned_ceiling_coverage.c77_independent_multiregime_replication_protocol analyze
