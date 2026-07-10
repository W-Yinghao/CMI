#!/usr/bin/env bash
#SBATCH --job-name=c77-synthetic
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7%2
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs/%x_%A_%a.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/tmp/c77-pycache-${SLURM_JOB_ID}
mkdir -p /projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs
cd /home/infres/yinwang/CMI_AAAI_oaci
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python \
  -m oaci.conditioned_ceiling_coverage.synthetic_multiregime_generator \
  shard --index "${SLURM_ARRAY_TASK_ID}" --count 8 --workers 48
