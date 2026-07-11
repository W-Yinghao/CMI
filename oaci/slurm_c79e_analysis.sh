#!/usr/bin/env bash
#SBATCH --job-name=c79e-analysis
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_oaci
export PYTHONPYCACHEPREFIX="/tmp/c79e-analysis-${SLURM_JOB_ID}/pycache"
export JOBLIB_TEMP_FOLDER="/tmp/c79e-analysis-${SLURM_JOB_ID}/joblib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${JOBLIB_TEMP_FOLDER}"
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m \
  oaci.conditioned_ceiling_coverage.c79e_seed4_replication run-analysis
