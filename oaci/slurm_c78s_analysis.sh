#!/usr/bin/env bash
#SBATCH --job-name=c78s-analysis
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI/c78f_oaci/oaci/logs/c78s-analysis-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI/c78f_oaci/oaci/logs/c78s-analysis-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI/c78f_oaci
export PYTHONPYCACHEPREFIX="/tmp/c78s-analysis-${SLURM_JOB_ID}/pycache"
export JOBLIB_TEMP_FOLDER="/tmp/c78s-analysis-${SLURM_JOB_ID}/joblib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p oaci/logs "${JOBLIB_TEMP_FOLDER}"
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m \
  oaci.conditioned_ceiling_coverage.c78s_seed3_scientific_analysis run
