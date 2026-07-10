#!/usr/bin/env bash
#SBATCH --job-name=c78-auth-test
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_oaci
export PYTHONPYCACHEPREFIX="/tmp/c78-auth-test-pycache-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m pytest -q \
  --basetemp="/tmp/c78-auth-test-${SLURM_JOB_ID}" \
  -o "cache_dir=/tmp/c78-auth-test-cache-${SLURM_JOB_ID}" \
  oaci/tests/test_c78_authorized_pilot.py
