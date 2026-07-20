#!/usr/bin/env bash
#SBATCH --job-name=c78r-protocol-test
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_oaci
export PYTHONPYCACHEPREFIX="/tmp/c78r-protocol-test-pycache-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m pytest -q \
  --basetemp="/tmp/c78r-protocol-test-${SLURM_JOB_ID}" \
  -o "cache_dir=/tmp/c78r-protocol-test-cache-${SLURM_JOB_ID}" \
  oaci/tests/test_c78r_seed3_SRC_canary.py
/home/infres/yinwang/anaconda3/envs/icml/bin/python \
  -m oaci.conditioned_ceiling_coverage.c78r_protocol_red_team
