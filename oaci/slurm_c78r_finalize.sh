#!/usr/bin/env bash
#SBATCH --job-name=c78r-finalize
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_oaci
export PYTHONPYCACHEPREFIX="/tmp/c78r-finalize-pycache-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/infres/yinwang/anaconda3/envs/icml/bin/python \
  -m oaci.conditioned_ceiling_coverage.c78r_finalize
