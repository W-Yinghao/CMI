#!/usr/bin/env bash
#SBATCH --job-name=c78f-wave-gate
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=08:00:00
set -euo pipefail
wave="${1:?wave A or B required}"
cd /home/infres/yinwang/CMI_AAAI_oaci
export PYTHONPYCACHEPREFIX="/tmp/c78f-wave-${wave}-${SLURM_JOB_ID}/pycache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/infres/yinwang/anaconda3/envs/icml/bin/python \
  -m oaci.conditioned_ceiling_coverage.c78f_instrument validate-wave --wave "${wave}"
