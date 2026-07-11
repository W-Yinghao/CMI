#!/usr/bin/env bash
#SBATCH --job-name=c78s-redteam
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI/c78f_oaci/oaci/logs/c78s-redteam-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI/c78f_oaci/oaci/logs/c78s-redteam-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI/c78f_oaci
export PYTHONPYCACHEPREFIX="/tmp/c78s-redteam-${SLURM_JOB_ID}/pycache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p oaci/logs
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m \
  oaci.conditioned_ceiling_coverage.c78s_red_team result
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m \
  oaci.conditioned_ceiling_coverage.c78s_seed3_scientific_analysis finalize
