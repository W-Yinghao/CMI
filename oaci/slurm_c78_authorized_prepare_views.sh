#!/usr/bin/env bash
#SBATCH --job-name=c78-auth-views
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail
token="${1:?exact CLI authorization token required}"
cd /home/infres/yinwang/CMI_AAAI_oaci
scratch="/tmp/c78-views-${SLURM_JOB_ID}"
mkdir -p "${scratch}" "${scratch}/mne" "${scratch}/cache" "${scratch}/mpl"
export TMPDIR="${scratch}"
export XDG_CACHE_HOME="${scratch}/cache"
export MPLCONFIGDIR="${scratch}/mpl"
export MNE_CACHE_DIR="${scratch}/mne"
export PYTHONPYCACHEPREFIX="${scratch}/pycache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/infres/yinwang/anaconda3/envs/icml/bin/python \
  -m oaci.conditioned_ceiling_coverage.c78_authorized_instrument prepare-views \
  --authorization-token "${token}"
