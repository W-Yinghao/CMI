#!/usr/bin/env bash
#SBATCH --job-name=c78f-instrument
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
set -euo pipefail
target="${1:?target required}"
cd /home/infres/yinwang/CMI_AAAI_oaci
scratch="/tmp/c78f-instrument-${target}-${SLURM_JOB_ID}"
mkdir -p "${scratch}/mne" "${scratch}/cache" "${scratch}/mpl"
export TMPDIR="${scratch}"
export XDG_CACHE_HOME="${scratch}/cache"
export MPLCONFIGDIR="${scratch}/mpl"
export MNE_CACHE_DIR="${scratch}/mne"
export PYTHONPYCACHEPREFIX="${scratch}/pycache"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
/home/infres/yinwang/anaconda3/envs/icml/bin/python \
  -m oaci.conditioned_ceiling_coverage.c78f_instrument instrument-target \
  --target "${target}" --workers 4 --threads-per-worker 12
