#!/usr/bin/env bash
#SBATCH --job-name=c78f-src
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
target="${1:?target required}"
cd /home/infres/yinwang/CMI_AAAI_oaci
scratch="/tmp/c78f-src-${target}-${SLURM_JOB_ID}"
mkdir -p "${scratch}/mne" "${scratch}/cache" "${scratch}/mpl"
export TMPDIR="${scratch}"
export XDG_CACHE_HOME="${scratch}/cache"
export MPLCONFIGDIR="${scratch}/mpl"
export MNE_CACHE_DIR="${scratch}/mne"
export PYTHONPYCACHEPREFIX="${scratch}/pycache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
/home/infres/yinwang/anaconda3/envs/icml/bin/python \
  -m oaci.conditioned_ceiling_coverage.c78f_train src --target "${target}"
