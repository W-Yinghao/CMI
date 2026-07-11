#!/usr/bin/env bash
#SBATCH --job-name=c79e-field
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
action="${1:?train-oaci-erm|train-src required}"
target="${2:?target required}"
cd /home/infres/yinwang/CMI_AAAI_oaci
scratch="/tmp/c79e-${action}-${target}-${SLURM_JOB_ID}"
mkdir -p "${scratch}/mne" "${scratch}/cache" "${scratch}/mpl"
export TMPDIR="${scratch}"
export XDG_CACHE_HOME="${scratch}/cache"
export MPLCONFIGDIR="${scratch}/mpl"
export MNE_CACHE_DIR="${scratch}/mne"
export PYTHONPYCACHEPREFIX="${scratch}/pycache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m \
  oaci.conditioned_ceiling_coverage.c79e_seed4_replication "${action}" --target "${target}"

