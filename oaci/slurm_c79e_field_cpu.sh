#!/usr/bin/env bash
#SBATCH --job-name=c79e-field-cpu
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
set -euo pipefail
action="${1:?instrument-target|validate-wave|freeze-field|prepare-primary-label-views required}"
shift
cd /home/infres/yinwang/CMI_AAAI_oaci
scratch="/tmp/c79e-${action}-${SLURM_JOB_ID}"
mkdir -p "${scratch}/mne" "${scratch}/cache" "${scratch}/mpl"
export TMPDIR="${scratch}"
export XDG_CACHE_HOME="${scratch}/cache"
export MPLCONFIGDIR="${scratch}/mpl"
export MNE_CACHE_DIR="${scratch}/mne"
export PYTHONPYCACHEPREFIX="${scratch}/pycache"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m \
  oaci.conditioned_ceiling_coverage.c79e_seed4_replication "${action}" "$@"
