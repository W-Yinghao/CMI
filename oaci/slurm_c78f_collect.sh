#!/usr/bin/env bash
#SBATCH --job-name=c78f-collect
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
set -euo pipefail
action="${1:?freeze|labels|collect|finalize required}"
cd /home/infres/yinwang/CMI_AAAI_oaci
scratch="/tmp/c78f-${action}-${SLURM_JOB_ID}"
mkdir -p "${scratch}/mne" "${scratch}/cache" "${scratch}/mpl"
export TMPDIR="${scratch}"
export XDG_CACHE_HOME="${scratch}/cache"
export MPLCONFIGDIR="${scratch}/mpl"
export MNE_CACHE_DIR="${scratch}/mne"
export PYTHONPYCACHEPREFIX="${scratch}/pycache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
case "${action}" in
  freeze)
    module="oaci.conditioned_ceiling_coverage.c78f_instrument"
    args=(freeze-full-field)
    ;;
  labels)
    module="oaci.conditioned_ceiling_coverage.c78f_instrument"
    args=(prepare-all-label-views)
    ;;
  collect)
    module="oaci.conditioned_ceiling_coverage.c78f_collect"
    args=()
    ;;
  finalize)
    module="oaci.conditioned_ceiling_coverage.c78f_finalize"
    args=(finalize)
    ;;
  *)
    echo "unknown action: ${action}" >&2
    exit 2
    ;;
esac
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m "${module}" "${args[@]}"
