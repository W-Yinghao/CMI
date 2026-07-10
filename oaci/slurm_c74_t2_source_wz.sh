#!/usr/bin/env bash
#SBATCH --job-name=c74-wz
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz/logs/%x_%A_%a.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz/logs/%x_%A_%a.err

set -euo pipefail

STAGE="${1:?usage: sbatch --array=1-9 oaci/slurm_c74_t2_source_wz.sh P0_pilot|P1_expansion}"
case "${STAGE}" in
  P0_pilot|P1_expansion) ;;
  *) echo "invalid C74 stage: ${STAGE}" >&2; exit 2 ;;
esac

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "C74 instrumentation requires a Slurm array target ID" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p /projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz/logs
cd /home/infres/yinwang/CMI_AAAI_oaci

/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python \
  -m oaci.conditioned_ceiling_coverage.c74_t2_source_wz_instrumentation \
  instrument \
  --stage "${STAGE}" \
  --target-id "${SLURM_ARRAY_TASK_ID}" \
  --num-threads "${SLURM_CPUS_PER_TASK}" \
  --authorization-token C74_T2_SOURCE_WZ_REINFERENCE_AUTHORIZED
