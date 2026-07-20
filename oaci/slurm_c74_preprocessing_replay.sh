#!/usr/bin/env bash
#SBATCH --job-name=c74-preproc
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz/logs/%x_%j.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz/logs/%x_%j.err

set -euo pipefail

MODE="${1:?usage: sbatch [--nodelist=nodecpu0X] oaci/slurm_c74_preprocessing_replay.sh capture nodecpu0X|compare}"
REPLICATE="${2:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MNE_DONTWRITE_HOME=true

cd /home/infres/yinwang/CMI_AAAI_oaci
if [[ "${MODE}" == "capture" ]]; then
  /home/infres/yinwang/anaconda3/envs/eeg2025/bin/python \
    -m oaci.conditioned_ceiling_coverage.c74_preprocessing_replay \
    capture \
    --replicate "${REPLICATE}" \
    --num-threads "${SLURM_CPUS_PER_TASK}" \
    --authorization-token C74_T2_SOURCE_WZ_REINFERENCE_AUTHORIZED
elif [[ "${MODE}" == "compare" ]]; then
  /home/infres/yinwang/anaconda3/envs/eeg2025/bin/python \
    -m oaci.conditioned_ceiling_coverage.c74_preprocessing_replay \
    compare \
    --num-threads "${SLURM_CPUS_PER_TASK}" \
    --authorization-token C74_T2_SOURCE_WZ_REINFERENCE_AUTHORIZED
else
  echo "invalid C74 preprocessing replay mode: ${MODE}" >&2
  exit 2
fi
