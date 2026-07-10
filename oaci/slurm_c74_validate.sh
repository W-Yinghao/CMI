#!/usr/bin/env bash
#SBATCH --job-name=c74-validate
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz/logs/%x_%j.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz/logs/%x_%j.err

set -euo pipefail

STAGE="${1:?usage: sbatch oaci/slurm_c74_validate.sh P0_pilot|P1_expansion}"
case "${STAGE}" in
  P0_pilot|P1_expansion) ;;
  *) echo "invalid C74 stage: ${STAGE}" >&2; exit 2 ;;
esac

mkdir -p /projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz/logs
cd /home/infres/yinwang/CMI_AAAI_oaci

/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python \
  -m oaci.conditioned_ceiling_coverage.c74_t2_source_wz_instrumentation \
  validate \
  --stage "${STAGE}"
