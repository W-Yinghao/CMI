#!/usr/bin/env bash
#SBATCH --job-name=cedar01f-inventory
#SBATCH --output=logs/cedar01f_inventory_%j.out
#SBATCH --error=logs/cedar01f_inventory_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd /home/infres/yinwang/CMI_AAAI

RUN_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="results/cedar/feature_inventory/${RUN_ID}"
mkdir -p "${OUT_DIR}"

PYTHONPATH=. python -m cedar_eeg.data.feature_inventory \
  --root results \
  --root archive \
  --include-archive \
  --csv-out "${OUT_DIR}/feature_inventory.csv" \
  --json-out "${OUT_DIR}/feature_inventory.json"
