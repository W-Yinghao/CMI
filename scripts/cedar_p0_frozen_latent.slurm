#!/usr/bin/env bash
#SBATCH --job-name=cedar-p0
#SBATCH --output=logs/cedar_p0_%j.out
#SBATCH --error=logs/cedar_p0_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

FEATURE_NPZ="${FEATURE_NPZ:?set FEATURE_NPZ to a saved frozen-feature .npz}"
OUT="${OUT:-results/cedar_p0/p0_${SLURM_JOB_ID}.json}"
PROBE="${PROBE:-linear}"
N_SPLITS="${N_SPLITS:-3}"
SEED="${SEED:-0}"

python -m cedar_eeg.runners.run_p0_frozen_latent \
  --feature-npz "$FEATURE_NPZ" \
  --out "$OUT" \
  --probe "$PROBE" \
  --n-splits "$N_SPLITS" \
  --seed "$SEED"
