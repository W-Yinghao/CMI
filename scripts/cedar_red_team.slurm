#!/usr/bin/env bash
#SBATCH --job-name=cedar-redteam
#SBATCH --output=logs/cedar_redteam_%j.out
#SBATCH --error=logs/cedar_redteam_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

P0_JSON="${P0_JSON:?set P0_JSON to a CEDAR P0 result JSON}"
OUT="${OUT:-results/cedar_p0/red_team_${SLURM_JOB_ID}.json}"

python -m cedar_eeg.runners.run_red_team \
  --p0-json "$P0_JSON" \
  --out "$OUT"
