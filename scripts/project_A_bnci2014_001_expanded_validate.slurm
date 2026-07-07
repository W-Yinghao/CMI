#!/bin/bash
# Project A — validate the expanded BNCI2014_001 grid + emit the TRACKED summary digest.
# Run this ONCE after the array job (project_A_bnci2014_001_expanded_array.slurm) completes, so
# no array task races to write the summary. CPU-only; no GPU needed. NO --time (cluster convention).
#SBATCH --job-name=projA_bnci_exp_val
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/project_A_bnci_exp_validate_%j.out
#SBATCH --error=logs/project_A_bnci_exp_validate_%j.err

set -euo pipefail
mkdir -p logs

ROOT=notes/project_A_observability/results/step9_bnci2014_001_expanded
SUMMARY=notes/project_A_observability/results_summaries

conda run -n icml python -m h2cmi.observability.validate_results \
  --root "$ROOT" \
  --step "Step 9" --dataset BNCI2014_001 \
  --expected-targets 1 2 3 4 5 6 7 8 9 \
  --expected-seeds 0 1 2 \
  --out-json "$SUMMARY/step9_bnci2014_001_expanded_summary.json" \
  --out-md   "$SUMMARY/step9_bnci2014_001_expanded_summary.md"

echo "validation + digest done -> $SUMMARY/step9_bnci2014_001_expanded_summary.{json,md}"
