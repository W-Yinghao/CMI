#!/bin/bash
# Project A — expanded audited BNCI2014_001 grid (subjects 1-9 x seeds 0-2 = 27 cells) on ONE GPU,
# then validate + emit the TRACKED summary digest. PRIMARY submission. SLURM schedules whichever
# GPU partition is free among A100/H100/V100/A40. Not a SOTA / full-table run — an audited
# claim-boundary + stability validation. NO --time (cluster convention).
#SBATCH --job-name=projA_bnci_exp
#SBATCH --partition=A100,H100,V100,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/project_A_bnci_exp_%j.out
#SBATCH --error=logs/project_A_bnci_exp_%j.err

set -euo pipefail
mkdir -p logs
ROOT=notes/project_A_observability/results/step9_bnci2014_001_expanded
SUMMARY=notes/project_A_observability/results_summaries

# grid (idempotent: --resume only fills missing/failed cells)
conda run -n icml python -m h2cmi.run_real_audited_grid \
  --dataset BNCI2014_001 \
  --subjects 1 2 3 4 5 6 7 8 9 \
  --target-subjects 1 2 3 4 5 6 7 8 9 \
  --seeds 0 1 2 \
  --epochs 50 \
  --fast \
  --align-factor subject \
  --device cuda \
  --root-outdir "$ROOT" \
  --resume

# validate every run's claim boundary + write the tracked (committable) digest (single job -> no race)
conda run -n icml python -m h2cmi.observability.validate_results \
  --root "$ROOT" \
  --step "Step 9" --dataset BNCI2014_001 \
  --expected-targets 1 2 3 4 5 6 7 8 9 \
  --expected-seeds 0 1 2 \
  --out-json "$SUMMARY/step9_bnci2014_001_expanded_summary.json" \
  --out-md   "$SUMMARY/step9_bnci2014_001_expanded_summary.md"

echo "expanded grid + digest done -> $SUMMARY/step9_bnci2014_001_expanded_summary.{json,md}"
