#!/bin/bash
# Project A Step 10 — validate each dataset grid + emit tracked per-dataset digests, then COMBINE
# them into one chance-normalized multi-dataset digest. Run ONCE, after the GPU array finishes
# (submit with --dependency=afterok:<arrayid> so no task races the summaries). CPU-only. Also
# regenerates the Step-9 4-class reference digest under the new normalized schema (its raw run dirs
# persist in results/). NO --time (cluster convention).
#SBATCH --job-name=projA_step10_validate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/project_A_step10_validate_%j.out
#SBATCH --error=logs/project_A_step10_validate_%j.err

set -euo pipefail
mkdir -p logs
SUMMARY=notes/project_A_observability/results_summaries

# (1) regenerate Step-9 reference digest with n_classes + chance-normalized fields (schema upgrade)
conda run -n icml python -m h2cmi.observability.validate_results \
  --root notes/project_A_observability/results/step9_bnci2014_001_expanded \
  --step "Step 9" --dataset BNCI2014_001 \
  --expected-targets 1 2 3 4 5 6 7 8 9 --expected-seeds 0 1 2 \
  --out-json "$SUMMARY/step9_bnci2014_001_expanded_summary.json" \
  --out-md   "$SUMMARY/step9_bnci2014_001_expanded_summary.md"

# (2) validate each Step-10 dataset (grid_manifest.json in each root drives expected_cells)
for DATASET in BNCI2014_004 BNCI2015_001; do
  SLUG=$(echo "$DATASET" | tr 'A-Z' 'a-z')
  conda run -n icml python -m h2cmi.observability.validate_results \
    --root "notes/project_A_observability/results/step10_${DATASET}" \
    --step "Step 10" --dataset "$DATASET" \
    --out-json "$SUMMARY/step10_${SLUG}_summary.json" \
    --out-md   "$SUMMARY/step10_${SLUG}_summary.md"
done

# (3) combine into the chance-normalized multi-dataset digest (raw bAcc never pooled across K)
conda run -n icml python -m h2cmi.observability.combine_summaries \
  --inputs \
    "$SUMMARY/step9_bnci2014_001_expanded_summary.json" \
    "$SUMMARY/step10_bnci2014_004_summary.json" \
    "$SUMMARY/step10_bnci2015_001_summary.json" \
  --out-json "$SUMMARY/step10_moabb_multidataset_summary.json" \
  --out-md   "$SUMMARY/step10_moabb_multidataset_summary.md"

echo "step10 validate + combine done -> $SUMMARY/step10_moabb_multidataset_summary.{json,md}"
