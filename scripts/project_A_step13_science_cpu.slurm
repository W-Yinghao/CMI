#!/bin/bash
# Project A Step 13 — science pipeline (CPU) over the instrumented diagnostics grids: harm attribution
# (rich R1) -> retrospective harm predictor -> real minimal-label curves -> Step-13 dashboard (vs the
# Step-12 harm predictor). Reads gitignored step13_* raw dirs + tracked digests; writes tracked
# step13_* summaries. Run AFTER the diagnostics GPU array (afterok). No GPU. NO --time.
#SBATCH --job-name=projA_step13_science
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_step13_science_%j.out
#SBATCH --error=logs/project_A_step13_science_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SUM=notes/project_A_observability/results_summaries
R1=notes/project_A_observability/results/step13_BNCI2014_001_diagnostics
R4=notes/project_A_observability/results/step13_BNCI2014_004_diagnostics

conda run -n icml python -m h2cmi.observability.harm_attribution \
  --roots "$R1" "$R4" \
  --out-json "$SUM/step13_harm_attribution_table.json" \
  --out-md   "$SUM/step13_harm_attribution_table.md"

conda run -n icml python -m h2cmi.observability.harm_predictor \
  --input "$SUM/step13_harm_attribution_table.json" \
  --out-json "$SUM/step13_harm_predictor_summary.json" \
  --out-md   "$SUM/step13_harm_predictor_summary.md"

conda run -n icml python -m h2cmi.observability.real_minimal_labels \
  --roots "$R1" "$R4" \
  --ks 0 1 2 4 8 16 32 64 128 256 \
  --repeats 200 \
  --out-json "$SUM/step13_real_minimal_label_curves.json" \
  --out-md   "$SUM/step13_real_minimal_label_curves.md"

conda run -n icml python -m h2cmi.observability.science_dashboard \
  --harm-table "$SUM/step13_harm_attribution_table.json" \
  --harm-predictor "$SUM/step13_harm_predictor_summary.json" \
  --real-minimal-labels "$SUM/step13_real_minimal_label_curves.json" \
  --step12-harm-predictor "$SUM/step12_harm_predictor_summary.json" \
  --multidataset "$SUM/step10_moabb_multidataset_summary.json" \
  --out-json "$SUM/step13_science_dashboard.json" \
  --out-md   "$SUM/step13_science_dashboard.md"

echo "step13 science pipeline done -> $SUM/step13_*"
