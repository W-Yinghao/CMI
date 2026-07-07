#!/bin/bash
# Project A Step 14 — metric-semantics + power pipeline (CPU only; reuses the Step-13 raw diagnostics
# grids and tracked tables). (0) re-emits the Step-13 harm-predictor digest with the correct step
# label (fixes the Step-12 provenance bug, n_perm 100 preserves values); (1) Step-14 harm predictor
# with n_perm 1000 + p90/p95/p99 + robust margin; (2) harm power; (3) coverage-decomposed real
# minimal-label curves; (4) Step-14 dashboard. No training, no GPU, no new datasets. NO --time.
#SBATCH --job-name=projA_step14_science
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_step14_science_%j.out
#SBATCH --error=logs/project_A_step14_science_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SUM=notes/project_A_observability/results_summaries
R1=notes/project_A_observability/results/step13_BNCI2014_001_diagnostics
R4=notes/project_A_observability/results/step13_BNCI2014_004_diagnostics
HT=$SUM/step13_harm_attribution_table.json

# (0) provenance fix: re-emit Step-13 harm predictor with the correct step label (values preserved)
conda run -n icml python -m h2cmi.observability.harm_predictor \
  --input "$HT" --step-label "Step 13" --n-perm 100 \
  --out-json "$SUM/step13_harm_predictor_summary.json" \
  --out-md   "$SUM/step13_harm_predictor_summary.md"

# (1) Step-14 harm predictor: larger permutation null + robustness margin
conda run -n icml python -m h2cmi.observability.harm_predictor \
  --input "$HT" --step-label "Step 14" --n-perm 1000 --perm-seed 0 --robust-margin 0.03 \
  --out-json "$SUM/step14_harm_predictor_summary.json" \
  --out-md   "$SUM/step14_harm_predictor_summary.md"

# (2) power / sensitivity
conda run -n icml python -m h2cmi.observability.harm_power \
  --harm-table "$HT" --harm-predictor "$SUM/step14_harm_predictor_summary.json" \
  --out-json "$SUM/step14_harm_power_summary.json" \
  --out-md   "$SUM/step14_harm_power_summary.md"

# (3) coverage-decomposed real minimal-label curves
conda run -n icml python -m h2cmi.observability.real_minimal_labels \
  --roots "$R1" "$R4" --ks 0 1 2 4 8 16 32 64 128 256 --repeats 200 --step-label "Step 14" \
  --out-json "$SUM/step14_real_minimal_label_curves.json" \
  --out-md   "$SUM/step14_real_minimal_label_curves.md"

# (4) Step-14 dashboard
conda run -n icml python -m h2cmi.observability.science_dashboard \
  --harm-table "$HT" --harm-predictor "$SUM/step14_harm_predictor_summary.json" \
  --real-minimal-labels "$SUM/step14_real_minimal_label_curves.json" \
  --harm-power "$SUM/step14_harm_power_summary.json" \
  --step12-harm-predictor "$SUM/step12_harm_predictor_summary.json" \
  --multidataset "$SUM/step10_moabb_multidataset_summary.json" \
  --step-label "Step 14" \
  --out-json "$SUM/step14_science_dashboard.json" \
  --out-md   "$SUM/step14_science_dashboard.md"

echo "step14 science pipeline done -> $SUM/step14_* (+ step13 harm-predictor provenance fixed)"
