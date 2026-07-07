#!/bin/bash
# Project A Step 15 — coverage-aware harm-control policy pipeline (CPU only; reuses the Step-13 raw
# diagnostics grids with per-trial oracle predictions). (1) evaluate adapt/identity/abstain policies
# over k x tau; (2) Step-15 dashboard (best deployable policy by the predeclared rule). No training,
# no GPU, no new datasets. NO --time.
#SBATCH --job-name=projA_step15_science
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_step15_science_%j.out
#SBATCH --error=logs/project_A_step15_science_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SUM=notes/project_A_observability/results_summaries
R1=notes/project_A_observability/results/step13_BNCI2014_001_diagnostics
R4=notes/project_A_observability/results/step13_BNCI2014_004_diagnostics

conda run -n icml python -m h2cmi.observability.harm_control \
  --roots "$R1" "$R4" \
  --ks 0 1 2 4 8 16 32 64 128 256 \
  --repeats 500 \
  --taus 0.0 0.01 0.02 0.05 \
  --ci-method normal \
  --out-json "$SUM/step15_harm_control_summary.json" \
  --out-md   "$SUM/step15_harm_control_summary.md"

conda run -n icml python -m h2cmi.observability.science_dashboard \
  --harm-table "$SUM/step13_harm_attribution_table.json" \
  --harm-predictor "$SUM/step14_harm_predictor_summary.json" \
  --real-minimal-labels "$SUM/step14_real_minimal_label_curves.json" \
  --harm-control "$SUM/step15_harm_control_summary.json" \
  --multidataset "$SUM/step10_moabb_multidataset_summary.json" \
  --step-label "Step 15" \
  --out-json "$SUM/step15_science_dashboard.json" \
  --out-md   "$SUM/step15_science_dashboard.md"

echo "step15 harm-control pipeline done -> $SUM/step15_*"
