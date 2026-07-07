#!/bin/bash
# Project A Step 16 — benefit anatomy + sequential label-acquisition frontier (CPU only; reuses the
# Step-13 raw diagnostics grids). (0) regenerate the Step-15 digests with the renamed
# best_label_based_attempt field; (1) oracle-only benefit anatomy; (2) sequential harm-control over
# label budgets; (3) policy frontier; (4) Step-16 dashboard. No training, no GPU, no new datasets.
# NO --time.
#SBATCH --job-name=projA_step16_science
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_step16_science_%j.out
#SBATCH --error=logs/project_A_step16_science_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SUM=notes/project_A_observability/results_summaries
R1=notes/project_A_observability/results/step13_BNCI2014_001_diagnostics
R4=notes/project_A_observability/results/step13_BNCI2014_004_diagnostics

# (0) regenerate Step-15 digests with the renamed best_label_based_attempt field
conda run -n icml python -m h2cmi.observability.harm_control \
  --roots "$R1" "$R4" --ks 0 1 2 4 8 16 32 64 128 256 --repeats 500 --taus 0.0 0.01 0.02 0.05 \
  --ci-method normal \
  --out-json "$SUM/step15_harm_control_summary.json" --out-md "$SUM/step15_harm_control_summary.md"
conda run -n icml python -m h2cmi.observability.science_dashboard \
  --harm-table "$SUM/step13_harm_attribution_table.json" \
  --harm-predictor "$SUM/step14_harm_predictor_summary.json" \
  --real-minimal-labels "$SUM/step14_real_minimal_label_curves.json" \
  --harm-control "$SUM/step15_harm_control_summary.json" \
  --multidataset "$SUM/step10_moabb_multidataset_summary.json" --step-label "Step 15" \
  --out-json "$SUM/step15_science_dashboard.json" --out-md "$SUM/step15_science_dashboard.md"

# (1) oracle-only benefit anatomy
conda run -n icml python -m h2cmi.observability.benefit_anatomy \
  --roots "$R1" "$R4" \
  --out-json "$SUM/step16_benefit_anatomy.json" --out-md "$SUM/step16_benefit_anatomy.md"

# (2) sequential label-acquisition harm-control
conda run -n icml python -m h2cmi.observability.sequential_harm_control \
  --roots "$R1" "$R4" --budgets 8 16 32 64 128 256 512 full --batch-size 8 --repeats 500 \
  --taus 0.0 0.01 0.02 0.05 \
  --out-json "$SUM/step16_sequential_harm_control.json" --out-md "$SUM/step16_sequential_harm_control.md"

# (3) policy frontier
conda run -n icml python -m h2cmi.observability.policy_frontier \
  --static "$SUM/step15_harm_control_summary.json" \
  --sequential "$SUM/step16_sequential_harm_control.json" \
  --out-json "$SUM/step16_policy_frontier.json" --out-md "$SUM/step16_policy_frontier.md"

# (4) Step-16 dashboard
conda run -n icml python -m h2cmi.observability.science_dashboard \
  --harm-table "$SUM/step13_harm_attribution_table.json" \
  --harm-predictor "$SUM/step14_harm_predictor_summary.json" \
  --real-minimal-labels "$SUM/step14_real_minimal_label_curves.json" \
  --harm-control "$SUM/step15_harm_control_summary.json" \
  --benefit-anatomy "$SUM/step16_benefit_anatomy.json" \
  --sequential-harm-control "$SUM/step16_sequential_harm_control.json" \
  --policy-frontier "$SUM/step16_policy_frontier.json" \
  --multidataset "$SUM/step10_moabb_multidataset_summary.json" --step-label "Step 16" \
  --out-json "$SUM/step16_science_dashboard.json" --out-md "$SUM/step16_science_dashboard.md"

echo "step16 science pipeline done -> $SUM/step16_* (+ step15 digests renamed)"
