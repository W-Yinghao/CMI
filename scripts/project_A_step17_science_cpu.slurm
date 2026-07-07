#!/bin/bash
# Project A Step 17 — estimand-consistent harm control (accuracy vs balanced-accuracy) + class-balanced
# calibration policies (CPU only; reuses the Step-13 raw diagnostics grids). No training, no GPU, no new
# datasets, no retraining. (1) dual-estimand policy evaluator under iid + class-balanced (C13) sampling;
# (2) per-estimand harm-control frontier; (3) Step-17 dashboard with cross-estimand diagnostics.
# NO --time.
#SBATCH --job-name=projA_step17_science
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_step17_science_%j.out
#SBATCH --error=logs/project_A_step17_science_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SUM=notes/project_A_observability/results_summaries
R1=notes/project_A_observability/results/step13_BNCI2014_001_diagnostics
R4=notes/project_A_observability/results/step13_BNCI2014_004_diagnostics

# (1) dual-estimand policy evaluator (iid + class-balanced C13); bAcc abstains on missing/under-budget class
conda run -n icml python -m h2cmi.observability.estimand_consistency \
  --roots "$R1" "$R4" --ks 0 1 2 4 8 16 32 64 128 256 full --repeats 500 \
  --taus 0.0 0.01 0.02 0.05 --sampling iid class_balanced --seed 0 \
  --out-json "$SUM/step17_estimand_consistency.json" --out-md "$SUM/step17_estimand_consistency.md"

# (2) per-estimand harm-control frontier (accuracy vs balanced-accuracy, kept strictly separate)
conda run -n icml python -m h2cmi.observability.estimand_frontier \
  --consistency "$SUM/step17_estimand_consistency.json" \
  --out-json "$SUM/step17_estimand_frontier.json" --out-md "$SUM/step17_estimand_frontier.md"

# (3) Step-17 dashboard (cross-estimand diagnostics + claim-boundary conjunction)
conda run -n icml python -m h2cmi.observability.science_dashboard \
  --estimand-consistency "$SUM/step17_estimand_consistency.json" \
  --estimand-frontier "$SUM/step17_estimand_frontier.json" --step-label "Step 17" \
  --out-json "$SUM/step17_science_dashboard.json" --out-md "$SUM/step17_science_dashboard.md"

echo "step17 science pipeline done -> $SUM/step17_*"
