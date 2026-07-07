#!/bin/bash
# Project A Step 19 — prior-uncertainty robustness frontier (CPU only; reuses the Step-18 class-wise
# recall deltas). No training, no GPU, no new datasets, no retraining. (1) exact L1-ball robust gain
# bounds + sign-flip radii; (2) prior-robust adaptation policy (adapt/block/abstain); (3) Step-19
# dashboard. NO --time.
#SBATCH --job-name=projA_step19_science
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_step19_science_%j.out
#SBATCH --error=logs/project_A_step19_science_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SUM=notes/project_A_observability/results_summaries

# (1) prior-uncertainty robustness frontier (exact L1-ball bounds + flip radii)
conda run -n icml python -m h2cmi.observability.prior_uncertainty \
  --prior-stress "$SUM/step18_prior_stress.json" \
  --rhos 0.0 0.05 0.10 0.20 0.30 0.50 1.0 2.0 --taus 0.0 0.01 0.02 0.05 \
  --out-json "$SUM/step19_prior_uncertainty_frontier.json" \
  --out-md "$SUM/step19_prior_uncertainty_frontier.md"

# (2) prior-robust adaptation policy (worst-case adapt/block/abstain over declared L1 sets)
conda run -n icml python -m h2cmi.observability.prior_robust_policy \
  --prior-uncertainty "$SUM/step19_prior_uncertainty_frontier.json" \
  --harm-thresholds 0.05 0.10 0.20 --rhos 0.05 0.10 0.20 0.50 \
  --out-json "$SUM/step19_prior_robust_policy.json" --out-md "$SUM/step19_prior_robust_policy.md"

# (3) Step-19 dashboard
conda run -n icml python -m h2cmi.observability.science_dashboard \
  --prior-uncertainty "$SUM/step19_prior_uncertainty_frontier.json" \
  --prior-robust-policy "$SUM/step19_prior_robust_policy.json" --step-label "Step 19" \
  --out-json "$SUM/step19_science_dashboard.json" --out-md "$SUM/step19_science_dashboard.md"

echo "step19 science pipeline done -> $SUM/step19_*"
