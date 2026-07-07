#!/bin/bash
# Project A Step 18 — TTA harm mechanisms + deployment-prior stress (CPU only; reuses the Step-13 raw
# per-trial predictions). No training, no GPU, no new datasets, no retraining. (1) harm-mechanism
# decomposition (gain/loss + per-class recall deltas + transitions); (2) counterfactual deployment-prior
# stress (contract C14); (3) Step-18 dashboard. NO --time.
#SBATCH --job-name=projA_step18_science
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_step18_science_%j.out
#SBATCH --error=logs/project_A_step18_science_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SUM=notes/project_A_observability/results_summaries
R1=notes/project_A_observability/results/step13_BNCI2014_001_diagnostics
R4=notes/project_A_observability/results/step13_BNCI2014_004_diagnostics

# (1) harm-mechanism decomposition (oracle/evaluation-only)
conda run -n icml python -m h2cmi.observability.harm_mechanisms \
  --roots "$R1" "$R4" \
  --out-json "$SUM/step18_harm_mechanisms.json" --out-md "$SUM/step18_harm_mechanisms.md"

# (2) counterfactual deployment-prior stress (declared priors; contract C14)
conda run -n icml python -m h2cmi.observability.prior_stress \
  --harm-mechanisms "$SUM/step18_harm_mechanisms.json" \
  --out-json "$SUM/step18_prior_stress.json" --out-md "$SUM/step18_prior_stress.md"

# (3) Step-18 dashboard
conda run -n icml python -m h2cmi.observability.science_dashboard \
  --harm-mechanisms "$SUM/step18_harm_mechanisms.json" \
  --prior-stress "$SUM/step18_prior_stress.json" --step-label "Step 18" \
  --out-json "$SUM/step18_science_dashboard.json" --out-md "$SUM/step18_science_dashboard.md"

echo "step18 science pipeline done -> $SUM/step18_*"
