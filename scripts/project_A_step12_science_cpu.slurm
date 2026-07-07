#!/bin/bash
# Project A Step 12 — science pipeline (CPU): harm attribution -> retrospective harm predictor ->
# minimal-paired phase transition -> science dashboard. Reads the gitignored Step 9 / Step 10 raw run
# dirs and the tracked multi-dataset digest; writes tracked step12_* summaries. No training, no GPU.
# NO --time (cluster convention — every prior Project-A job omits it).
#SBATCH --job-name=projA_step12_science
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_step12_science_%j.out
#SBATCH --error=logs/project_A_step12_science_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SUM=notes/project_A_observability/results_summaries

conda run -n icml python -m h2cmi.observability.harm_attribution \
  --roots \
    notes/project_A_observability/results/step9_bnci2014_001_expanded \
    notes/project_A_observability/results/step10_BNCI2014_004 \
  --out-json "$SUM/step12_harm_attribution_table.json" \
  --out-md   "$SUM/step12_harm_attribution_table.md"

conda run -n icml python -m h2cmi.observability.harm_predictor \
  --input "$SUM/step12_harm_attribution_table.json" \
  --out-json "$SUM/step12_harm_predictor_summary.json" \
  --out-md   "$SUM/step12_harm_predictor_summary.md"

conda run -n icml python notes/project_A_observability/science/run_minimal_paired_phase_transition.py

conda run -n icml python -m h2cmi.observability.science_dashboard \
  --harm-table "$SUM/step12_harm_attribution_table.json" \
  --harm-predictor "$SUM/step12_harm_predictor_summary.json" \
  --phase-transition "$SUM/step12_minimal_paired_phase_transition.json" \
  --multidataset "$SUM/step10_moabb_multidataset_summary.json" \
  --out-json "$SUM/step12_science_dashboard.json" \
  --out-md   "$SUM/step12_science_dashboard.md"

echo "step12 science pipeline done -> $SUM/step12_*"
