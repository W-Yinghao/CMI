#!/bin/bash
# Project A Step 20 — final observability-contract closeout and claim ledger (CPU only; reads the tracked
# step-13..19 digests, assembles the evidence ledger + deployment decision ladder + forbidden-claim flags
# + final verdict). No training, no GPU, no new datasets, no retraining, no new rescue policy. NO --time.
#SBATCH --job-name=projA_step20_closeout
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/project_A_step20_closeout_%j.out
#SBATCH --error=logs/project_A_step20_closeout_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
SUM=notes/project_A_observability/results_summaries

conda run -n icml python -m h2cmi.observability.closeout \
  --summaries "$SUM" \
  --out-json "$SUM/step20_closeout.json" --out-md "$SUM/step20_closeout.md"

echo "step20 closeout done -> $SUM/step20_closeout.{json,md}"
