#!/bin/bash
# Project A — audited BNCI2014_001 mini-grid (targets 1/2/3 x seeds 0/1/2), then validate +
# emit the TRACKED summary digest. Not a SOTA / full-table run — an audited claim-boundary
# validation. Convention matches the repo (partition=V100, gpu:1, NO --time/walltime).
#SBATCH --job-name=projA_bnci_minigrid
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/project_A_bnci_minigrid_%j.out
#SBATCH --error=logs/project_A_bnci_minigrid_%j.err

set -euo pipefail
mkdir -p logs

ROOT=notes/project_A_observability/results/step8_bnci2014_001_minigrid
SUMMARY=notes/project_A_observability/results_summaries

# GPU if visible, else CPU with fewer epochs (the runner graceful-skips a missing cache).
DEVICE=cuda; EPOCHS=20
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  || { DEVICE=cpu; EPOCHS=5; }

conda run -n icml python -m h2cmi.run_real_audited_grid \
  --dataset BNCI2014_001 \
  --subjects 1 2 3 \
  --target-subjects 1 2 3 \
  --seeds 0 1 2 \
  --epochs "$EPOCHS" \
  --fast \
  --align-factor subject \
  --device "$DEVICE" \
  --root-outdir "$ROOT"

# validate every run's claim boundary + write the tracked (committable) digest
conda run -n icml python -m h2cmi.observability.validate_results \
  --root "$ROOT" \
  --step "Step 8" --dataset BNCI2014_001 \
  --out-json "$SUMMARY/step8_bnci2014_001_minigrid_summary.json" \
  --out-md   "$SUMMARY/step8_bnci2014_001_minigrid_summary.md"

echo "mini-grid + digest done -> $SUMMARY/step8_bnci2014_001_minigrid_summary.{json,md}"
