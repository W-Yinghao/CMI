#!/bin/bash
# Project A — audited real-EEG Tier-2 pilot (ONE dataset, ONE target subject, ONE seed).
# Not a SOTA / full-table run: it validates the data interface and the audited claim boundary.
# Convention matches the repo's other scripts (partition=V100, gpu:1, NO --time/walltime).
#SBATCH --job-name=projA_audit_pilot
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/projA_audit_pilot_%j.out
#SBATCH --error=logs/projA_audit_pilot_%j.err

set -euo pipefail
mkdir -p logs

# One dataset, one held-out target subject, one seed, few epochs. Falls back to --device cpu
# if no GPU is visible. The runner writes a legal SKIP artifact if the MOABB cache is missing.
DEVICE=cuda
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || DEVICE=cpu

conda run -n icml python -m h2cmi.run_real_audited \
  --dataset BNCI2014_001 \
  --subjects 1 2 3 \
  --target-subject 1 \
  --n-classes 4 \
  --epochs 2 \
  --n-perm 20 \
  --fast \
  --device "$DEVICE" \
  --seed 0 \
  --outdir notes/project_A_observability/results/slurm_bnci2014_001_s1_seed0

echo "audited pilot done -> notes/project_A_observability/results/slurm_bnci2014_001_s1_seed0"
