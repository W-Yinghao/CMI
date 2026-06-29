#!/bin/bash
#SBATCH --job-name=cigl-phase3a-bnci001
#SBATCH --partition=A100,V100,V100-32GB,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/cigl/%x-%j.out
#SBATCH --error=logs/cigl/%x-%j.err
#
# CIGL Phase 3A — regularizer-effect PILOT on BNCI2014_001 fold-0 (source-only). Exploratory:
# NO full LOSO, NO SEED/DEAP, NO SOTA, NO target data in training/selection.
#
# Submission convention (per PI): multi-partition list, let SLURM schedule on whichever frees first;
# default QOS (no --qos); --time omitted to inherit the partition MAX. Just:
#     sbatch scripts/sbatch_cigl_phase3a_bnci001.sh
#
# Runs from the CIGL worktree checked out on project/cigl-phase3a-regularizer-pilot.
set -euo pipefail

WORKTREE=/home/infres/yinwang/CMI_AAAI_cigl
cd "$WORKTREE"
mkdir -p logs/cigl results/cigl/phase3a_pilot

PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
if [ ! -x "$PY" ]; then
    echo "ERROR: eeg2025 python not found at $PY — adjust PY or create the env." >&2
    exit 2
fi

export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  branch=$(git -C "$WORKTREE" rev-parse --abbrev-ref HEAD)  commit=$(git -C "$WORKTREE" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)" \
    || { echo "ERROR: CUDA not available on this node." >&2; exit 3; }

# 7 configs x 3 seeds at n_perm=20, then ERM/full_cigl/best-Pareto re-audited at n_perm=50.
# Do NOT reduce configs/seeds/epochs/probe_epochs/n_perm.
srun "$PY" scripts/run_cigl_phase3a_regularizer_pilot.py \
    --dataset BNCI2014_001 \
    --device cuda \
    --fold 0 \
    --seeds 0 1 2 \
    --n_perm 20 \
    --n_perm_confirm 50 \
    --epochs 80 \
    --probe_epochs 100 \
    --gate_alpha 0.05

echo "DONE. Per-config/seed + summary JSON under results/cigl/phase3a_pilot/ (gitignored); build the"
echo "tracked reviewer summary docs/CIGL_15_PHASE3A_RESULTS_BNCI2014_001.md from them."
