#!/bin/bash
#SBATCH --job-name=cigl-p3ar-bnci001
#SBATCH --partition=A100,V100,V100-32GB,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/cigl/%x-%j.out
#SBATCH --error=logs/cigl/%x-%j.err
#
# CIGL Phase 3A-R — baseline repair + gentle-lambda re-pilot on BNCI2014_001 fold-0 (source-only).
# Exploratory: NO full LOSO, NO SEED/DEAP, NO SOTA, NO target data in training/selection, NO lambda grid.
#
# Multi-partition, default QOS, --time omitted (inherit partition max). Just:
#     sbatch scripts/sbatch_cigl_phase3a_baseline_repair_bnci001.sh
# Runs from the CIGL worktree on project/cigl-phase3a-baseline-repair.
set -euo pipefail

WORKTREE=/home/infres/yinwang/CMI_AAAI_cigl
cd "$WORKTREE"
mkdir -p logs/cigl results/cigl/phase3a_baseline_repair

PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
if [ ! -x "$PY" ]; then
    echo "ERROR: eeg2025 python not found at $PY." >&2
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

# Part A (baseline adequacy) then conditional Part B (gentle micro-ladder). Do NOT reduce seeds/epochs/n_perm.
srun "$PY" scripts/run_cigl_phase3a_baseline_repair.py \
    --dataset BNCI2014_001 \
    --device cuda \
    --fold 0 \
    --seeds 0 1 2 \
    --epochs 80 \
    --probe_epochs 100 \
    --n_perm 20 \
    --n_perm_confirm 50

echo "DONE. Per-candidate/config + summary JSON under results/cigl/phase3a_baseline_repair/ (gitignored);"
echo "build the tracked reviewer summary from them after reviewer approval."
