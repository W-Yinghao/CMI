#!/bin/bash
#SBATCH --job-name=cigl-gate2-bnci001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cigl/%x-%j.out
#SBATCH --error=logs/cigl/%x-%j.err
#
# CIGL BINDING Gate-2 evidence — source-only GraphCMINet-ERM leakage probe on BNCI2014_001.
# Exploratory DIAGNOSTIC: NO regularizer training, NO lambda sweep, NO SOTA, NO target data.
#
# PARTITION + QOS ARE INTENTIONALLY NOT HARD-CODED. Supply them at submit time (confirm with the PI):
#     sbatch -p <partition> --qos=<qos> scripts/sbatch_cigl_gate2_bnci001.sh
# Candidate GPU partitions on this cluster (per project): V100 / V100-32GB / A100 (and H100/L40S/A40
# for longer walltime). Do NOT guess the final partition/QOS — the PI confirms.
#
# This script must run from the CIGL worktree checked out on project/cigl-gate2-bnci001-evidence.
set -euo pipefail

WORKTREE=/home/infres/yinwang/CMI_AAAI_cigl
cd "$WORKTREE"
mkdir -p logs/cigl results/cigl/phase2_real

# eeg2025: torch 2.6.0+cu124, moabb 1.5.0 (the runner builds GraphCMINet DIRECTLY, no braindecode).
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
if [ ! -x "$PY" ]; then
    echo "ERROR: eeg2025 python not found at $PY — adjust PY or create the env." >&2
    exit 2
fi

# Offline MOABB/MNE datalake (read-only); the runner must NOT download on a compute node.
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  branch=$(git -C "$WORKTREE" rev-parse --abbrev-ref HEAD)  commit=$(git -C "$WORKTREE" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
# fail closed if no GPU is visible (so we never silently fall back to a multi-hour CPU run)
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)" \
    || { echo "ERROR: CUDA not available on this node — check partition/--gres." >&2; exit 3; }

# BINDING Gate-2 run — do NOT reduce seeds / n_perm / epochs / probe_epochs.
srun "$PY" scripts/run_cigl_phase2_real_probe.py \
    --dataset BNCI2014_001 \
    --device cuda \
    --seeds 0 1 2 \
    --n_perm 50 \
    --max_folds 1 \
    --epochs 80 \
    --probe_epochs 100 \
    --gate_alpha 0.05

echo "DONE. Per-seed + summary JSON under results/cigl/phase2_real/ (gitignored; build the tracked"
echo "reviewer summary docs/CIGL_13_GATE2_REAL_RESULTS_BNCI2014_001.md from them)."
