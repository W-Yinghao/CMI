#!/bin/bash
# CIGL_69C_METACMI_PHASE2_SEED0 — MetaCE + MetaCMI-Direct beta0.1 on EEGNetMini + EEGConformerMini (both
# internal/faithful-minimal/audit-compatible, NOT official), BNCI2014_001 full-LOSO seed0 (9 folds x 2 backbones
# x 2 methods = 36 GPU runs). ERM is REUSED from CIGL_69A comparators (NOT rerun). Isolated out_dir
# metacmi_phase2/ so the frozen ERM comparators in metacmi_gate/ are never touched. NO beta=0.5, NO conformer_full.
# Both backbones have a single linear head -> R3 head-replay is exact = CLASSIFIER-LEVEL reliance.
#SBATCH --job-name=metacmi-p2-bnci2014
#SBATCH --partition=A100,V100,V100-32GB,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/cigl/metacmi-p2-bnci2014-%j.out
#SBATCH --error=logs/cigl/metacmi-p2-bnci2014-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cigl
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then echo "FATAL: no CUDA"; exit 1; fi
"$PY" scripts/run_metacmi_gate.py --dataset BNCI2014_001 --device cuda \
  --backbones eegnet conformer --methods metace metacmi_direct_beta0.1 \
  --seed 0 --epochs 80 --probe_epochs 100 --n_perm 50 \
  --out_dir results/cigl/metacmi_phase2
