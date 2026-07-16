#!/bin/bash
#SBATCH --job-name=cmitrace-dumpregen
#SBATCH --partition=A100,V100,V100-32GB,A40,P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/cmi_trace/%x-%j.out
#SBATCH --error=logs/cmi_trace/%x-%j.out
# Stage 7: regenerate frozen EEGNet/TSMNet dumps (idempotent; skips banked folds). Params via env:
# DR_DATASET, DR_BACKBONE (EEGNet|TSMNet), DR_SEED. GPU env eeg2025; fail closed on no-CUDA.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cmi_trace
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data

DR_DATASET="${DR_DATASET:-BNCI2014_001}"
DR_BACKBONE="${DR_BACKBONE:-EEGNet}"
DR_SEED="${DR_SEED:-0}"

# EEGNet dumps: env icml (moabb 1.2.0 + braindecode 0.8 compatible; eeg2025 has moabb-1.5/torchaudio env-rot).
# TSMNet dumps: eeg2025 (has spdnets) — BLOCKED by torchaudio ABI break; see execution note.
PY="${DR_PY:-/home/infres/yinwang/anaconda3/envs/icml/bin/python}"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) dataset=${DR_DATASET} backbone=${DR_BACKBONE} seed=${DR_SEED}"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: CUDA unavailable on $(hostname); refusing CPU fallback." >&2; exit 1
fi

"$PY" -m tos_cmi.run_eeg_frozen_pilot \
  --dataset "${DR_DATASET}" --backbone "${DR_BACKBONE}" \
  --target-subjects all --configs erm:0 --seed "${DR_SEED}" --device cuda \
  --out-root tos_cmi/results/tos_cmi_eeg_frozen
