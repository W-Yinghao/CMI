#!/bin/bash
#SBATCH --job-name=fcigl-gate-bnci2015
#SBATCH --partition=A100,V100,V100-32GB,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/cigl/fcigl-gate-bnci2015-%j.out
#SBATCH --error=logs/cigl/fcigl-gate-bnci2015-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cigl
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
STAGE="/home/infres/yinwang/mne_stage_bnci"; export MNE_DATASETS_BNCI_PATH="$STAGE" MNE_DATA="$STAGE"
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: CUDA not available on $(hostname); refusing to run on CPU." >&2; exit 1
fi
"$PY" scripts/run_cigl_functional_gate.py --dataset BNCI2015_001 --device cuda --seed 0 \
  --epochs 80 --probe_epochs 100 --n_perm 50 --fcigl_k 2 --fcigl_update_every 10
