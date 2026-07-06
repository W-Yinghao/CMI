#!/bin/bash
# CIGL_69A2 (b) — conformer_full ERM full-LOSO seed0 on BNCI2015_001 (12 folds = 12 GPU runs).
# ISOLATED out_dir metacmi_gate_full/ so the frozen CIGL_69A evidence (metacmi_gate/) is NEVER touched or
# clobbered. conformer_full ONLY -> exactly 12 runs; eegnet/conformer ERM are reused from CIGL_69A at readout.
# Same protocol/preprocessing/split/audit as CIGL_69A (epochs 80, probe 100, n_perm 50, seed 0).
# BNCI2015_001 needs the readable datalake stage (owner-locked ~bci/database mirror) -- same stage CIGL_69A used.
#SBATCH --job-name=metacmi-full-erm-bnci2015
#SBATCH --partition=A100,V100,V100-32GB,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/cigl/metacmi-full-erm-bnci2015-%j.out
#SBATCH --error=logs/cigl/metacmi-full-erm-bnci2015-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cigl
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
STAGE="/home/infres/yinwang/mne_stage_bnci"; export MNE_DATASETS_BNCI_PATH="$STAGE" MNE_DATA="$STAGE"
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then echo "FATAL: no CUDA"; exit 1; fi
"$PY" scripts/run_metacmi_gate.py --dataset BNCI2015_001 --device cuda \
  --backbones conformer_full --methods erm \
  --seed 0 --epochs 80 --probe_epochs 100 --n_perm 50 \
  --out_dir results/cigl/metacmi_gate_full
