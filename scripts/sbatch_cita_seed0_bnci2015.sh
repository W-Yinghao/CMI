#!/bin/bash
# CITA_01 seed0 gate — target-unlabeled offline transductive adaptation on EEGNetMini + EEGConformerMini
# (internal/faithful-minimal, NOT official), BNCI2015_001 full-LOSO seed0. Per (fold,backbone): train source-ERM
# M0 ONCE, adapt a COPY per method {erm_no_adapt, tta_control, cita_cmi} -> clean ERM->TTA->CITA attribution
# (ERM reused within-fold). lam_cita=0.010 (NO grid). Classifier-level R3 (linear heads). Target y FORBIDDEN in
# adaptation. NO ConformerFull. BNCI2015_001 needs the readable datalake stage (same stage CIGL_69A/69C used).
#SBATCH --job-name=cita-s0-bnci2015
#SBATCH --partition=A100,V100,V100-32GB,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/cita/cita-s0-bnci2015-%j.out
#SBATCH --error=logs/cita/cita-s0-bnci2015-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cita
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
STAGE="/home/infres/yinwang/mne_stage_bnci"; export MNE_DATASETS_BNCI_PATH="$STAGE" MNE_DATA="$STAGE"
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then echo "FATAL: no CUDA"; exit 1; fi
"$PY" scripts/run_cita_gate.py --dataset BNCI2015_001 --device cuda \
  --backbones eegnet conformer --methods erm_no_adapt tta_control cita_cmi \
  --seed 0 --epochs 80 --adapt_steps 50 --lam_cita 0.010 --probe_epochs 100 --n_perm 50
