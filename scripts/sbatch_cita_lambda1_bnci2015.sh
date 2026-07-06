#!/bin/bash
# CITA_02_ACTIVE_CMI_LAMBDA_PROBE — the ONE PM-approved active-CMI probe: lambda_cita=1.0 (NOT a grid).
# BNCI2015_001 full-LOSO seed0. Matched attribution: per (fold,backbone) train source-ERM M0 ONCE, adapt a COPY
# for erm_no_adapt (free M0 eval) + tta_control + cita_cmi lambda1.0 -> all from the IDENTICAL fold-M0. Isolated
# out_dir gate_lambda1/. Records loss-scale diagnostics to prove the CMI term is ACTIVE. NO grid, NO seeds 1/2,
# NO ConformerFull. BNCI2015_001 needs the readable datalake stage (same stage CIGL_69A/69C/CITA seed0 used).
#SBATCH --job-name=cita-l1-bnci2015
#SBATCH --partition=A100,V100,V100-32GB,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/cita/cita-l1-bnci2015-%j.out
#SBATCH --error=logs/cita/cita-l1-bnci2015-%j.out
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
  --seed 0 --epochs 80 --adapt_steps 50 --lam_cita 1.0 --probe_epochs 100 --n_perm 50 \
  --out_dir results/cita/gate_lambda1
