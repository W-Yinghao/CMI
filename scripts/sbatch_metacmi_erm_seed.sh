#!/bin/bash
# CIGL_69A2 idle-slot fill — parametrized ERM multi-seed replication (NO MetaCMI/beta; Phase-2 stays held).
# Params via --export: DS (dataset), SEED, BB (space-joined backbones), OUTDIR. Same protocol as CIGL_69A
# (epochs 80, probe 100, n_perm 50). conformer_full -> OUTDIR=results/cigl/metacmi_gate_full (isolated);
# eegnet/conformer -> OUTDIR=results/cigl/metacmi_gate (adds seed>=1 files next to frozen seed0, no clobber).
#SBATCH --job-name=metacmi-erm-ms
#SBATCH --partition=A100,V100,V100-32GB,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/cigl/metacmi-erm-ms-%j.out
#SBATCH --error=logs/cigl/metacmi-erm-ms-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cigl
: "${DS:?need DS}" "${SEED:?need SEED}" "${BB:?need BB}" "${OUTDIR:?need OUTDIR}"
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [ "$DS" = "BNCI2015_001" ]; then STAGE="/home/infres/yinwang/mne_stage_bnci"; export MNE_DATASETS_BNCI_PATH="$STAGE" MNE_DATA="$STAGE"; fi
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) DS=$DS SEED=$SEED BB='$BB' OUTDIR=$OUTDIR"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then echo "FATAL: no CUDA"; exit 1; fi
"$PY" scripts/run_metacmi_gate.py --dataset "$DS" --device cuda --backbones $BB --methods erm \
  --seed "$SEED" --epochs 80 --probe_epochs 100 --n_perm 50 --out_dir "$OUTDIR"
