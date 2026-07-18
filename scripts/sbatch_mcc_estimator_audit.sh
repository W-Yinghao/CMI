#!/bin/bash
# MCC estimator audit — 63-cell FAIL-RESUMABLE SLURM ARRAY. Per bundle: load the hash-verified ERM warm-up (no
# training) and compute the exact two-pass full-source MCC gradient vs K=4/K=16 episodic estimators + diagnostics.
# GPU + env `icml` (forward+backward through EEGNet). NO training, NO 189-arm fleet. Manuscript FROZEN.
#
#   sbatch --array=0-62 scripts/sbatch_mcc_estimator_audit.sh
#   # after 63/63: python scripts/aggregate_mcc_estimator_audit.py --from-dir results/cmi_trace_mcc_estimator_audit --expect 63
#SBATCH --job-name=mcc-audit
#SBATCH --partition=A100,V100,V100-32GB,A40,P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/mcc/mcc-audit-%A_%a.out
#SBATCH --error=logs/mcc/mcc-audit-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/mcc
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
OUTDIR="results/cmi_trace_mcc_estimator_audit"
IDX="${SLURM_ARRAY_TASK_ID:-${1:?need a bundle index}}"
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) bundle=$IDX"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: CUDA not available on $(hostname); refusing CPU." >&2; exit 1
fi
BUNDLE=$("$PY" -c "from tos_cmi.train.run_mcc_arms import enumerate_bundles as e; ds,s,sd=e()[$IDX]; print(f'{ds}_sub{s}_seed{sd}')")
if compgen -G "$OUTDIR/cell_$(printf '%03d' "$IDX")_*.done" > /dev/null; then echo "bundle $IDX ($BUNDLE) already done -> skip"; exit 0; fi
"$PY" -m scripts.run_mcc_estimator_audit --bundle-index "$IDX" --device cuda --out-dir "$OUTDIR" \
  --cache-dir results/cmi_trace_mcc/warmup_cache --verify-warmup-from results/cmi_trace_mcc --R 64 --K-large 16
echo "[mcc-audit] bundle $IDX complete"
