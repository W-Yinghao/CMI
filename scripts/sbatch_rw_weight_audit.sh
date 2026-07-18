#!/bin/bash
# RW0 frozen weight audit -- 63-cell CPU array (NO training, no GPU: warm-up forward + sklearn LOSO LogReg). env
# `icml` (braindecode/moabb). Resumable (skips .done). Submit via SLURM, never on the login node. Manuscript FROZEN.
#
#   sbatch --array=0-62 scripts/sbatch_rw_weight_audit.sh
#   # after 63/63: python scripts/aggregate_rw_weight_audit.py --from-dir results/cmi_trace_risk_weighted_mcc --expect 63
#SBATCH --job-name=rw-waudit
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/mcc/rw-waudit-%A_%a.out
#SBATCH --error=logs/mcc/rw-waudit-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/mcc
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
OUTDIR="results/cmi_trace_risk_weighted_mcc"
IDX="${SLURM_ARRAY_TASK_ID:-${1:?need a cell index}}"
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) cell=$IDX"
if compgen -G "$OUTDIR/wcells/cell_$(printf '%03d' "$IDX")_*.done" > /dev/null; then echo "cell $IDX already done -> skip"; exit 0; fi
"$PY" -m scripts.run_risk_weight_audit --device cpu --bundle-index "$IDX" --out-dir "$OUTDIR"
echo "[rw-waudit] cell $IDX complete"
