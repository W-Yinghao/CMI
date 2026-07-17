#!/bin/bash
# MCC 3-arm continuation training — real EEG GPU tranche. FAIL-RESUMABLE SLURM ARRAY over the 63 bundles
# (dataset x LOSO subject x seed); each bundle trains ONE ERM warm-up then forks 3 arms (A/B/C) = 189 cells.
# GPU + env `icml` (braindecode 0.8 + moabb 1.2; NOT c84c). Fixed settings from config
# cmi_trace_mechanism_consistency.yaml (lambda=0.25, 20 continuation epochs, no sweep). Manuscript FROZEN.
# Only the project owner may stop a scientific line.
#
#   sbatch --array=0-62 scripts/sbatch_mcc_arms.sh
#   # after all complete: python scripts/aggregate_mcc.py --from-dir results/cmi_trace_mcc --expect 63
#SBATCH --job-name=mcc-arms
#SBATCH --partition=A100,V100,V100-32GB,A40,P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/mcc/mcc-%A_%a.out
#SBATCH --error=logs/mcc/mcc-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/mcc
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
OUTDIR="results/cmi_trace_mcc"
IDX="${SLURM_ARRAY_TASK_ID:-${1:?need a bundle index}}"
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) bundle=$IDX"
echo "config_hash=$(sha256sum configs/cmi_trace_mechanism_consistency.yaml | cut -c1-16)"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: CUDA not available on $(hostname); MCC training refuses to run on CPU." >&2; exit 1
fi
# resumable: skip a bundle whose .done already exists
BUNDLE=$("$PY" -c "from tos_cmi.train.run_mcc_arms import enumerate_bundles as e; ds,s,sd=e()[$IDX]; print(f'{ds}_sub{s}_seed{sd}')")
if [ -f "$OUTDIR/$BUNDLE.done" ]; then echo "bundle $IDX ($BUNDLE) already done -> skip"; exit 0; fi
"$PY" -m tos_cmi.train.run_mcc_arms --bundle-index "$IDX" --device cuda --out-dir "$OUTDIR" \
  --warmup-epochs 300 --cont-epochs 20 --K 4 --lam-mcc 0.25 --ramp 5 --cont-lr 1e-4
echo "[mcc-arms] bundle $IDX complete"
