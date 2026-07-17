#!/bin/bash
# MCC-λ1: the single bounded-strength lever (λ_MCC 0.25 -> 1.0). FAIL-RESUMABLE SLURM ARRAY over the 63 bundles;
# each re-runs all 3 arms (A/B/C) from the SAME (hash-verified) ERM warm-up as the λ=0.25 round, so there is one
# training SHA per round and no stale-A ambiguity. GPU + env `icml`. ONLY λ changes; every other setting fixed
# (20 epochs, LR 1e-4, K=4, sampler, source-only selection, EEGNet 16d). Manuscript FROZEN; no amendment.
#
#   sbatch --array=0-62 scripts/sbatch_mcc_lambda1.sh
#   # after 189/189: python scripts/aggregate_mcc.py --from-dir results/cmi_trace_mcc_lambda1 --expect 63 \
#   #                    --compare-dir results/cmi_trace_mcc
#SBATCH --job-name=mcc-l1
#SBATCH --partition=A100,V100,V100-32GB,A40,P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/mcc/mcc-l1-%A_%a.out
#SBATCH --error=logs/mcc/mcc-l1-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/mcc
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
OUTDIR="results/cmi_trace_mcc_lambda1"
IDX="${SLURM_ARRAY_TASK_ID:-${1:?need a bundle index}}"
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) bundle=$IDX lambda=1.0"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: CUDA not available on $(hostname); MCC training refuses to run on CPU." >&2; exit 1
fi
BUNDLE=$("$PY" -c "from tos_cmi.train.run_mcc_arms import enumerate_bundles as e; ds,s,sd=e()[$IDX]; print(f'{ds}_sub{s}_seed{sd}')")
if [ -f "$OUTDIR/$BUNDLE.done" ]; then echo "bundle $IDX ($BUNDLE) already done -> skip"; exit 0; fi
# reuse the λ=0.25 warm-up cache; verify the warm-up hash matches that round per bundle (fail loud on mismatch).
"$PY" -m tos_cmi.train.run_mcc_arms --bundle-index "$IDX" --device cuda --out-dir "$OUTDIR" \
  --cache-dir results/cmi_trace_mcc/warmup_cache --verify-warmup-from results/cmi_trace_mcc \
  --warmup-epochs 300 --cont-epochs 20 --K 4 --lam-mcc 1.0 --ramp 5 --cont-lr 1e-4
echo "[mcc-l1] bundle $IDX complete"
