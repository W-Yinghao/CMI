#!/bin/bash
# Risk-Weighted MCC 3-arm training -- 63-bundle FAIL-RESUMABLE SLURM ARRAY (A=ERM / B=true-RW / C=weight-permuted
# from one hash-verified warm-up = 189 arms). GPU + env `icml`. P100 (sm_60) EXCLUDED (icml torch cu128 has no
# sm_60 kernels). Manuscript FROZEN.
#
#   sbatch --array=0-62 scripts/sbatch_rw_mcc_arms.sh
#   # after 189/189: python scripts/aggregate_rw_mcc.py --from-dir results/cmi_trace_rw_mcc --expect 63
#SBATCH --job-name=rw-mcc
# --time=1 day: fits A100/H100 MaxTime (1d) so all partitions are usable, and is ample for a ~15-min 3-arm bundle.
# (Default 2d would be rejected by A100/H100 -> PartitionTimeLimit; 1d clears that without under-cutting the run.)
#SBATCH --partition=A100,V100,V100-32GB,A40,H100
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/mcc/rw-mcc-%A_%a.out
#SBATCH --error=logs/mcc/rw-mcc-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/mcc
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
OUTDIR="results/cmi_trace_rw_mcc"
IDX="${SLURM_ARRAY_TASK_ID:-${1:?need a bundle index}}"
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) bundle=$IDX"
if ! "$PY" -c "import torch,sys
try: (torch.ones(8,device='cuda')@torch.ones(8,device='cuda')).item(); sys.exit(0)
except Exception as e: print('GPU kernel test failed:',e); sys.exit(1)"; then
  echo "GPU on $(hostname) cannot run torch kernels -> requeue" >&2; scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true; exit 1
fi
BUNDLE=$("$PY" -c "from tos_cmi.train.run_mcc_arms import enumerate_bundles as e; ds,s,sd=e()[$IDX]; print(f'{ds}_sub{s}_seed{sd}')")
if [ -f "$OUTDIR/$BUNDLE.done" ]; then echo "bundle $IDX ($BUNDLE) already done -> skip"; exit 0; fi
"$PY" -m scripts.run_rw_mcc_arms --bundle-index "$IDX" --device cuda --out-dir "$OUTDIR" \
  --cache-dir results/cmi_trace_mcc/warmup_cache --verify-warmup-from results/cmi_trace_mcc \
  --warmup-epochs 300 --cont-epochs 20 --K 4 --lam-rw 1.0 --ramp 5 --cont-lr 1e-4
echo "[rw-mcc] bundle $IDX complete"
