#!/usr/bin/env bash
# C10b epoch-level GPU inference replay: for each C8 fold, reload the staging fold + trained trajectories and
# forward every candidate (ERM + risk-feasible OACI trajectory) to recompute per-candidate source_guard /
# source_audit / target metrics + selection & audit leakage point estimates. NO retrain / NO objective /
# NO schema change. The identity gate (selected checkpoints reproduce stored prediction hashes) is enforced
# per fold inside the module (a mismatch raises). Writes one JSON per fold to OACI_OUT_DIR. Submit:
#   OACI_LOSO_ROOT=/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012 \
#     OACI_OUT_DIR=/projects/EEG-foundation-model/yinghao/oaci-c10-replay \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out oaci/slurm_c10_replay.sh
# Single-fold smoke: OACI_SEEDS=0 OACI_TARGETS=1. Per cluster policy: NO --time.
#SBATCH --job-name=oaci-c10-replay
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -u
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE KMP_DETERMINISTIC_REDUCTION=true
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
: "${OACI_LOSO_ROOT:?requires OACI_LOSO_ROOT}"
: "${OACI_OUT_DIR:?requires OACI_OUT_DIR}"
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
case "$OACI_OUT_DIR" in "$(pwd)"/*|"$(pwd)") echo "OACI_OUT_DIR must be OUTSIDE the repo" >&2; exit 1;; esac
mkdir -p "$OACI_OUT_DIR"
SEEDS="${OACI_SEEDS:-0 1 2}"; TARGETS="${OACI_TARGETS:-1 2 3 4 5 6 7 8 9}"
echo "[c10-replay] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) out=$OACI_OUT_DIR seeds=[$SEEDS] targets=[$TARGETS]"
ok=0; fail=0
for s in $SEEDS; do for t in $TARGETS; do
  echo "--- replay seed-$s target-$(printf %03d $t) ---"
  $PY -m oaci.diagnostics.candidate_replay --loso-root "$OACI_LOSO_ROOT" --seed "$s" --target "$t" \
      --out-dir "$OACI_OUT_DIR"
  if [ $? -eq 0 ]; then ok=$((ok+1)); else fail=$((fail+1)); echo "FOLD FAILED: seed-$s target-$t"; fi
done; done
echo "=== C10 replay: ok=$ok fail=$fail ==="
[ "$fail" -eq 0 ] && echo "=== OVERALL: PASS ===" || echo "=== OVERALL: FAIL ($fail folds) ==="
exit "$fail"
