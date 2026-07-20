#!/usr/bin/env bash
# C18-P FULL candidate-replay extraction (GPU, NO retrain). One ARRAY task per (seed,target) fold: re-infer the
# C17 candidate checkpoints, identity-gated, persisting per-unit source logits + per-candidate Z-features +
# base support graphs for downstream CPU mask recompute. 27 folds = seeds {0,1,2} x targets {1..9}. Submit:
#   OACI_LOSO_ROOT=/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012 \
#     OACI_C10_DIR=/projects/EEG-foundation-model/yinghao/oaci-c10-replay \
#     OACI_OUT_DIR=/projects/EEG-foundation-model/yinghao/oaci-c18-extract \
#     sbatch --array=0-26 --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%A_%a.out \
#       oaci/slurm_c18p_extract.sh
# Per cluster policy: NO --time. NO seeds [3,4], NO BNCI2014_004 (targets stay 1..9).
#SBATCH --job-name=oaci-c18p-extract
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
: "${OACI_C10_DIR:?requires OACI_C10_DIR}"
: "${OACI_OUT_DIR:?requires OACI_OUT_DIR}"
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
case "$OACI_OUT_DIR" in "$(pwd)"/*|"$(pwd)") echo "OACI_OUT_DIR must be OUTSIDE the repo" >&2; exit 1;; esac
mkdir -p "$OACI_OUT_DIR"
# Two modes: ARRAY (one fold per task, when SLURM_ARRAY_TASK_ID set) or SERIAL loop over OACI_SEEDS x
# OACI_TARGETS (default all 27; used when the QOS submit limit blocks an array). Already-extracted folds skip.
run_fold() {
  local SEED=$1 TARGET=$2
  local MANIFEST="$OACI_OUT_DIR/seed-$SEED-target-$(printf %03d $TARGET)/extract_manifest.json"
  if [ -f "$MANIFEST" ] && [ "${OACI_FORCE:-0}" != "1" ]; then echo "skip seed-$SEED target-$TARGET (done)"; return 0; fi
  echo "--- extract seed-$SEED target-$TARGET ---"
  $PY -m oaci.support_stress.replay_extract --loso-root "$OACI_LOSO_ROOT" --seed "$SEED" --target "$TARGET" \
      --out-dir "$OACI_OUT_DIR" --c10-dir "$OACI_C10_DIR"
}
echo "[c18p-extract] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null)"
fail=0
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  IDX="$SLURM_ARRAY_TASK_ID"; run_fold $(( IDX / 9 )) $(( IDX % 9 + 1 )) || fail=1
else
  for s in ${OACI_SEEDS:-0 1 2}; do for t in ${OACI_TARGETS:-1 2 3 4 5 6 7 8 9}; do
    run_fold "$s" "$t" || { fail=$((fail+1)); echo "FOLD FAILED seed-$s target-$t"; }
  done; done
fi
echo "=== c18p-extract: fail=$fail ==="
exit "$fail"
