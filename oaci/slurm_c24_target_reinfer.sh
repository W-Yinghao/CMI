#!/usr/bin/env bash
# C24 Stage-2/3 — NO-RETRAINING target-audit re-inference (GPU, V100). Forwards every feasible-OACI candidate
# checkpoint on the target_audit role and persists a LABEL-FREE confidence-geometry summary per candidate to
# fill C24 R3/R4. Reuses candidate_replay's forward + identity machinery (predictions reproduce stored
# artifacts). NO retraining, NO objective, NO selection, NO target labels in the feature path.
#
# MODE via OACI_C24_MODE:
#   p0   -> one deterministic slice (OACI_P0_SEED/OACI_P0_TARGET, default 0/4), gate G1-G8, write no sidecar.
#   full -> ARRAY over 27 folds (seeds {0,1,2} x targets {1..9}); each task writes a per-fold partial.
#   merge-> combine per-fold partials into the single sidecar.
# Submit P0:
#   OACI_C24_MODE=p0 sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%A.out \
#     oaci/slurm_c24_target_reinfer.sh
# Submit full (after P0 passes):
#   OACI_C24_MODE=full sbatch --array=0-26 --output=.../%x-%A_%a.out oaci/slurm_c24_target_reinfer.sh
# Per cluster policy: NO --time. NO seeds [3,4], NO BNCI2014_004 (targets stay 1..9).
#SBATCH --job-name=oaci-c24-reinfer
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
: "${OACI_LOSO_ROOT:=/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012}"
: "${OACI_C10_DIR:=/projects/EEG-foundation-model/yinghao/oaci-c10-replay}"
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
: "${OACI_C24_MODE:=p0}"
: "${OACI_SIDECAR:=/projects/EEG-foundation-model/yinghao/oaci-c24-target-unlabeled.json}"
: "${OACI_FOLD_DIR:=/projects/EEG-foundation-model/yinghao/oaci-c24-reinfer-folds}"
echo "[c24-reinfer] mode=$OACI_C24_MODE node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null)"

case "$OACI_C24_MODE" in
  p0)
    $PY -m oaci.information_ladder.target_reinfer --loso-root "$OACI_LOSO_ROOT" --c10-dir "$OACI_C10_DIR" \
        --seed "${OACI_P0_SEED:-0}" --target "${OACI_P0_TARGET:-4}" --p0 \
        --p0-out oaci/reports/C24_R3R4_P0_SMOKE.json
    ;;
  full)
    mkdir -p "$OACI_FOLD_DIR"
    IDX="${SLURM_ARRAY_TASK_ID:?full mode needs --array}"; SEED=$(( IDX / 9 )); TARGET=$(( IDX % 9 + 1 ))
    PART="$OACI_FOLD_DIR/seed-$SEED-target-$(printf %03d $TARGET).json"
    if [ -f "$PART" ] && [ "${OACI_FORCE:-0}" != "1" ]; then echo "skip seed-$SEED target-$TARGET (done)"; exit 0; fi
    $PY -m oaci.information_ladder.target_reinfer --loso-root "$OACI_LOSO_ROOT" --c10-dir "$OACI_C10_DIR" \
        --seed "$SEED" --target "$TARGET" --fold-out-dir "$OACI_FOLD_DIR"
    ;;
  merge)
    $PY -m oaci.information_ladder.target_reinfer --merge-from "$OACI_FOLD_DIR" --out-sidecar "$OACI_SIDECAR"
    ;;
  *) echo "unknown OACI_C24_MODE=$OACI_C24_MODE" >&2; exit 1;;
esac
