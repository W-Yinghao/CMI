#!/usr/bin/env bash
# C26 re-persistence re-inference (GPU, V100). Evidence completion: re-run the P0-validated no-retraining
# target_audit forward and persist the per-SAMPLE evidence the C24 aggregate sidecar lacks (per-sample logits +
# deterministic split membership + QUARANTINED labels) so C26 Q1 split-stability + Q5 label diagnostics resolve.
# NO training / tuning / selection. Reuses candidate_replay forward + identity gate.
#
# MODE via OACI_C26_MODE:
#   p0    -> one deterministic slice (OACI_P0_SEED/OACI_P0_TARGET default 0/4), gates G1-G8, no persistence.
#   full  -> ARRAY over 27 folds; each writes seed-S-target-TTT.{unlabeled,labels}.npz.
#   summarize -> CPU: npz -> C26 split sidecar JSON (splits label-free + quarantined label diagnostics).
# Submit P0:
#   OACI_C26_MODE=p0 sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%A.out oaci/slurm_c26_repersist.sh
# Submit full (after P0):
#   OACI_C26_MODE=full sbatch --array=0-26%16 --output=.../%x-%A_%a.out oaci/slurm_c26_repersist.sh
# Per cluster policy: NO --time. NO seeds [3,4], NO BNCI2014_004 (targets 1..9).
#SBATCH --job-name=oaci-c26-repersist
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
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
: "${OACI_C26_MODE:=p0}"
: "${OACI_RP_DIR:=/projects/EEG-foundation-model/yinghao/oaci-c26-repersist}"
: "${OACI_SPLIT_SIDECAR:=/projects/EEG-foundation-model/yinghao/oaci-c26-predmix-splits.json}"
echo "[c26-repersist] mode=$OACI_C26_MODE node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null)"

case "$OACI_C26_MODE" in
  p0)
    $PY -m oaci.predmix_mechanism.target_repersist --loso-root "$OACI_LOSO_ROOT" \
        --seed "${OACI_P0_SEED:-0}" --target "${OACI_P0_TARGET:-4}" --p0 --p0-out oaci/reports/C26_RP_P0_SMOKE.json
    ;;
  full)
    mkdir -p "$OACI_RP_DIR"
    IDX="${SLURM_ARRAY_TASK_ID:?full mode needs --array}"; SEED=$(( IDX / 9 )); TARGET=$(( IDX % 9 + 1 ))
    if [ -f "$OACI_RP_DIR/seed-$SEED-target-$(printf %03d $TARGET).unlabeled.npz" ] && [ "${OACI_FORCE:-0}" != "1" ]; then
      echo "skip seed-$SEED target-$TARGET (done)"; exit 0; fi
    $PY -m oaci.predmix_mechanism.target_repersist --loso-root "$OACI_LOSO_ROOT" \
        --seed "$SEED" --target "$TARGET" --out-dir "$OACI_RP_DIR"
    ;;
  summarize)
    $PY -m oaci.predmix_mechanism.target_repersist --summarize-from "$OACI_RP_DIR" --out-sidecar "$OACI_SPLIT_SIDECAR"
    ;;
  *) echo "unknown OACI_C26_MODE=$OACI_C26_MODE" >&2; exit 1;;
esac
