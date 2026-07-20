#!/usr/bin/env bash
# C18-P0 candidate-replay smoke / identity gate (GPU, NO retrain). Re-infer the C17 candidate checkpoints for
# ONE deterministic slice and verify: (1) selected-ckpt identity vs stored logits, (2) recomputed no-mask
# source scalars reproduce the persisted C10 atlas within tolerance, (3) per-sample source logits + Z-features
# round-trip. The full C18-P replay may run only if this PASSES. Submit:
#   OACI_LOSO_ROOT=/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012 \
#     OACI_C10_DIR=/projects/EEG-foundation-model/yinghao/oaci-c10-replay \
#     OACI_OUT_DIR=/projects/EEG-foundation-model/yinghao/oaci-c18-extract \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out oaci/slurm_c18p0_smoke.sh
# Per cluster policy: NO --time.
#SBATCH --job-name=oaci-c18p0-smoke
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
SEED="${OACI_SEEDS:-0}"; TARGET="${OACI_TARGETS:-1}"
echo "[c18p0] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) seed=$SEED target=$TARGET out=$OACI_OUT_DIR"
$PY -m oaci.support_stress.smoke --loso-root "$OACI_LOSO_ROOT" --seed "$SEED" --target "$TARGET" \
    --out-dir "$OACI_OUT_DIR" --c10-dir "$OACI_C10_DIR" --report oaci/reports/C18_P0_SMOKE.json
rc=$?
echo "=== C18-P0 rc=$rc ==="
exit "$rc"
