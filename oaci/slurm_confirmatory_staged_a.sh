#!/usr/bin/env bash
# Staged confirmatory one-fold PHASE A (V100 GPU record): train + GPU-prefetch every feasible candidate's
# features/logits into a staging dir, then submit PHASE B (CPU replay) with an afterok dependency. The
# V100 is released at the end of THIS job; the long CPU leakage scoring runs in Phase B without a GPU.
# Submit:
#   OACI_DATALAKE_ROOT=/projects/EEG-foundation-model/datalake/raw OACI_BOOTSTRAP_MODE=full \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out \
#       oaci/slurm_confirmatory_staged_a.sh
#SBATCH --job-name=oaci-staged-A
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE KMP_DETERMINISTIC_REDUCTION=true
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; REPO="$(pwd)"
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
: "${OACI_OUT_ROOT:=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-staged/${SLURM_JOB_ID:-local}}"
TARGET="${OACI_TARGET_SUBJECT:-1}"; SEED="${OACI_MODEL_SEED:-0}"; BMODE="${OACI_BOOTSTRAP_MODE:-full}"
LEAK_JOBS="${OACI_LEAKAGE_JOBS:-16}"; STAGING="$OACI_OUT_ROOT/staging"
nvidia-smi >/dev/null; [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "no GPU" >&2; exit 1; }
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()==1 else 1)" || { echo "need 1 GPU" >&2; exit 1; }
[ -d "$OACI_DATALAKE_ROOT/MNE-bnci-data" ] || { echo "no datalake" >&2; exit 1; }
case "$OACI_OUT_ROOT" in "$REPO"/*|"$REPO") echo "out must be outside repo" >&2; exit 1;; esac
[ -z "$(git -C "$REPO" status --porcelain -- oaci)" ] || { echo "dirty tree" >&2; exit 1; }
mkdir -p "$STAGING"
echo "[staged-A] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) commit=$(git -C "$REPO" rev-parse --short HEAD) target=$TARGET seed=$SEED bootstrap=$BMODE"

set +e
$PY -m oaci.confirmatory.staged_demo phase-a --protocol oaci/protocol/confirmatory_v2.yaml \
    --datalake-root "$OACI_DATALAKE_ROOT" --staging-dir "$STAGING" \
    --manifest-out "$OACI_OUT_ROOT/pilot_manifest.yaml" --target-subject "$TARGET" --model-seed "$SEED" \
    --bootstrap-mode "$BMODE" >"$OACI_OUT_ROOT/phase-a-report.json" 2>"$OACI_OUT_ROOT/phase-a.err"
a_rc=$?
echo "=== phase A rc=$a_rc ==="
[ "$a_rc" -ne 0 ] && { tail -25 "$OACI_OUT_ROOT/phase-a.err"; echo "=== OVERALL: FAIL ==="; exit 1; }
$PY -c "import json;r=json.load(open('$OACI_OUT_ROOT/phase-a-report.json'));print('staging_bytes',r['staging_bytes'],'levels',r['levels'])"

# chain Phase B (CPU) -- it reads the staging dir; the V100 is freed when this job ends
B=$(OACI_DATALAKE_ROOT="$OACI_DATALAKE_ROOT" OACI_OUT_ROOT="$OACI_OUT_ROOT" OACI_TARGET_SUBJECT="$TARGET" \
    OACI_MODEL_SEED="$SEED" OACI_BOOTSTRAP_MODE="$BMODE" OACI_LEAKAGE_JOBS="$LEAK_JOBS" OACI_REPO="$REPO" \
    sbatch --parsable --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out \
      oaci/slurm_confirmatory_staged_b.sh 2>&1)
echo "=== submitted phase B: $B ==="
echo "=== OVERALL: PASS (phase A) ==="
