#!/usr/bin/env bash
# V100 confirmatory ONE-FOLD pipeline validation (BNCI2014_001, target=subject-001, full budget, seeds
# 0,1,2). NOT confirmatory efficacy evidence. Determinism env is exported BEFORE any Python; the
# materialized manifest, artifacts and logs live OUTSIDE the repo; the datalake is read-only. Submit:
#   mkdir -p /projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs
#   OACI_DATALAKE_ROOT=/projects/EEG-foundation-model/datalake/raw \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out \
#       oaci/slurm_confirmatory_onefold.sh
# Per cluster policy: NO --time/walltime. Do NOT override CUDA_VISIBLE_DEVICES (use the SLURM allocation).
#SBATCH --job-name=oaci-confirmatory-onefold
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE
export KMP_DETERMINISTIC_REDUCTION=true

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$(pwd)"
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
: "${OACI_OUT_ROOT:=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-onefold/${SLURM_JOB_ID:-local}}"
TARGET="${OACI_TARGET_SUBJECT:-1}"
SEEDS="${OACI_MODEL_SEEDS:-0,1,2}"

nvidia-smi >/dev/null
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "no CUDA_VISIBLE_DEVICES from SLURM" >&2; exit 1; }
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()==1 else 1)" \
  || { echo "expected exactly one visible GPU" >&2; exit 1; }
[ -d "$OACI_DATALAKE_ROOT/MNE-bnci-data" ] || { echo "datalake missing BNCI at $OACI_DATALAKE_ROOT" >&2; exit 1; }
case "$OACI_OUT_ROOT" in "$REPO"/*|"$REPO") echo "output root must be OUTSIDE the repo" >&2; exit 1;; esac
[ -z "$(git -C "$REPO" status --porcelain -- oaci)" ] || { echo "scientific tree is dirty" >&2; exit 1; }
mkdir -p "$OACI_OUT_ROOT"

echo "[confirmatory-onefold] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) commit=$(git -C "$REPO" rev-parse --short HEAD) target=$TARGET seeds=$SEEDS"

set +e
$PY -m oaci.confirmatory.demo --protocol oaci/protocol/confirmatory_v2.yaml \
    --datalake-root "$OACI_DATALAKE_ROOT" --output-root "$OACI_OUT_ROOT/artifacts" \
    --manifest-out "$OACI_OUT_ROOT/pilot_manifest.yaml" --repo-root "$REPO" \
    --target-subject "$TARGET" --model-seeds "$SEEDS" \
    >"$OACI_OUT_ROOT/onefold-report.json" 2>"$OACI_OUT_ROOT/onefold.err"
demo_rc=$?

ver_rc=1
if [ "$demo_rc" -eq 0 ]; then
  ver_rc=0
  for d in "$OACI_OUT_ROOT"/artifacts/seed-*/*/; do
    [ -f "$d/COMMITTED.json" ] || continue
    $PY -m oaci.artifacts.verify "$d" || ver_rc=1
  done
  $PY - "$OACI_OUT_ROOT/onefold-report.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"target={r['target_subject']} seeds={r['model_seeds']} deep_verified={r['all_seeds_deep_verified']} "
      f"target_fit_empty={r['all_target_fit_ids_empty']} manifest={r['manifest_hash'][:12]}")
for sb in r["seeds"]:
    print(f"  seed {sb['model_seed']}: sci={sb['artifact_scientific_hash'][:12]} "
          f"pure={sb['artifact_pure_science_hash'][:12]} verified={sb['deep_verification_ok']}")
sys.exit(0 if (r["all_seeds_deep_verified"] and r["all_target_fit_ids_empty"]) else 1)
PYEOF
  [ $? -ne 0 ] && ver_rc=1
fi

echo "=== rc: demo=$demo_rc verify=$ver_rc ==="
[ "$demo_rc" -ne 0 ] && tail -25 "$OACI_OUT_ROOT/onefold.err" 2>/dev/null
if [ "$demo_rc" -eq 0 ] && [ "$ver_rc" -eq 0 ]; then fail=0; else fail=1; fi
echo "=== OVERALL: $([ "$fail" -eq 0 ] && echo PASS || echo FAIL) (exit $fail) ==="
exit "$fail"
