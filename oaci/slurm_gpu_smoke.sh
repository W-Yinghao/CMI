#!/usr/bin/env bash
# V100 two-order BNCI GPU smoke. Determinism env is exported BEFORE any Python (CUBLAS_WORKSPACE_CONFIG
# must exist before the CUDA context is created). Logs + artifacts live OUTSIDE the repo; the read-only
# datalake is never written. Submit from the worktree root (logs outside the repo):
#   mkdir -p /projects/EEG-foundation-model/yinghao/oaci-gpu-logs
#   OACI_DATALAKE_ROOT=/projects/EEG-foundation-model/datalake/raw \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-gpu-logs/%x-%j.out oaci/slurm_gpu_smoke.sh
# Per cluster policy: NO --time/walltime. Do NOT override CUDA_VISIBLE_DEVICES (use the SLURM allocation).
#SBATCH --job-name=oaci-bnci-gpu-smoke
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail

# --- determinism env: exported before the FIRST python invocation ---
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
: "${OACI_ARTIFACT_ROOT:=/projects/EEG-foundation-model/yinghao/oaci-gpu-artifacts/${SLURM_JOB_ID:-local}}"

# --- preconditions ---
nvidia-smi >/dev/null
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "no CUDA_VISIBLE_DEVICES from SLURM" >&2; exit 1; }
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()==1 else 1)" \
  || { echo "expected exactly one visible GPU" >&2; exit 1; }
[ -d "$OACI_DATALAKE_ROOT/MNE-bnci-data" ] || { echo "datalake missing BNCI at $OACI_DATALAKE_ROOT" >&2; exit 1; }
case "$OACI_ARTIFACT_ROOT" in "$REPO"/*|"$REPO") echo "artifact root must be OUTSIDE the repo" >&2; exit 1;; esac
[ -z "$(git -C "$REPO" status --porcelain -- oaci)" ] || { echo "scientific tree is dirty" >&2; exit 1; }
mkdir -p "$OACI_ARTIFACT_ROOT"

echo "[gpu-smoke] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) commit=$(git -C "$REPO" rev-parse --short HEAD)"

set +e
$PY -m oaci.tests.test_bnci_gpu_runtime; rt_rc=$?
$PY -m oaci.tests.test_bnci_gpu_runner; rn_rc=$?
$PY -m oaci.runner.bnci_gpu_demo --manifest oaci/protocol/smoke_v1.yaml --datalake-root "$OACI_DATALAKE_ROOT" \
    --artifact-root "$OACI_ARTIFACT_ROOT/demo" --repo-root "$REPO" --model-seed 0 \
    >"$OACI_ARTIFACT_ROOT/gpu-smoke-report.json" 2>"$OACI_ARTIFACT_ROOT/gpu-smoke.err"; demo_rc=$?

val_rc=1
if [ "$demo_rc" -eq 0 ]; then
  CANON=$($PY -c "import json;print(json.load(open('$OACI_ARTIFACT_ROOT/gpu-smoke-report.json'))['canonical_artifact']['dir'])")
  REV=$($PY -c "import json;print(json.load(open('$OACI_ARTIFACT_ROOT/gpu-smoke-report.json'))['reversed_artifact']['dir'])")
  $PY -m oaci.artifacts.verify "$CANON"; vc=$?
  $PY -m oaci.artifacts.verify "$REV"; vr=$?
  $PY - "$OACI_ARTIFACT_ROOT/gpu-smoke-report.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
oc = r["order_comparison"]
ok = (oc["all_equal"] and r["bn_all_equal_to_erm"] and r["rng_unchanged"]
      and r["canonical_artifact"]["deep_verification_ok"] and r["reversed_artifact"]["deep_verification_ok"])
print(f"comparison all_equal={oc['all_equal']} bn_equal={r['bn_all_equal_to_erm']} "
      f"artifact_hash={r['canonical_artifact']['artifact_scientific_hash'][:12]} "
      f"reversed={r['reversed_artifact']['artifact_scientific_hash'][:12]}")
sys.exit(0 if ok else 1)
PYEOF
  cmp_rc=$?
  if [ "$vc" -eq 0 ] && [ "$vr" -eq 0 ] && [ "$cmp_rc" -eq 0 ]; then val_rc=0; fi
fi

echo "=== rc: runtime=$rt_rc runner=$rn_rc demo=$demo_rc validator=$val_rc ==="
[ "$demo_rc" -ne 0 ] && tail -25 "$OACI_ARTIFACT_ROOT/gpu-smoke.err" 2>/dev/null
if [ "$rt_rc" -eq 0 ] && [ "$rn_rc" -eq 0 ] && [ "$demo_rc" -eq 0 ] && [ "$val_rc" -eq 0 ]; then fail=0; else fail=1; fi
echo "=== OVERALL: $([ "$fail" -eq 0 ] && echo PASS || echo FAIL) (exit $fail) ==="
exit "$fail"
