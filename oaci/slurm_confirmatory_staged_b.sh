#!/usr/bin/env bash
# Staged confirmatory one-fold PHASE B (CPU replay): rebuild the fold + context, resume
# select/audit/finalize from the staging store (full-bootstrap leakage, process-parallel, NO GPU), and
# write -> deep-verify the artifact. Normally submitted by slurm_confirmatory_staged_a.sh; reads
# $OACI_OUT_ROOT/staging. Per cluster policy: NO --time.
#SBATCH --job-name=oaci-staged-B
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; REPO="${OACI_REPO:-$(pwd)}"
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
: "${OACI_OUT_ROOT:?Phase B requires OACI_OUT_ROOT (the Phase A output root)}"
TARGET="${OACI_TARGET_SUBJECT:-1}"; SEED="${OACI_MODEL_SEED:-0}"; BMODE="${OACI_BOOTSTRAP_MODE:-full}"
LEAK_JOBS="${OACI_LEAKAGE_JOBS:-16}"; STAGING="$OACI_OUT_ROOT/staging"
[ -f "$STAGING/phase_a.json" ] || { echo "no Phase A staging at $STAGING" >&2; exit 1; }
case "$OACI_OUT_ROOT" in "$REPO"/*|"$REPO") echo "out must be outside repo" >&2; exit 1;; esac
[ -z "$(git -C "$REPO" status --porcelain -- oaci)" ] || { echo "dirty tree" >&2; exit 1; }
echo "[staged-B] node=$(hostname) commit=$(git -C "$REPO" rev-parse --short HEAD) target=$TARGET seed=$SEED bootstrap=$BMODE leakage_jobs=$LEAK_JOBS"

set +e
$PY -m oaci.confirmatory.staged_demo phase-b --protocol oaci/protocol/confirmatory_v2.yaml \
    --datalake-root "$OACI_DATALAKE_ROOT" --staging-dir "$STAGING" \
    --manifest-out "$OACI_OUT_ROOT/pilot_manifest_b.yaml" --target-subject "$TARGET" --model-seed "$SEED" \
    --bootstrap-mode "$BMODE" --leakage-jobs "$LEAK_JOBS" --output-root "$OACI_OUT_ROOT/artifacts" \
    --repo-root "$REPO" >"$OACI_OUT_ROOT/phase-b-report.json" 2>"$OACI_OUT_ROOT/phase-b.err"
b_rc=$?
echo "=== phase B rc=$b_rc ==="
if [ "$b_rc" -eq 0 ]; then
  $PY -c "import json;r=json.load(open('$OACI_OUT_ROOT/phase-b-report.json'));print('deep_verified',r['deep_verification_ok'],'target_fit_empty',r['target_fit_ids_empty'],'sci',r['artifact_scientific_hash'][:12],'pure',r['artifact_pure_science_hash'][:12])"
else
  tail -25 "$OACI_OUT_ROOT/phase-b.err"
fi
echo "=== OVERALL: $([ "$b_rc" -eq 0 ] && echo PASS || echo FAIL) (exit $b_rc) ==="
exit "$b_rc"
