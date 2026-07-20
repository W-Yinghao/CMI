#!/usr/bin/env bash
# Confirmatory one-fold REAL-DATA preflight on a CPU node (fold build only; no training; no GPU). De-risks
# the V100 one-fold run by validating the pilot fold builds on real BNCI2014_001 for the held-out target.
# Submit:
#   mkdir -p /projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs
#   OACI_DATALAKE_ROOT=/projects/EEG-foundation-model/datalake/raw \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out \
#       oaci/slurm_confirmatory_preflight.sh
# Per cluster policy: NO --time/walltime. CPU partition.
#SBATCH --job-name=oaci-confirmatory-preflight
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -u
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
: "${OACI_OUT_ROOT:=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-preflight/${SLURM_JOB_ID:-local}}"
TARGET="${OACI_TARGET_SUBJECT:-1}"
mkdir -p "$OACI_OUT_ROOT"

echo "[confirmatory-preflight] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) target=$TARGET"
$PY -m oaci.confirmatory.preflight --protocol oaci/protocol/confirmatory_v2.yaml \
    --datalake-root "$OACI_DATALAKE_ROOT" --manifest-out "$OACI_OUT_ROOT/pilot_manifest.yaml" \
    --target-subject "$TARGET" >"$OACI_OUT_ROOT/preflight-report.json" 2>"$OACI_OUT_ROOT/preflight.err"
rc=$?
echo "=== preflight rc=$rc ==="
if [ "$rc" -eq 0 ]; then
  $PY - "$OACI_OUT_ROOT/preflight-report.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"acceptance_ok={r['acceptance_ok']} X={r['X_shape']} roles={r['role_counts']} "
      f"target_fit={r['target_seen_by_fit']} manifest={r['manifest_hash'][:12]}")
PYEOF
else
  tail -25 "$OACI_OUT_ROOT/preflight.err" 2>/dev/null
fi
echo "=== OVERALL: $([ "$rc" -eq 0 ] && echo PASS || echo FAIL) (exit $rc) ==="
exit "$rc"
