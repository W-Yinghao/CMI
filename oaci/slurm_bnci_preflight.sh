#!/bin/bash
# Dedicated BNCI2014-001 six-subject CPU preflight: loads the REAL data ONCE, runs the contract tests
# and the preflight CLI, then a lightweight validator over the emitted JSON. Reads the read-only
# datalake; writes nothing to the repo or datalake. Submit from the worktree root:
#   OACI_DATALAKE_ROOT=/projects/EEG-foundation-model/datalake/raw sbatch oaci/slurm_bnci_preflight.sh
# Per cluster policy: NO --time/walltime.
#SBATCH --job-name=oaci-bnci-preflight
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/oaci-bnci-preflight-%j.out
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
[ -d "$OACI_DATALAKE_ROOT/MNE-bnci-data" ] || { echo "datalake missing BNCI at $OACI_DATALAKE_ROOT" >&2; exit 1; }
WORK="${SLURM_TMPDIR:-$(mktemp -d)}"
trap 'rm -rf "$WORK"' EXIT

echo "[bnci-preflight] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) datalake=$OACI_DATALAKE_ROOT"

set +e          # capture return codes without aborting on the first failure
# 1) the real six-subject contract tests (one shared load inside the module)
OACI_DATALAKE_ROOT="$OACI_DATALAKE_ROOT" $PY -m oaci.tests.test_bnci_real_preflight; tests_rc=$?

# 2) the preflight CLI (canonical JSON on stdout) + a lightweight JSON validator
$PY -m oaci.runner.bnci_preflight --manifest oaci/protocol/smoke_v1.yaml \
    --datalake-root "$OACI_DATALAKE_ROOT" >"$WORK/preflight.json" 2>"$WORK/preflight.err"; cli_rc=$?
$PY - "$WORK/preflight.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
ok = (s["acceptance_ok"] and s["X_shape"] == [3456, 22, 385]
      and s["role_trials"] == {"source_train": 1728, "source_audit": 1152, "target_audit": 576}
      and s["network_attempt_count"] == 0 and s["raw_fingerprint_unchanged"]
      and s["level1"]["deleted_cell"] == {"count": 0.0, "mass": 0.0, "rows": 0}
      and s["audit_status"] == "estimable" and s["target_seen_by_fit"] is False)
print(f"validator acceptance_ok={s['acceptance_ok']} fold_scope={s['fold_scope_hash'][:12]} "
      f"resolved={s['resolved_preprocess_hash'][:12]} headers={s['header_count']}")
sys.exit(0 if ok else 1)
PYEOF
val_rc=$?

echo "=== rc: tests=$tests_rc cli=$cli_rc validator=$val_rc ==="
[ "$tests_rc" -ne 0 ] && tail -25 "$WORK/preflight.err" 2>/dev/null
if [ "$tests_rc" -eq 0 ] && [ "$cli_rc" -eq 0 ] && [ "$val_rc" -eq 0 ]; then fail=0; else fail=1; fi
echo "=== OVERALL: $([ "$fail" -eq 0 ] && echo PASS || echo FAIL) (exit $fail) ==="
exit "$fail"
