#!/bin/bash
#SBATCH --job-name=b3confirm
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_csc_b3_frozen
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_csc_b3_frozen/csc/results/b3_confirmatory.slurm.%j.out
# CSC Route B3 unseen-cluster confirmatory -- FROZEN-WORKTREE, FAIL-CLOSED wrapper (A-line pattern).
# Runs ONLY from a git worktree checked out at tag csc-b3-confirmatory-v1. This wrapper is part of the
# freeze package; it is NOT invoked until the reviewer authorizes tag creation + the run. Infra errors exit
# 2; a scientific FAIL still writes + preserves the artifact and exits 0 (verdict is inside the JSON).
# NEVER touches real EEG; NEVER touches the A tag csc-confirmatory-v1 / dee8958.
set -uo pipefail
TAG="csc-b3-confirmatory-v1"
FROZEN="/home/infres/yinwang/CMI_AAAI_csc_b3_frozen"
OUT="$FROZEN/csc/results/b3_confirmatory_result.json"
TMP_OUT="$FROZEN/csc/results/b3_confirmatory_result.${SLURM_JOB_ID:-manual}.tmp.json"

infra_fail() { echo "[b3confirm] INFRA-FAIL: $1" >&2; exit 2; }

cd "$FROZEN" 2>/dev/null || infra_fail "frozen worktree $FROZEN missing (create: git worktree add $FROZEN $TAG)"

# 1. frozen-code provenance: HEAD must be the tag commit, tree clean
command -v git >/dev/null || infra_fail "git missing"
TAGC=$(git rev-parse "${TAG}^{commit}" 2>/dev/null) || infra_fail "tag $TAG does not exist"
HEADC=$(git rev-parse HEAD 2>/dev/null) || infra_fail "no HEAD"
[ "$TAGC" = "$HEADC" ] || infra_fail "HEAD $HEADC != $TAG $TAGC"
[ -z "$(git status --porcelain)" ] || infra_fail "working tree not clean"

# 2. remove any stale artifact so we cannot mistake an old result for a fresh one
rm -f "$OUT" "$TMP_OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$FROZEN"
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate icml
echo "[b3confirm] host=$(hostname) job=${SLURM_JOB_ID:-na} HEAD=$HEADC tag=$TAG cpus=${SLURM_CPUS_PER_TASK:-na}"

# 3. guarded execute -> temp artifact (runner re-verifies manifest hash + code hashes + HEAD==tag + seeds)
python -u -m csc.mininfo.run_b3_confirmatory --execute --jobs "${SLURM_CPUS_PER_TASK:-24}" --out "$TMP_OUT"
RC=$?
[ "$RC" -eq 2 ] && infra_fail "runner refused (fail-closed provenance) rc=2"
[ "$RC" -ne 0 ] && infra_fail "runner crashed rc=$RC"

# 4. freshness: the temp artifact must exist and belong to THIS job
[ -f "$TMP_OUT" ] || infra_fail "no fresh artifact produced"
python -c "import json,sys; d=json.load(open('$TMP_OUT')); sys.exit(0 if d.get('verdict') and d.get('manifest_hash') else 1)" \
  || infra_fail "artifact missing verdict/manifest_hash"
mv "$TMP_OUT" "$OUT"

# 5. integrity hashes of the preserved artifact + log
sha256sum "$OUT" | tee "$OUT.sha256"
echo "[b3confirm] done rc=0 (scientific verdict is inside $OUT; C6 red-team still required)"
exit 0
