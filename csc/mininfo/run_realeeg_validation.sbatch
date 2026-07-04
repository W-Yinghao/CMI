#!/bin/bash
#SBATCH --job-name=realeeg_val
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_csc_realeeg_frozen
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_csc_realeeg_frozen/csc/results/realeeg_validation.slurm.%j.out
# CSC real-EEG validation -- FROZEN-WORKTREE, FAIL-CLOSED wrapper (A/B3-line pattern). Runs ONLY from a git
# worktree checked out at tag csc-realeeg-v1. NOT invoked until the reviewer authorizes tag creation + a run.
# Infra errors exit 2; a scientific FAIL still writes + preserves the artifact and exits 0 (verdict inside JSON).
# NEVER touches the synthetic tags dee8958 / 0595f64. NO clinical/PD claim.
set -uo pipefail
TAG="csc-realeeg-v1"
FROZEN="/home/infres/yinwang/CMI_AAAI_csc_realeeg_frozen"
OUT="$FROZEN/csc/results/realeeg_validation_result.json"
TMP_OUT="$FROZEN/csc/results/realeeg_validation_result.${SLURM_JOB_ID:-manual}.tmp.json"
CACHE="/home/infres/yinwang/realeeg_feas/cache/LEE2019_B3.npz"

infra_fail() { echo "[realeeg] INFRA-FAIL: $1" >&2; exit 2; }

cd "$FROZEN" 2>/dev/null || infra_fail "frozen worktree $FROZEN missing (create: git worktree add $FROZEN $TAG)"

# 1. frozen-code provenance: HEAD must be the tag commit, tree clean
command -v git >/dev/null || infra_fail "git missing"
TAGC=$(git rev-parse "${TAG}^{commit}" 2>/dev/null) || infra_fail "tag $TAG does not exist"
HEADC=$(git rev-parse HEAD 2>/dev/null) || infra_fail "no HEAD"
[ "$TAGC" = "$HEADC" ] || infra_fail "HEAD $HEADC != $TAG $TAGC"
[ -z "$(git status --porcelain)" ] || infra_fail "working tree not clean"

# 2. cache present + sha256 matches the frozen cache manifest
[ -f "$CACHE" ] || infra_fail "cache $CACHE missing (rebuild with build_lee2019_b3_cache.py)"
WANT=$(python -c "import json;print(json.load(open('$FROZEN/csc/mininfo/realeeg_lee2019_cache_manifest.json'))['provenance']['cache_sha256'])")
GOT=$(sha256sum "$CACHE" | cut -d' ' -f1)
[ "$WANT" = "$GOT" ] || infra_fail "cache sha256 mismatch ($GOT != $WANT)"

# 3. remove any stale artifact so we cannot mistake an old result for a fresh one
rm -f "$OUT" "$TMP_OUT"
mkdir -p "$FROZEN/csc/results"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$FROZEN"
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate icml
echo "[realeeg] host=$(hostname) job=${SLURM_JOB_ID:-na} HEAD=$HEADC tag=$TAG"

# 4. guarded execute -> temp artifact (runner re-verifies HEAD==tag + clean + method/cache hashes internally)
python -u -m csc.mininfo.run_realeeg_validation --execute --jobs "${SLURM_CPUS_PER_TASK:-24}" --out "$TMP_OUT"
RC=$?
[ "$RC" -eq 2 ] && infra_fail "runner refused (fail-closed provenance) rc=2"
[ "$RC" -ne 0 ] && infra_fail "runner crashed rc=$RC"

# 5. freshness + provenance: the temp artifact must exist and match this HEAD/tag
[ -f "$TMP_OUT" ] || infra_fail "no fresh artifact produced"
HEADC="$HEADC" python - "$TMP_OUT" <<'PYCHK' || infra_fail "artifact provenance/freshness check failed"
import json, os, sys
art = json.load(open(sys.argv[1])); cp = art.get("code_provenance", {})
ok = (art.get("verdict") and cp.get("git_status_clean") is True
      and cp.get("git_head") == os.environ.get("HEADC") and cp.get("tag") == "csc-realeeg-v1")
sys.exit(0 if ok else 1)
PYCHK
mv "$TMP_OUT" "$OUT"

# 6. integrity hash of the preserved artifact
sha256sum "$OUT" | tee "$OUT.sha256"
echo "[realeeg] done rc=0 (verdict inside $OUT; TIER1 B3 safety gates PASS/FAIL; red-team still required)"
exit 0
