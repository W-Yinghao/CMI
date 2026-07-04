#!/bin/bash
#SBATCH --job-name=realeeg_val_v2
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_csc_realeeg_v2_frozen
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_csc_realeeg_v2_frozen/csc/results/realeeg_validation_v2.slurm.%j.out
# CSC real-EEG validation v2 -- FROZEN-WORKTREE, FAIL-CLOSED, PARALLEL, CHECKPOINTED wrapper. Runs ONLY from a
# git worktree checked out at tag csc-realeeg-v2. v2 is a PERFORMANCE-ONLY successor to the serial v1 (which was
# infrastructure-infeasible: ~6.3 days serial > the 5-day cpu-high wall, end-only write -> killed, no artifact).
# Same science; parallel over the 801 independent cohorts + canonical sort before verdict + streaming checkpoint.
# Infra errors exit 2; a scientific FAIL still writes + preserves the artifact and exits 0 (verdict inside JSON).
# NEVER touches the synthetic tags dee8958 / 0595f64. NO clinical/PD claim.
# Resume a killed run: submit with  RESUME=1 sbatch ...  (skips the fresh-wipe, passes --resume; same frozen inputs only).
set -uo pipefail
TAG="csc-realeeg-v2"
FROZEN="/home/infres/yinwang/CMI_AAAI_csc_realeeg_v2_frozen"
OUT="$FROZEN/csc/results/realeeg_validation_v2.final.json"
TMP_OUT="$FROZEN/csc/results/realeeg_validation_v2.final.${SLURM_JOB_ID:-manual}.tmp.json"
PARTIAL="$FROZEN/csc/results/realeeg_validation_v2.partial.jsonl"
CHECKPOINT="$FROZEN/csc/results/realeeg_validation_v2.checkpoint.json"
CACHE="/home/infres/yinwang/realeeg_feas/cache/LEE2019_B3.npz"
RESUME="${RESUME:-0}"

infra_fail() { echo "[realeeg-v2] INFRA-FAIL: $1" >&2; exit 2; }

cd "$FROZEN" 2>/dev/null || infra_fail "frozen worktree $FROZEN missing (create: git worktree add --detach $FROZEN $TAG)"

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

# 3. fresh run wipes any stale artifact/partial/checkpoint; a resume KEEPS them (runner re-verifies binding)
mkdir -p "$FROZEN/csc/results"
rm -f "$OUT" "$TMP_OUT"
if [ "$RESUME" != "1" ]; then rm -f "$PARTIAL" "$CHECKPOINT"; fi

# 4. thread-pin BLAS to 1 (parallelism is across cohorts, not within BLAS) to avoid oversubscription
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$FROZEN"
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate icml
JOBS=$(( ${SLURM_CPUS_PER_TASK:-24} < 24 ? ${SLURM_CPUS_PER_TASK:-24} : 24 ))
RESUME_FLAG=""; [ "$RESUME" = "1" ] && RESUME_FLAG="--resume"
echo "[realeeg-v2] host=$(hostname) job=${SLURM_JOB_ID:-na} HEAD=$HEADC tag=$TAG jobs=$JOBS resume=$RESUME"

# 5. guarded parallel execute -> temp artifact (runner re-verifies HEAD==tag + clean + method/cache/engine hashes;
#    streams partial+checkpoint; a missing/duplicate/worker-errored cohort exits 2 with NO endpoint evaluated)
python -u -m csc.mininfo.run_realeeg_validation --execute --jobs "$JOBS" $RESUME_FLAG \
       --out "$TMP_OUT" --partial "$PARTIAL" --checkpoint "$CHECKPOINT"
RC=$?
[ "$RC" -eq 2 ] && infra_fail "runner refused / infra failure (fail-closed) rc=2"
[ "$RC" -ne 0 ] && infra_fail "runner crashed rc=$RC"

# 6. freshness + provenance: the temp artifact must exist, be v2-parallel, complete (801), and match this HEAD/tag
[ -f "$TMP_OUT" ] || infra_fail "no fresh artifact produced"
HEADC="$HEADC" python - "$TMP_OUT" "$FROZEN" <<'PYCHK' || infra_fail "artifact provenance/freshness check failed"
import json, os, sys, hashlib
art = json.load(open(sys.argv[1])); frozen = sys.argv[2]
fr = art.get("frozen_refs", {}); mp = art.get("manifest_provenance", {}); ex = art.get("execution", {})
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
bankp = os.path.join(frozen, "csc/mininfo/realeeg_bank_manifest.json")
ok = (art.get("verdict") and art.get("per_cohort") is not None
      and art.get("schema_version") == "v2"
      and ex.get("mode") == "cohort_parallel" and ex.get("performance_only_change") is True
      and ex.get("n_tasks") == 801 and ex.get("n_tasks_expected") == 801
      and isinstance(ex.get("task_table_sha256"), str) and len(ex.get("task_table_sha256")) == 64
      and len(art.get("per_cohort")) == 801
      and fr.get("git_status_clean") is True
      and fr.get("git_head") == os.environ.get("HEADC")
      and fr.get("expected_code_ref") == "refs/tags/csc-realeeg-v2"
      and fr.get("synthetic_tags_untouched") is True
      and art.get("base_seed") == 20000000
      and mp.get("bank_manifest_sha256") == sha(bankp))
sys.exit(0 if ok else 1)
PYCHK
mv "$TMP_OUT" "$OUT"

# 7. integrity hash of the preserved artifact
sha256sum "$OUT" | tee "$OUT.sha256"
echo "[realeeg-v2] done rc=0 (verdict inside $OUT; TIER1 B3 safety gates PASS/FAIL; red-team still required)"
exit 0
