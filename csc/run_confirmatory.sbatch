#!/bin/bash
# csc CSC CONFIRMATORY run — CPU-only, parallel (joblib, BLAS=1). SEPARATELY AUTHORIZED ONLY.
#
#   *** DO NOT SUBMIT until the reviewer authorizes the confirmatory run. ***
#
# This runs `python -m csc.run_confirmatory --execute`, which generates the UNSEEN core clusters at
# base_seed=900000 and evaluates the two frozen endpoints. A scientific FAIL (runner exit 1) is a
# VALID outcome: this wrapper ALWAYS preserves the JSON artifact + sha256 (and the log + sha256), then
# exits with the SCIENTIFIC return code. If NO JSON artifact is produced it exits with an
# INFRASTRUCTURE code (2) so a crash is never confused with a scientific FAIL.
#
# The run MUST execute on the FROZEN tagged commit (refs/tags/csc-confirmatory-v1). Before the
# authorized run, create a DETACHED worktree AT THE TAG and submit from there:
#   git worktree add --detach /home/infres/yinwang/CMI_AAAI_csc_frozen csc-confirmatory-v1
#   sbatch --chdir=/home/infres/yinwang/CMI_AAAI_csc_frozen \
#          /home/infres/yinwang/CMI_AAAI_csc_frozen/csc/run_confirmatory.sbatch
# --chdir defaults to the frozen worktree (NOT the main repo dir, which may be on another branch); the
# in-script git guard below fail-closes (exit 2) unless HEAD == the frozen tag AND the tree is clean,
# so a wrong cwd/branch consumes NO unseen seeds.
#SBATCH --job-name=csc_confirm
#SBATCH --partition=CPU
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_csc_frozen
#SBATCH --output=logs/csc_confirm_%j.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
set -u
FROZEN_TAG="csc-confirmatory-v1"
WORKDIR="${CSC_FROZEN_DIR:-/home/infres/yinwang/CMI_AAAI_csc_frozen}"
OUT="${1:-csc/results/confirmatory.json}"
cd "$WORKDIR" || { echo "INFRA: cannot cd to frozen worktree $WORKDIR"; exit 2; }
# --- frozen-code git guard (BEFORE any compute; fail-closed exit 2 = infrastructure, no seeds spent) ---
HEAD=$(git rev-parse HEAD 2>/dev/null) || { echo "INFRA: not a git checkout"; exit 2; }
EXPECT=$(git rev-parse "${FROZEN_TAG}^{commit}" 2>/dev/null) || { echo "INFRA: tag $FROZEN_TAG not found"; exit 2; }
echo "git_head=$HEAD frozen_tag=$FROZEN_TAG expected=$EXPECT workdir=$WORKDIR"
[ "$HEAD" = "$EXPECT" ] || { echo "INFRA: HEAD($HEAD) != frozen tag $FROZEN_TAG ($EXPECT)"; exit 2; }
git diff --quiet HEAD -- || { echo "INFRA: working tree is DIRTY"; exit 2; }
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONWARNINGS=ignore
JOBS=${SLURM_CPUS_PER_TASK:-8}
LOG="logs/csc_confirm_${SLURM_JOB_ID}.out"
BASE="${OUT%.json}"
echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK jobs=$JOBS out=$OUT started=$(date -Is)"
mkdir -p csc/results logs
SCI_RC=0
/home/infres/yinwang/anaconda3/bin/conda run -n icml --no-capture-output \
    python -u -m csc.run_confirmatory --execute --jobs "$JOBS" --out "$OUT" \
    || SCI_RC=$?
echo "scientific_rc=$SCI_RC finished=$(date -Is)"
# ALWAYS preserve artifacts (a scientific FAIL must not become a missing artifact)
cp "$LOG" "${BASE}.log" 2>/dev/null || true
sha256sum "${BASE}.log" > "${BASE}.log.sha256" 2>/dev/null || true
if [ -f "$OUT" ]; then
    sha256sum "$OUT" > "${OUT}.sha256" 2>/dev/null || true
    echo "artifact_present=yes json_sha256=$(cut -d' ' -f1 ${OUT}.sha256 2>/dev/null)"
    echo "EXIT scientific_rc=$SCI_RC (0=PASS, 1=scientific FAIL; both valid, artifact preserved)"
    exit $SCI_RC
else
    echo "artifact_present=NO -> INFRASTRUCTURE FAILURE (no scientific result produced)"
    exit 2
fi
