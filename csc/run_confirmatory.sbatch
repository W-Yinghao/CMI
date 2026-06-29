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
# The run MUST execute on the FROZEN tagged commit. The main repo dir may be on another branch, so
# submit from the frozen checkout, e.g.:
#   sbatch --chdir=/home/infres/yinwang/CMI_AAAI_csc /home/infres/yinwang/CMI_AAAI_csc/csc/run_confirmatory.sbatch
#SBATCH --job-name=csc_confirm
#SBATCH --partition=CPU
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI
#SBATCH --output=logs/csc_confirm_%j.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
set -u
OUT="${1:-csc/results/confirmatory.json}"
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
