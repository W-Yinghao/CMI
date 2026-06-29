#!/bin/bash
# csc CSC-P1.5 DEVELOPMENT difficulty-envelope — CPU-only, PARALLEL (joblib over independent
# (cell,cluster) tasks; result identical to serial). Submit with:
#   sbatch csc/run_envelope_p15.sbatch canary csc/results/envelope_p15_canary.json
#   sbatch csc/run_envelope_p15.sbatch full   csc/results/envelope_p15_dev_c12.json   csc/results/envelope_p15_canary.json
# Conventions (cmi-compute-env): CPU partition, NO --time (inherit partition max), full-path conda
# in the batch shell, no nested srun. DEVELOPMENT sweep: results MAY NOT select params / define the
# operating region / seed a confirmatory claim.
#SBATCH --job-name=csc_p15
#SBATCH --partition=CPU
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI
#SBATCH --output=logs/csc_p15_%j.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
set -e
PHASE="${1:-both}"
OUT="${2:-csc/results/envelope_p15.json}"
CANARY_REF="${3:-}"
# BLAS single-threaded: joblib runs one cluster per core, so threaded BLAS would oversubscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONWARNINGS=ignore
JOBS=${SLURM_CPUS_PER_TASK:-8}
echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK jobs=$JOBS phase=$PHASE out=$OUT started=$(date -Is)"
mkdir -p csc/results
RC=0
EXTRA=""
[ -n "$CANARY_REF" ] && EXTRA="--canary_ref $CANARY_REF"
/home/infres/yinwang/anaconda3/bin/conda run -n icml --no-capture-output \
    python -u -m csc.run_envelope_p15 --phase "$PHASE" --out "$OUT" --jobs "$JOBS" $EXTRA \
    || RC=$?
echo "finished=$(date -Is) p15_rc=$RC"
# committable stdout provenance: copy the run log + record its SHA256 next to the artifact
LOG="logs/csc_p15_${SLURM_JOB_ID}.out"
BASE="${OUT%.json}"
cp "$LOG" "${BASE}.log" 2>/dev/null || true
sha256sum "${BASE}.log" > "${BASE}.log.sha256" 2>/dev/null || true
echo "stdout_sha256=$(cut -d' ' -f1 ${BASE}.log.sha256 2>/dev/null)"
exit $RC          # fail-closed: nonzero if preflight/canary failed or the driver errored
