#!/bin/bash
# csc CSC-P1.5 DEVELOPMENT difficulty-envelope — CPU-only. Submit with:
#   sbatch csc/run_envelope_p15.sbatch canary csc/results/envelope_p15_canary.json
#   sbatch csc/run_envelope_p15.sbatch full   csc/results/envelope_p15_dev_c12.json   # after canary passes
# Conventions (cmi-compute-env): CPU partition, NO --time (inherit partition max), full-path conda
# in the batch shell, no nested srun. DEVELOPMENT sweep: results MAY NOT select params / define the
# operating region / seed a confirmatory claim.
#SBATCH --job-name=csc_p15
#SBATCH --partition=CPU
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI
#SBATCH --output=logs/csc_p15_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -e
PHASE="${1:-both}"
OUT="${2:-csc/results/envelope_p15.json}"
CANARY_REF="${3:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONWARNINGS=ignore
echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK phase=$PHASE out=$OUT started=$(date -Is)"
mkdir -p csc/results
RC=0
EXTRA=""
[ -n "$CANARY_REF" ] && EXTRA="--canary_ref $CANARY_REF"
/home/infres/yinwang/anaconda3/bin/conda run -n icml --no-capture-output \
    python -m csc.run_envelope_p15 --phase "$PHASE" --out "$OUT" $EXTRA \
    || RC=$?
echo "finished=$(date -Is) p15_rc=$RC"
# committable stdout provenance: copy the run log + record its SHA256 next to the artifact
LOG="logs/csc_p15_${SLURM_JOB_ID}.out"
BASE="${OUT%.json}"
cp "$LOG" "${BASE}.log" 2>/dev/null || true
sha256sum "${BASE}.log" > "${BASE}.log.sha256" 2>/dev/null || true
echo "stdout_sha256=$(cut -d' ' -f1 ${BASE}.log.sha256 2>/dev/null)"
exit $RC          # fail-closed: nonzero if preflight/canary failed or the driver errored
