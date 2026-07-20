#!/usr/bin/env bash
# C37 exact selector trace recovery: CPU-bound leakage-UCL replay from Phase-A
# source-train feature stores. No training, no model forward, no target labels
# for UCL replay. Per cluster policy: NO --time/walltime.
# Submit from worktree root:
#   OACI_C37_MODE=make-worklist bash oaci/slurm_c37_trace_recovery.sh
#   OACI_C37_MODE=worker OACI_C37_KIND=selected sbatch --array=0-2%3 oaci/slurm_c37_trace_recovery.sh
#   OACI_C37_MODE=worker OACI_C37_KIND=better sbatch --array=0-37%4 oaci/slurm_c37_trace_recovery.sh
#   OACI_C37_MODE=aggregate sbatch oaci/slurm_c37_trace_recovery.sh
# Optional:
#   OACI_C37_LEAKAGE_JOBS=36 OACI_C37_MODE=p0-only sbatch oaci/slurm_c37_trace_recovery.sh
#   OACI_C37_LEAKAGE_JOBS=48 OACI_C37_MODE=full sbatch oaci/slurm_c37_trace_recovery.sh
#SBATCH --job-name=oaci-c37-trace
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --output=logs/oaci-c37-trace-%A_%a.out
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE
export PYTHONHASHSEED=0

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

JOBS="${OACI_C37_LEAKAGE_JOBS:-${SLURM_CPUS_PER_TASK:-48}}"
MODE="${OACI_C37_MODE:-worker}"
KIND="${OACI_C37_KIND:-selected}"
INDEX="${OACI_C37_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
WORK_DIR="${OACI_C37_WORK_DIR:-/projects/EEG-foundation-model/yinghao/oaci-c37-trace-recovery/${COMMIT}}"

echo "[c37-trace] node=$(hostname) partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK:-?} jobs=$JOBS mode=$MODE kind=$KIND index=$INDEX commit=$COMMIT"
echo "[c37-trace] python=$PY"
echo "[c37-trace] work_dir=$WORK_DIR"

if [ "$MODE" = "make-worklist" ]; then
  "$PY" -m oaci.selector_trace_recovery.report --make-worklist --work-dir "$WORK_DIR" --n-jobs "$JOBS"
elif [ "$MODE" = "worker" ]; then
  "$PY" -m oaci.selector_trace_recovery.report --worker --work-dir "$WORK_DIR" --kind "$KIND" --index "$INDEX" --n-jobs "$JOBS"
elif [ "$MODE" = "aggregate" ]; then
  "$PY" -m oaci.selector_trace_recovery.report --aggregate --work-dir "$WORK_DIR" --n-jobs "$JOBS"
elif [ "$MODE" = "p0-only" ]; then
  "$PY" -m oaci.selector_trace_recovery.report --p0-only --n-jobs "$JOBS"
elif [ "$MODE" = "full" ]; then
  "$PY" -m oaci.selector_trace_recovery.report --n-jobs "$JOBS"
else
  echo "unknown OACI_C37_MODE=$MODE (expected make-worklist|worker|aggregate|full|p0-only)" >&2
  exit 2
fi

echo "=== C37 trace recovery complete ==="
