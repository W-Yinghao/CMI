#!/bin/bash
#SBATCH --job-name=cmitrace-theory
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/theory_spectrum/%x-%j.out
#SBATCH --error=logs/theory_spectrum/%x-%j.out
# CPU-only fleet for the theory-spectrum experiments (E1/E2/E3) over BANKED frozen artifacts — no GPU, no
# retrain. Cluster convention: no --qos and no --time (default QOS, no walltime cap). Resumable: every runner
# skips a cell whose per-cell JSON already exists. Address cells by real (dataset,method,seed,fold) id.
#
# Parameters via env:
#   EXP  = e1 | e2 | e3
#   DS   = BNCI2014_001 | BNCI2015_001         (E1)
#   BB   = EEGNet | TSMNet                      (E2)
#   SEEDS, KSPEC, NPERM                          (optional overrides)
#
# Launch examples (DO NOT run until owner GO):
#   sbatch --export=ALL,EXP=e1,DS=BNCI2014_001 scripts/sbatch_theory_spectrum.sh
#   sbatch --export=ALL,EXP=e1,DS=BNCI2015_001 scripts/sbatch_theory_spectrum.sh
#   sbatch --export=ALL,EXP=e2,BB=TSMNet       scripts/sbatch_theory_spectrum.sh
#   sbatch --export=ALL,EXP=e2,BB=EEGNet       scripts/sbatch_theory_spectrum.sh
#   sbatch --export=ALL,EXP=e3                 scripts/sbatch_theory_spectrum.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/theory_spectrum
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)"
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python

EXP="${EXP:?set EXP=e1|e2|e3}"
SEEDS="${SEEDS:-0 1 2}"
KSPEC="${KSPEC:-16}"; NPERM="${NPERM:-50}"
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) EXP=${EXP}"

case "$EXP" in
  e1) "$PY" scripts/run_e1_spectrum.py --dataset "${DS:?set DS}" --seeds $SEEDS --k_spec "$KSPEC" --n_perm "$NPERM" ;;
  e2) "$PY" scripts/run_e2_rank_threshold.py --backbone "${BB:?set BB}" --seeds $SEEDS --n_perm "$NPERM" ;;
  e3) "$PY" scripts/run_e3_kstar.py --seeds 0 1 2 3 4 --spur_strengths 2.0 3.0 4.0 ;;
  all)
    # ONE job (1 submit slot) running the whole E1+E2 fleet 8-way parallel across cores. Per-cell
    # skip-if-done makes it resumable. intra-op threads=1 to avoid oversubscription across the 8 workers.
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    pids=()
    for ds in BNCI2014_001 BNCI2015_001; do for s in 0 1 2; do
      "$PY" scripts/run_e1_spectrum.py --dataset "$ds" --seeds "$s" --k_spec "$KSPEC" --n_perm "$NPERM" \
        > "logs/theory_spectrum/e1_${ds}_s${s}.log" 2>&1 & pids+=($!)
    done; done
    for bb in TSMNet EEGNet; do
      "$PY" scripts/run_e2_rank_threshold.py --backbone "$bb" --seeds 0 1 2 --n_perm "$NPERM" \
        > "logs/theory_spectrum/e2_${bb}.log" 2>&1 & pids+=($!)
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    echo "fleet workers done (rc=$rc); running guarded E1 aggregate"
    "$PY" scripts/aggregate_e1_spectrum.py || true
    exit $rc ;;
  *) echo "unknown EXP=$EXP" >&2; exit 2 ;;
esac
echo "done EXP=${EXP}"
