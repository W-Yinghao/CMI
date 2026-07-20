#!/usr/bin/env bash
# C18 report (CPU): assemble the Controlled Support-Mismatch x Identifiability Stress Test from the persisted
# GPU extraction. Identity gates (S0 reproduces C17 0.602) + per-regime mask-stress probe (H2/H3) + boundary
# S6-vs-S7 (H4) + leakage estimability/abstention (H5) + observability-dropout appendix (C18-D) + severity +
# taxonomy + tables. Leakage recompute (dominant cost) is parallelised across cpus-per-task. Submit:
#   OACI_EXTRACT_DIR=/projects/EEG-foundation-model/yinghao/oaci-c18-extract \
#     OACI_C10_DIR=/projects/EEG-foundation-model/yinghao/oaci-c10-replay \
#     OACI_LOSO_ROOT=/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012 \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out oaci/slurm_c18_report.sh
# Per cluster policy: NO --time.
#SBATCH --job-name=oaci-c18-report
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
: "${OACI_EXTRACT_DIR:?requires OACI_EXTRACT_DIR}"
: "${OACI_C10_DIR:?requires OACI_C10_DIR}"
: "${OACI_LOSO_ROOT:?requires OACI_LOSO_ROOT}"
NW="${OACI_N_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
echo "[c18-report] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) workers=$NW"
$PY -m oaci.support_stress.report --extract-dir "$OACI_EXTRACT_DIR" --c10-dir "$OACI_C10_DIR" \
    --loso-root "$OACI_LOSO_ROOT" --out-dir oaci/reports --n-workers "$NW"
rc=$?
echo "=== C18 report rc=$rc ==="
exit "$rc"
