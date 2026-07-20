#!/usr/bin/env bash
# C19 pre-registered source-only competence probe (CPU, DIAGNOSTIC-ONLY, no retrain). Robust-core + endpoint-
# augmented LOTO probes across S0/S2/S3 from the committed C18 replay artifacts; leakage recompute parallelised
# across cpus-per-task. Submit:
#   OACI_EXTRACT_DIR=/projects/EEG-foundation-model/yinghao/oaci-c18-extract \
#     OACI_C10_DIR=/projects/EEG-foundation-model/yinghao/oaci-c10-replay \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out oaci/slurm_c19_probe.sh
# Per cluster policy: NO --time.
#SBATCH --job-name=oaci-c19-probe
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
NW="${OACI_N_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
echo "[c19-probe] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) workers=$NW"
$PY -m oaci.competence_probe.report --extract-dir "$OACI_EXTRACT_DIR" --c10-dir "$OACI_C10_DIR" \
    --out-dir oaci/reports --n-workers "$NW"
rc=$?
echo "=== C19 probe rc=$rc ==="
exit "$rc"
