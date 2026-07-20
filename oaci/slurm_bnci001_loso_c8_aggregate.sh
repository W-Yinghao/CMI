#!/usr/bin/env bash
# BNCI2014_001 LOSO seeds-[0,1,2] C8 MULTI-SEED aggregation (post-sweep, CPU): read all 27 committed fold
# artifacts (deep-verify), report native K1 (per fold/level/seed + counts) + the real multi-seed K2, and
# write oaci/reports/C8_BNCI001_LOSO_SEEDS012_K1K2.{json,md}. Run ONLY after all 27 Phase-B complete. Submit:
#   OACI_LOSO_ROOT=/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012 \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out \
#       oaci/slurm_bnci001_loso_c8_aggregate.sh
# Per cluster policy: NO --time. CPU partition.
#SBATCH --job-name=oaci-c8-aggregate
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
: "${OACI_LOSO_ROOT:?C8 aggregation requires OACI_LOSO_ROOT (the sweep root holding seed-<s>/target-00N/)}"
# This run's Phase-B spanned TWO commits (execution-only: staged_b.sh 16->32 CPU + controller --leakage-jobs).
# The narrow approved-provenance guard accepts ONLY these two provenance groups, and ONLY when every
# scientific-identity invariant matches (schema/probe/K1/K2 config, deep-verify, target_fit empty). The two
# groups correspond to commits 7931091 (pre-change, 11 folds) and a1a09b8 (32w change, 16 folds).
APPROVED="${OACI_APPROVED_PROV:-1b8310930b1b8d3a006207a0443c59946b5b816197d25d42d13664e87a35189f,39b1287a81d04af9e492dc6840fabb0b16e21aa5b68dd10c81dc0b79cbeabdbd}"
COMMITS="${OACI_TRANSITION_COMMITS:-7931091,a1a09b8}"
echo "[c8-aggregate] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) root=$OACI_LOSO_ROOT"
$PY -m oaci.confirmatory.c8_aggregate --loso-root "$OACI_LOSO_ROOT" --seeds "${OACI_SEEDS:-0,1,2}" \
    --approved-provenance "$APPROVED" --transition-commits "$COMMITS" \
    --out-json oaci/reports/C8_BNCI001_LOSO_SEEDS012_K1K2.json \
    --out-md   oaci/reports/C8_BNCI001_LOSO_SEEDS012_K1K2.md
rc=$?
echo "=== OVERALL: $([ "$rc" -eq 0 ] && echo PASS || echo FAIL) (exit $rc) ==="
exit "$rc"
