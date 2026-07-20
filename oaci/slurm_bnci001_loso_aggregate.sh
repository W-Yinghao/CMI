#!/usr/bin/env bash
# BNCI2014_001 LOSO seed-0 AGGREGATION (C6, post-sweep, CPU): read all nine committed fold artifacts
# (deep-verify each), check one protocol family + one code provenance + all target_fit empty, compute the
# descriptive k1/k2 aggregates, and write oaci/reports/C6_BNCI001_LOSO_SEED0.{json,md}. Run ONLY after all
# 9 Phase-B jobs have completed. Submit:
#   OACI_LOSO_ROOT=/projects/EEG-foundation-model/yinghao/oaci-loso-seed0 \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out \
#       oaci/slurm_bnci001_loso_aggregate.sh
# Per cluster policy: NO --time. CPU partition.
#SBATCH --job-name=oaci-loso-aggregate
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
: "${OACI_LOSO_ROOT:?Phase C aggregation requires OACI_LOSO_ROOT (the sweep root holding target-00N/)}"
echo "[loso-aggregate] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) root=$OACI_LOSO_ROOT"
$PY -m oaci.confirmatory.aggregate --loso-root "$OACI_LOSO_ROOT" \
    --out-json oaci/reports/C6_BNCI001_LOSO_SEED0.json \
    --out-md   oaci/reports/C6_BNCI001_LOSO_SEED0.md \
    ${OACI_PROTOCOL_FAMILY:+--protocol-family "$OACI_PROTOCOL_FAMILY"} \
    ${OACI_PROVENANCE_HASH:+--provenance-hash "$OACI_PROVENANCE_HASH"}
rc=$?
echo "=== OVERALL: $([ "$rc" -eq 0 ] && echo PASS || echo FAIL) (exit $rc) ==="
exit "$rc"
