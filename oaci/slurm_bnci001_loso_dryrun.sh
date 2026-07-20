#!/usr/bin/env bash
# BNCI2014_001 LOSO seed-0 DRY-RUN on a CPU node (C6a): materialize all 9 fold manifests + build each fold
# on real data (NO training, NO GPU) and verify roles / support tables / target-not-fit. Submit:
#   OACI_DATALAKE_ROOT=/projects/EEG-foundation-model/datalake/raw \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out \
#       oaci/slurm_bnci001_loso_dryrun.sh
# Per cluster policy: NO --time. CPU partition.
#SBATCH --job-name=oaci-loso-dryrun
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -u
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
: "${OACI_OUT_ROOT:=/projects/EEG-foundation-model/yinghao/oaci-loso-dryrun/${SLURM_JOB_ID:-local}}"
mkdir -p "$OACI_OUT_ROOT"
echo "[loso-dryrun] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null)"
$PY -m oaci.confirmatory.bnci001_loso --protocol oaci/protocol/confirmatory_v2.yaml \
    --datalake-root "$OACI_DATALAKE_ROOT" --out-dir "$OACI_OUT_ROOT/manifests" --bootstrap-mode full \
    >"$OACI_OUT_ROOT/loso-dryrun.json" 2>"$OACI_OUT_ROOT/loso-dryrun.err"
rc=$?
echo "=== dry-run rc=$rc ==="
if [ "$rc" -eq 0 ]; then
  $PY -c "import json;r=json.load(open('$OACI_OUT_ROOT/loso-dryrun.json'));print('all_ok',r['all_ok'],'n_folds',r['n_folds'],'unique_manifests',r['unique_manifest_hashes']);[print(' ',f['fold_id'],'ok',f['ok'],f['role_counts'],'L1zeros',f['level1_zeros']) for f in r['folds']]"
else
  tail -25 "$OACI_OUT_ROOT/loso-dryrun.err"
fi
echo "=== OVERALL: $([ "$rc" -eq 0 ] && echo PASS || echo FAIL) (exit $rc) ==="
exit "$rc"
