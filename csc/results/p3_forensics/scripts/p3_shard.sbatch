#!/bin/bash
#SBATCH --job-name=p3_forensic
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/infres/yinwang/realeeg_feas/p3_forensic/logs/shard_%j.out
# CSC-realEEG-P3.0b/c certifier-internal forensic -- ONE shard (BASE x COND = 100 cohorts). DEVELOPMENT
# forensic only (NO method change, NO freeze, NOT confirmatory). Worker exit 2 (worker error) => infra failure.
set -uo pipefail
: "${BASE:?need BASE}"; : "${COND:?need COND}"
OUTDIR=/home/infres/yinwang/realeeg_feas/p3_forensic
OUT="$OUTDIR/shard_${BASE}_${COND}.jsonl"
mkdir -p "$OUTDIR/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate icml
echo "[p3 shard] host=$(hostname) job=${SLURM_JOB_ID:-na} base=$BASE cond=$COND cpus=${SLURM_CPUS_PER_TASK:-8}"
python -u /home/infres/yinwang/realeeg_feas/realeeg_internal_forensic.py \
    --base "$BASE" --condition "$COND" --jobs "${SLURM_CPUS_PER_TASK:-8}" --n 100 --out "$OUT"
RC=$?
echo "[p3 shard base=$BASE cond=$COND] rc=$RC"
exit $RC
