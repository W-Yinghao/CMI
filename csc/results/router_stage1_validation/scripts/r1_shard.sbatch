#!/bin/bash
#SBATCH --job-name=r1_router
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/infres/yinwang/realeeg_feas/router_stage1/logs/r1_%j.out
set -uo pipefail
: "${BASE:?}"; : "${COND:?}"; : "${START:?}"; : "${N:?}"; : "${PHASE:?}"
OUTDIR=/home/infres/yinwang/realeeg_feas/router_stage1/${PHASE}
OUT="$OUTDIR/r1_${BASE}_${COND}_${START}.jsonl"; mkdir -p "$OUTDIR" /home/infres/yinwang/realeeg_feas/router_stage1/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[r1 shard] host=$(hostname) job=${SLURM_JOB_ID:-na} $PHASE $BASE:$COND:$START+$N"
python -u /home/infres/yinwang/realeeg_feas/realeeg_r1_forensic.py --base "$BASE" --condition "$COND" --start "$START" --n "$N" --jobs "${SLURM_CPUS_PER_TASK:-8}" --out "$OUT"
RC=$?; echo "[r1 $PHASE $BASE:$COND:$START] rc=$RC"; exit $RC
