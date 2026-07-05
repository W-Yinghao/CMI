#!/bin/bash
#SBATCH --job-name=p3_oracle
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/infres/yinwang/realeeg_feas/p3_forensic/logs/oracle_%j.out
# P3.0d oracle-generator fixed-margin diagnostic -- ONE shard (BASE x COND, 100 cohorts, B=200). Dev-only,
# diagnostic_only. Worker exit 2 (worker error OR fidelity break) => infra failure.
set -uo pipefail
: "${BASE:?}"; : "${COND:?}"; B="${B:-200}"
OUTDIR=/home/infres/yinwang/realeeg_feas/p3_forensic/oracle
OUT="$OUTDIR/oracle_${BASE}_${COND}.jsonl"
mkdir -p "$OUTDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[oracle shard] host=$(hostname) job=${SLURM_JOB_ID:-na} base=$BASE cond=$COND B=$B"
python -u /home/infres/yinwang/realeeg_feas/realeeg_oracle_forensic.py \
    --base "$BASE" --condition "$COND" --jobs "${SLURM_CPUS_PER_TASK:-8}" --n 100 --B "$B" --out "$OUT"
RC=$?; echo "[oracle base=$BASE cond=$COND] rc=$RC"; exit $RC
