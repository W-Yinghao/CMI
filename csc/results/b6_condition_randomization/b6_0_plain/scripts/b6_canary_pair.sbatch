#!/bin/bash
#SBATCH --job-name=b6_canary
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/realeeg_feas/b6_canary/logs/b6_%j.out
set -uo pipefail
: "${CONDS:?}"    # space-separated B6 condition names
mkdir -p /home/infres/yinwang/realeeg_feas/b6_canary/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/infres/yinwang/realeeg_feas
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[b6_canary] host=$(hostname) job=${SLURM_JOB_ID:-na} conds='$CONDS' base=200000000 n=50 jobs=48 START"
RC=0
for C in $CONDS; do
  python -u realeeg_b6_canary.py --base 200000000 --condition "$C" --start 0 --n 50 --jobs 48 \
    --out b6_canary/b6_canary_${C}_0.jsonl || RC=$?
done
echo "[b6_canary] conds='$CONDS' DONE rc=$RC"; exit $RC
