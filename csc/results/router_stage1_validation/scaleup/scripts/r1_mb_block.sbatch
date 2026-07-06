#!/bin/bash
#SBATCH --job-name=r1mb_block
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/realeeg_feas/router_stage1/multiblock/logs/mb_%j.out
set -uo pipefail
: "${BASE:?}"
mkdir -p /home/infres/yinwang/realeeg_feas/router_stage1/multiblock/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/infres/yinwang/realeeg_feas
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[r1mb_block] host=$(hostname) job=${SLURM_JOB_ID:-na} BASE=$BASE cpus=48 START"
RC=0
for COND in NULL_cov NULL_cov_plus_label POS_concept; do
  OUT=router_stage1/multiblock/r1_${BASE}_${COND}_0.jsonl
  python -u realeeg_r1_forensic.py --base "$BASE" --condition "$COND" --start 0 --n 300 --jobs 48 --out "$OUT" || RC=$?
done
echo "[r1mb_block] BASE=$BASE DONE rc=$RC"; exit $RC
