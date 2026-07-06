#!/bin/bash
#SBATCH --job-name=b7s1
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/realeeg_feas/b7_stage1_rows/logs/b7s1_%j.out
set -uo pipefail
: "${COND:?}"; : "${NN:?}"
mkdir -p /home/infres/yinwang/realeeg_feas/b7_stage1_rows/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/infres/yinwang/realeeg_feas
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[b7s1] host=$(hostname) job=${SLURM_JOB_ID:-na} cond=$COND base=300000000 n=$NN jobs=48 START"
python -u realeeg_b6_canary.py --base 300000000 --condition "$COND" --start 0 --n "$NN" --jobs 48 \
  --out b7_stage1_rows/b7s1_${COND}_0.jsonl
RC=$?; echo "[b7s1] cond=$COND rc=$RC DONE"; exit $RC
