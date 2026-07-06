#!/bin/bash
#SBATCH --job-name=sc_delta
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/realeeg_feas/router_stage1/strongcov/logs/scd_%j.out
set -uo pipefail
: "${DELTA:?}"
mkdir -p /home/infres/yinwang/realeeg_feas/router_stage1/strongcov/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/infres/yinwang/realeeg_feas
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[sc_delta] host=$(hostname) job=${SLURM_JOB_ID:-na} DELTA=$DELTA cpus=48 START"
python -u realeeg_strongcov.py --base 100000000 --delta "$DELTA" --start 0 --n 300 --jobs 48 \
  --out router_stage1/strongcov/sc_100000000_d${DELTA}_0.jsonl
RC=$?; echo "[sc_delta] DELTA=$DELTA DONE rc=$RC"; exit $RC
