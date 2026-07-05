#!/bin/bash
#SBATCH --job-name=b5_canary
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --array=0-11
#SBATCH --output=/home/infres/yinwang/realeeg_feas/b5_features/b5_0_random_encoder/canary/slurm_%A_%a.out
# CSC-realEEG B5.0 6-condition canary (development-only; Z_deep frozen-random feature family).
# 12 array tasks = 6 conditions x 2 cohort-halves (start 0 / start 25), n=25 each -> n=50 per condition.
# base seed 30_000_000 (disjoint from v2 20e6, P3 50/60/70e6, R1 80/90e6). Certifier env = icml (P3/R1 parity).
set -uo pipefail
CONDS=(NULL_cov NULL_cov_plus_label NULL_label random_label_control POS_concept POS_concept_plus_cov)
IDX=$SLURM_ARRAY_TASK_ID
COND=${CONDS[$((IDX / 2))]}
START=$(( (IDX % 2) * 25 ))
N=25
OUT=/home/infres/yinwang/realeeg_feas/b5_features/b5_0_random_encoder/canary/b5_canary_${COND}_${START}.jsonl
cd /home/infres/yinwang/realeeg_feas
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[b5_canary] host=$(hostname) job=${SLURM_JOB_ID:-na} cond=$COND base=30000000 start=$START n=$N jobs=16 START"
python -u realeeg_b5_canary.py --base 30000000 --condition "$COND" --start "$START" --n "$N" --jobs 16 --out "$OUT"
RC=$?; echo "[b5_canary] cond=$COND start=$START DONE rc=$RC"; exit $RC
