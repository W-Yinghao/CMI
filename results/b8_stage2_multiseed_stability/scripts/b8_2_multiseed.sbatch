#!/bin/bash
#SBATCH --job-name=b8_2_ms
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/realeeg_feas/b8_2_multiseed/logs/b8_2_%j.out
set -uo pipefail
: "${BASE:?}"
mkdir -p /home/infres/yinwang/realeeg_feas/b8_2_multiseed/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/infres/yinwang/realeeg_feas
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
WORLDS="CONTRACT_NULL_balanced CONTRACT_NULL_prior_only CONTRACT_NULL_cov_plus_prior CONTRACT_random_label CONTRACT_POS_boundary CONTRACT_POS_boundary_plus_prior VIOLATION_cov_session VIOLATION_prior_shift VIOLATION_cov_plus_prior VIOLATION_condition_lock VIOLATION_quiet_cov_no_concept VIOLATION_quiet_cov_plus_concept"
echo "[b8_2_ms] host=$(hostname) job=${SLURM_JOB_ID:-na} BASE=$BASE n=50/world START (UNCHANGED B8.1 worker)"
RC=0
for W in $WORLDS; do
  python -u realeeg_b8_1_canary.py --base "$BASE" --world "$W" --start 0 --n 50 --jobs 48 \
    --out b8_2_multiseed/b8_2_${BASE}_${W}.jsonl || RC=$?
done
echo "[b8_2_ms] BASE=$BASE DONE rc=$RC"; exit $RC
