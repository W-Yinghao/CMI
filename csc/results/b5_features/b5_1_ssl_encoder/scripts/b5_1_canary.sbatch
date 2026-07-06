#!/bin/bash
#SBATCH --job-name=b5_1_canary
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --output=/home/infres/yinwang/realeeg_feas/b5_features/b5_1_ssl_encoder/canary/logs/b51_%j.out
# B5.1 SSL-encoder canary: ONE consolidated job (64 cores), all 6 conditions x n=80 (reviewer n>=60), base 40e6.
set -uo pipefail
CDIR=/home/infres/yinwang/realeeg_feas/b5_features/b5_1_ssl_encoder/canary
mkdir -p "$CDIR/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# point the (parametrized) canary worker at the B5.1 SSL cache
export B5_CACHE_PATH=/home/infres/yinwang/realeeg_feas/b5_features/b5_1_ssl_encoder/LEE2019_B5_1.npz
export B5_MANIFEST_PATH=/home/infres/yinwang/realeeg_feas/b5_features/b5_1_ssl_encoder/b5_1_feature_manifest.json
export B5_FAMILY=B5_1_ssl_eegnet
cd /home/infres/yinwang/realeeg_feas
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[b5_1_canary] host=$(hostname) job=${SLURM_JOB_ID:-na} family=$B5_FAMILY base=40000000 n=80 jobs=64 START"
RC=0
for COND in NULL_cov NULL_cov_plus_label NULL_label random_label_control POS_concept POS_concept_plus_cov; do
  python -u realeeg_b5_canary.py --base 40000000 --condition "$COND" --start 0 --n 80 --jobs 64 \
    --out "$CDIR/b5_1_canary_${COND}_0.jsonl" || RC=$?
done
echo "[b5_1_canary] DONE rc=$RC"; exit $RC
