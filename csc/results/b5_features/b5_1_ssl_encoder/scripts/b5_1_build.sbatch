#!/bin/bash
#SBATCH --job-name=b5_1_build
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/infres/yinwang/realeeg_feas/b5_features/b5_1_build.%j.out
set -uo pipefail
mkdir -p /home/infres/yinwang/realeeg_feas/b5_features/b5_1_ssl_encoder
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[b5.1 build] host=$(hostname) job=${SLURM_JOB_ID:-na} gpu=$(nvidia-smi -L 2>/dev/null | head -1)"
python -u /home/infres/yinwang/realeeg_feas/b5_1_build_cache.py --subjects 54 \
  --out /home/infres/yinwang/realeeg_feas/b5_features/b5_1_ssl_encoder
echo "[b5.1 build] rc=$?"
