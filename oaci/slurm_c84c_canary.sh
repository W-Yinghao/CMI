#!/usr/bin/env bash
#SBATCH --job-name=c84c-canary
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v2/slurm-%j.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v2/slurm-%j.err
set -euo pipefail

repo=/home/infres/yinwang/CMI_AAAI_oaci
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/icml
cd "$repo"

python -m oaci.multidataset.c84c_real_canary run-real \
  --authorization-record oaci/reports/C84C_PI_AUTHORIZATION_RECORD.json \
  --output-root /projects/EEG-foundation-model/yinghao/oaci-c84-canary-v2
