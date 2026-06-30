#!/bin/bash
#SBATCH --job-name=tos_erase
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
/home/infres/yinwang/anaconda3/envs/icml/bin/python -u -m tos_cmi.eeg.erasure_baselines "$1"
echo "ERASURE_SBATCH_DONE"
