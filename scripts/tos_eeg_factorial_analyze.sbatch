#!/bin/bash
#SBATCH --job-name=fact_an
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
/home/infres/yinwang/anaconda3/envs/icml/bin/python -u -m tos_cmi.eeg.factorial_analysis 0
echo "FACT_AN_SBATCH_DONE"
