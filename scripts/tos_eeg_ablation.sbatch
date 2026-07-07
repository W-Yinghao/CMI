#!/bin/bash
#SBATCH --job-name=tos_eeg_ablate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
set -e
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
/home/infres/yinwang/anaconda3/bin/conda run -n icml --no-capture-output python -u -m tos_cmi.eeg.ablation "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO" "${1:-all}"
echo "ABLATION SBATCH DONE"
