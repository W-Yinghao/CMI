#!/bin/bash
#SBATCH --job-name=eeg_preflight
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
/home/infres/yinwang/anaconda3/envs/icml/bin/python -u -m tos_cmi.eeg.preflight_datasets
echo "PREFLIGHT_SBATCH_DONE"
