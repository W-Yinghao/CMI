#!/bin/bash
#SBATCH --job-name=v2Asearch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/infres/yinwang/anaconda3/envs/icml/bin/python -u -m tos_cmi.eeg.v2_worldA_search --datasets Lee2019_MI Cho2017 --backbone EEGNet --seed 0 --folds 8
echo "V2ASEARCH_SBATCH_DONE"
