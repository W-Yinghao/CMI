#!/bin/bash
#SBATCH --job-name=tos_eeg_pilot
#SBATCH --partition=V100,V100-32GB,V100-16GB,A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
# NB: no --time (inherit partition max, per project compute-env policy). TSMNet needs a GPU.
set -e
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
CONDA=/home/infres/yinwang/anaconda3/bin/conda
echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-none}"
$CONDA run -n icml --no-capture-output python -u -m tos_cmi.run_eeg_frozen_pilot "$@"
echo "PILOT SBATCH DONE"
