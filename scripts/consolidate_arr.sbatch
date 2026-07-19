#!/bin/bash
#SBATCH --job-name=lb-consol
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_readout_prior/logs/lb-consol-%A_%a.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_readout_prior/logs/lb-consol-%A_%a.out
set -uo pipefail
cd /home/infres/yinwang/CMI_AAAI_readout_prior
export MNE_DATA=/projects/EEG-foundation-model/yinghao/mne_data
export MNE_DATASETS_STIEGER2021_PATH=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data
export CMI_EPOCH_CACHE=/home/infres/yinwang/cmi_epoch_cache
DS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" results/consolidate_cells.txt)
echo "host=$(hostname) consolidate $DS"
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/consolidate_cache.py "$DS"
