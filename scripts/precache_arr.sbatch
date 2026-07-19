#!/bin/bash
#SBATCH --job-name=lb-epoch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_readout_prior/logs/lb-epoch-%A_%a.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_readout_prior/logs/lb-epoch-%A_%a.out
set -uo pipefail
cd /home/infres/yinwang/CMI_AAAI_readout_prior
export MNE_DATA=/projects/EEG-foundation-model/yinghao/mne_data
export MNE_DATASETS_STIEGER2021_PATH=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data
export CMI_EPOCH_CACHE=/home/infres/yinwang/cmi_epoch_cache
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" /home/infres/yinwang/CMI_AAAI_readout_prior/results/precache_cells.txt)
read DS S <<< "$LINE"
echo "host=$(hostname) cell=$SLURM_ARRAY_TASK_ID -> $DS $S"
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/precache_one_subject.py "$DS" "$S"
