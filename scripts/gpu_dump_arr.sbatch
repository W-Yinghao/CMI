#!/bin/bash
#SBATCH --job-name=lb-dump
#SBATCH --partition=A100,H100,L40S,A40,V100,V100-32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_readout_prior/logs/cmi_trace/lb-dump-%A_%a.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_readout_prior/logs/cmi_trace/lb-dump-%A_%a.out
set -uo pipefail
cd /home/infres/yinwang/CMI_AAAI_readout_prior
export MNE_DATA=/projects/EEG-foundation-model/yinghao/mne_data
export MNE_DATASETS_STIEGER2021_PATH=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data
export CMI_EPOCH_CACHE=/home/infres/yinwang/cmi_epoch_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6} PYTHONUNBUFFERED=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" /home/infres/yinwang/CMI_AAAI_readout_prior/results/gpu_dump_cells.txt); read DS S SEED <<< "$LINE"
echo "host=$(hostname) cell=$SLURM_ARRAY_TASK_ID -> $DS sub$S seed$SEED commit=$(git rev-parse --short HEAD)"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: CUDA unavailable on $(hostname)"; exit 1
fi
"$PY" scripts/gpu_dump_one.py "$DS" "$S" "$SEED"
