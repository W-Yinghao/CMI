#!/bin/bash
#SBATCH --job-name=srsmoke
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY -u -m tos_cmi.eeg.run_source_rich_smoke --datasets Lee2019_MI --backbones EEGNet --seeds 0 \
    --folds 5 --alphas 0.5 1.0 2.0 --tag smoke
echo "=== SMOKE REPORT ==="
$PY -u -m tos_cmi.eeg.report_source_rich_smoke --tag smoke
echo "SRSMOKE_SBATCH_DONE"
