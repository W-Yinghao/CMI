#!/bin/bash
#SBATCH --job-name=phase2dry
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY -u -m tos_cmi.eeg.run_phase2_task_preserving --datasets Lee2019_MI Cho2017 --backbones EEGNet --seeds 0 --folds 15 --n-pseudo 8
echo "=== PHASE 2 REPORT ==="
$PY -u -m tos_cmi.eeg.report_phase2
echo "PHASE2_SBATCH_DONE"
