#!/bin/bash
#SBATCH --job-name=v2smoke
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY -u -m tos_cmi.eeg.run_v2_certificate --worlds A B C --datasets Lee2019_MI Cho2017 \
    --backbones EEGNet --seeds 0 --n-source all --alphas 0.25 0.5 1.0 2.0 --folds 5 --n-pseudo 8 --tag smoke
echo "=== V2 SMOKE REPORT ==="
$PY -u -m tos_cmi.eeg.report_v2 --tag smoke
echo "V2SMOKE_SBATCH_DONE"
