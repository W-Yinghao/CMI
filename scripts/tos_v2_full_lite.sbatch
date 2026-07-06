#!/bin/bash
#SBATCH --job-name=v2fulllite
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "=== pre-launch: HEAD $(git rev-parse --short HEAD), config $(sha256sum tos_cmi/eeg/configs/v2_certificate_fixed.yaml | cut -c1-12) ==="
OUT=tos_cmi/results/method_deepen/v2_full_lite
$PY -u -m tos_cmi.eeg.run_v2_certificate --worlds A B C --datasets Lee2019_MI Cho2017 \
    --backbones EEGNet TSMNet --seeds 0 1 2 --n-source all --alphas 0.25 0.5 1.0 2.0 \
    --folds 15 --n-pseudo 8 --tag full_lite --outdir $OUT
echo "=== V2 FULL-LITE REPORT ==="
$PY -u -m tos_cmi.eeg.report_v2 --tag full_lite --outdir $OUT
echo "V2FULLLITE_SBATCH_DONE"
