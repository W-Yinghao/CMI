#!/bin/bash
#SBATCH --job-name=v2timeprobe
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
t0=$SECONDS
echo "=== Lee TSMNet 3 folds, 1 alpha, all worlds (90 tasks), 16 cores ==="
$PY -u -m tos_cmi.eeg.run_v2_certificate --worlds A B C --datasets Lee2019_MI --backbones TSMNet \
    --seeds 0 --n-source all --alphas 1.0 --folds 3 --n-pseudo 8 --tag timeprobe \
    --outdir tos_cmi/results/method_deepen/v2_timeprobe
echo "ELAPSED_SECONDS=$((SECONDS - t0))"
echo "V2TIMEPROBE_SBATCH_DONE"
