#!/bin/bash
#SBATCH --job-name=eeg_an_ds
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=24
#SBATCH --output=%x_%j.out
# usage: sbatch tos_eeg_analyze_dataset.sbatch <DATASET>
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D="$1"; PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
R=tos_cmi/results/tos_cmi_eeg_frozen
NOISE='ConvergenceWarning|warnings.warn|UserWarning|FutureWarning|_multilayer|deprecat'
echo "### erasure_baselines $D TSMNet"; $PY -u -m tos_cmi.eeg.erasure_baselines "$R/${D}_TSMNet_LOSO" 2>&1 | grep -vE "$NOISE" | grep -E "===|full|LEACE|RLACE|INLP|TOS|random|chance|FAIL" | tail -12
echo "### erasure_baselines $D EEGNet"; $PY -u -m tos_cmi.eeg.erasure_baselines "$R/${D}_EEGNet_LOSO" 2>&1 | grep -vE "$NOISE" | grep -E "===|full|LEACE|RLACE|INLP|TOS|random|chance|FAIL" | tail -12
echo "### target deployment $D"; $PY -u -m tos_cmi.eeg.erasure_target_deploy --dataset "$D" --nrandom 8 2>&1 | grep -vE "$NOISE" | grep -E "===|full|TOS|LEACE|RLACE|INLP|random|Deploying|SKIP|DONE" | tail -30
echo "ANALYZE_${D}_DONE"
