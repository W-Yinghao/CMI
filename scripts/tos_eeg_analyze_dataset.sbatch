#!/bin/bash
#SBATCH --job-name=eeg_an_ds
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=24
#SBATCH --output=%x_%j.out
# usage: sbatch tos_eeg_analyze_dataset.sbatch <DATASET>
# Runs Track G + C12 FIRST (PM order), then the manifest gate + CONFIRM/MIXED/OVERTURN report. NOT LPC/factorial.
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D="$1"; PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
R=tos_cmi/results/tos_cmi_eeg_frozen
N='ConvergenceWarning|warnings.warn|UserWarning|FutureWarning|_multilayer|deprecat'
echo "### Track G erasure_baselines $D EEGNet"; $PY -u -m tos_cmi.eeg.erasure_baselines "$R/${D}_EEGNet_LOSO" 2>&1 | grep -vE "$N" | grep -E "===|full|LEACE|RLACE|INLP|TOS|random|chance|FAIL" | tail -12
echo "### Track G erasure_baselines $D TSMNet"; $PY -u -m tos_cmi.eeg.erasure_baselines "$R/${D}_TSMNet_LOSO" 2>&1 | grep -vE "$N" | grep -E "===|full|LEACE|RLACE|INLP|TOS|random|chance|FAIL|SKIP" | tail -12
echo "### C12 target deployment $D"; $PY -u -m tos_cmi.eeg.erasure_target_deploy --dataset "$D" --nrandom 8 2>&1 | grep -vE "$N" | grep -E "===|full|TOS|LEACE|RLACE|INLP|random|SKIP|DONE" | tail -20
echo "### manifest gate"; $PY -u -m tos_cmi.eeg.dataset_manifest "$D" 2>&1 | grep -vE "$N" | grep -E "$D|MANIFEST"
echo "### CONFIRM/MIXED/OVERTURN report"; $PY -u -m tos_cmi.eeg.bigN_report "$D" 2>&1 | grep -vE "$N"
echo "ANALYZE_${D}_DONE"
