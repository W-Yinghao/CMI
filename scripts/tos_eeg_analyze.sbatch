#!/bin/bash
#SBATCH --job-name=tos_eeg_an
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
set -e
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export TOS_SEED_FILTER="$1"
S="$1"
BB="${2:-TSMNet}"
B=tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_${BB}_LOSO
CONDA="/home/infres/yinwang/anaconda3/bin/conda run -n icml --no-capture-output"
echo "=== seed $S analyses ==="
$CONDA python -u -m tos_cmi.eeg.report      $B && mv $B/diagnostic_report.json  $B/diagnostic_report_seed$S.json
$CONDA python -u -m tos_cmi.eeg.geometry    $B && mv $B/geometry_report.json    $B/geometry_report_seed$S.json
$CONDA python -u -m tos_cmi.eeg.ablation    $B erm && mv $B/ablation_report.json $B/ablation_report_seed$S.json
$CONDA python -u -m tos_cmi.eeg.adversarial $B && mv $B/adversarial_report.json $B/adversarial_report_seed$S.json
echo "ANALYZE_SEED_${S}_DONE"
