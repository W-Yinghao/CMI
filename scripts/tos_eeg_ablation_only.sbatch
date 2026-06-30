#!/bin/bash
#SBATCH --job-name=eeg_abl
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export TOS_SEED_FILTER="$1"
B=tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_${2:-EEGNet}_LOSO
P=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$P -u -m tos_cmi.eeg.ablation    $B erm && mv $B/ablation_report.json    $B/ablation_report_seed$1.json
$P -u -m tos_cmi.eeg.adversarial $B     && mv $B/adversarial_report.json $B/adversarial_report_seed$1.json
echo "ABL_SEED_${1}_DONE"
