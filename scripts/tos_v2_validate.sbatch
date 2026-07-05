#!/bin/bash
#SBATCH --job-name=v2validate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "=== PRE-LAUNCH CHECKS ==="
echo "HEAD: $(git rev-parse HEAD)"
echo "config wc -l: $(git show HEAD:tos_cmi/eeg/configs/v2_certificate_fixed.yaml | wc -l)"
echo "config hash: $(sha256sum tos_cmi/eeg/configs/v2_certificate_fixed.yaml | cut -c1-12)"
echo "goal: $($PY -c "import yaml;print(yaml.safe_load(open('tos_cmi/eeg/configs/v2_certificate_fixed.yaml'))['goal'])")"
echo "--- tests ---"; $PY -u -m tos_cmi.eeg.test_v2_worlds | tail -3
echo "=== PLUMBING VALIDATION (folds=1, Lee EEGNet, all worlds) ==="
$PY -u -m tos_cmi.eeg.run_v2_certificate --worlds A B C --datasets Lee2019_MI --backbones EEGNet --seeds 0 \
    --n-source all --alphas 1.0 --folds 1 --n-pseudo 6 --tag validate --outdir tos_cmi/results/method_deepen/v2_validate | tail -12
$PY -u -m tos_cmi.eeg.report_v2 --tag validate --outdir tos_cmi/results/method_deepen/v2_validate | tail -6
echo "--- output files ---"; ls -1 tos_cmi/results/method_deepen/v2_validate/
echo "V2VALIDATE_SBATCH_DONE"
