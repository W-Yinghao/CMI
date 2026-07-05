#!/bin/bash
#SBATCH --job-name=v2probeA
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "=== tests ==="; $PY -u -m tos_cmi.eeg.test_v2_worlds | tail -2
for BB in TSMNet EEGNet; do
  echo "===== World A probe: $BB (Lee+Cho seed0 first3, alpha grid) ====="
  $PY -u -m tos_cmi.eeg.run_v2_certificate --worlds A --datasets Lee2019_MI Cho2017 --backbones $BB --seeds 0 \
      --n-source all --alphas 0.25 0.5 1.0 2.0 --folds 3 --n-pseudo 8 --tag probe_$BB \
      --outdir tos_cmi/results/method_deepen/v2_probeA 2>&1 | grep -E "nuisance=|World A|coverage|DEGEN" | head -12
done
echo "V2PROBEA_SBATCH_DONE"
