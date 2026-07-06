#!/bin/bash
#SBATCH --job-name=v2s2val
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY -u -m tos_cmi.eeg.run_v2_stage2_scoped --folds 1 --seeds 0 --n-source all --alphas 1.0 \
    --tag s2validate --outdir tos_cmi/results/method_deepen/v2_stage2_validate 2>&1 | \
  grep -E "Stage-2 scoped|composition|World A / EEGNet|n_source=all|World B|World C|stop conditions|false-accept|EEGNet World-A clean|random reproduces|World-B unsafe|World-C principled|TSMNet B/C degen|STAGE2_|degenerate" | head -40
echo "outputs:"; ls -1 tos_cmi/results/method_deepen/v2_stage2_validate/ 2>/dev/null
echo "V2S2VAL_SBATCH_DONE"
