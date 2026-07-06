#!/bin/bash
#SBATCH --job-name=v2final
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "=== tests ==="; $PY -u -m tos_cmi.eeg.test_v2_worlds | tail -2
echo "=== report_v2 (PM's exact CLI: --stage/--rows/--out-dir) ==="
$PY -u -m tos_cmi.eeg.report_v2 --stage stage2 \
    --rows tos_cmi/results/method_deepen/v2_stage2/v2_stage2_rows.csv \
    --out-dir tos_cmi/results/method_deepen/v2_stage2 2>&1 | grep -E "overall:|World A verdict" | head
echo "V2FINALGATE_SBATCH_DONE"
