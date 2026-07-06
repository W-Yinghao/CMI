#!/bin/bash
#SBATCH --job-name=v2s2gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "=== tests (incl stage2 config) ==="; $PY -u -m tos_cmi.eeg.test_v2_worlds | tail -3
echo "=== dry-run composition ==="; $PY -u -m tos_cmi.eeg.run_v2_stage2_scoped --dry-run 2>&1 | grep -E "Stage-2 scoped|composition|DRY_RUN"
echo "=== 1-fold run: must show TARGET_LEAK_STRUCTURAL_PASS + STAGE2_CLEAN ==="
$PY -u -m tos_cmi.eeg.run_v2_stage2_scoped --folds 1 --seeds 0 --n-source all --alphas 1.0 \
    --tag s2gate --outdir tos_cmi/results/method_deepen/v2_stage2_gate 2>&1 | \
  grep -E "TARGET_LEAK_STRUCTURAL_PASS|World A / EEGNet|stop conditions|target-leak|false-accept|clean positives|random reproduces|World-B unsafe|World-C principled|TSMNet B/C degen|STAGE2_" | head -25
rm -rf tos_cmi/results/method_deepen/v2_stage2_gate
echo "V2S2GATE_SBATCH_DONE"
