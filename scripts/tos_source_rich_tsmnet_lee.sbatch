#!/bin/bash
#SBATCH --job-name=srtsm_lee
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "=== Phase 1C TSMNet check (Lee2019_MI) ==="
echo "PROVENANCE: D_nuis is the injected nuisance label, NOT EEG subject identity."
echo "PROVENANCE: params FROZEN from Lee development / Cho confirmation (EEGNet). No retuning."
echo "PROVENANCE: TSMNet is a BACKBONE ROBUSTNESS check, not a second round of tuning."
echo "nuisance dim: m = _nuisance_m(z_dim=210, fraction_of_z_dim, 0.20, 4) = 42 (scaled by z_dim, same rule as V2)."
echo "FROZEN: frac 0.4/0.3/0.3, alpha 0.5/1.0/2.0, thresholds safety<=0.02 benefit>0.01, k=8, envs unchanged."
$PY -u -m tos_cmi.eeg.run_source_rich_smoke --datasets Lee2019_MI --backbones TSMNet --seeds 0 \
    --folds 5 --alphas 0.5 1.0 2.0 --frac 0.4 0.3 0.3 --tag tsmnet_lee
echo "=== TSMNet Lee REPORT ==="
$PY -u -m tos_cmi.eeg.report_source_rich_smoke --tag tsmnet_lee
echo "SRTSM_LEE_SBATCH_DONE"
