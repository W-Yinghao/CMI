#!/bin/bash
#SBATCH --job-name=tier1bf
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
TIER1_SMOKE_TOKEN=$(cat tos_cmi/results/.tier1_bf_token)
echo "=== Tier-1B label-budget frontier (k up to 50; hardened estimator; same data scope) ==="
$PY -u -m tos_cmi.eeg.run_target_info_tier1_smoke \
    --execute \
    --run-manifest tos_cmi/eeg/configs/target_info_tier1_run_manifest_APPROVED_BUDGET_FRONTIER.yaml \
    --enable-token "$TIER1_SMOKE_TOKEN"
echo "TIER1_BF_SBATCH_DONE (exit $?)"
