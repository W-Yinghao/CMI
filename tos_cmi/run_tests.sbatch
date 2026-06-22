#!/bin/bash
#SBATCH --job-name=tos_test
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
# NB: no --time / walltime (per project compute-env policy)
set -e
cd /home/infres/yinwang/CMI_AAAI_tos
CONDA=/home/infres/yinwang/anaconda3/bin/conda
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
for t in test_fisher test_subspace_identity_fallback test_projection_ablation \
         test_stability test_limits test_smoke test_score_fisher test_bayes_calibration; do
  echo "===== $t ====="
  $CONDA run -n icml python -m tos_cmi.tests.$t
done
echo "ALL TOS-CMI TESTS DONE"
