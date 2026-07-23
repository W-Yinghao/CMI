#!/bin/bash
#SBATCH --job-name=lb-matrix
#SBATCH --partition=CPU
#SBATCH --exclude=nodecpu10,nodecpu11,nodecpu12
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_readout_prior/logs/lb-matrix-%A_%a.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_readout_prior/logs/lb-matrix-%A_%a.out
set -uo pipefail
cd /home/infres/yinwang/CMI_AAAI_readout_prior
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} PYTHONUNBUFFERED=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
i=$SLURM_ARRAY_TASK_ID
# skip if this cell already has a .done (idempotent)
if ls results/cmi_trace_readout_prior_lockbox/cells/cell_$(printf '%03d' $i)_*.done >/dev/null 2>&1; then echo "SKIP cell $i"; exit 0; fi
"$PY" -m scripts.run_readout_prior_decomposition --cell-index "$i" --out-dir results/cmi_trace_readout_prior_lockbox
