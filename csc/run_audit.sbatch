#!/bin/bash
# csc CSC-P1.1 audit — CPU-only, submit with:  sbatch csc/run_audit.sbatch
# Conventions (cmi-compute-env): CPU partition, NO --time (inherit partition max), full-path
# conda in the batch shell, no nested srun. Output to gitignored logs/.
#SBATCH --job-name=csc_audit
#SBATCH --partition=CPU
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI
#SBATCH --output=logs/csc_audit_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -e
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONWARNINGS=ignore
echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK started=$(date -Is)"
mkdir -p csc/results
/home/infres/yinwang/anaconda3/bin/conda run -n icml --no-capture-output \
    python -m csc.run_audit --out csc/results/audit.json \
    --seeds 10 --n_boot 40 --n_dir_boot 120 --target_n_boot 120 --oracle_boot 150 --bank_seeds 4
echo "finished=$(date -Is)"
# committable stdout provenance: copy the run log + record its SHA256
LOG="logs/csc_audit_${SLURM_JOB_ID}.out"
cp "$LOG" "csc/results/audit_run.log" 2>/dev/null || true
sha256sum "csc/results/audit_run.log" > "csc/results/audit_run.log.sha256" 2>/dev/null || true
echo "stdout_sha256=$(cut -d' ' -f1 csc/results/audit_run.log.sha256 2>/dev/null)"
