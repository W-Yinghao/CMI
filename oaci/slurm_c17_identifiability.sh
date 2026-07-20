#!/usr/bin/env bash
#SBATCH --job-name=oaci-c17-ident
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
echo "[c17-ident] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null)"
$PY -m oaci.identifiability.report
echo "=== rc=$? ==="
