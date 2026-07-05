#!/bin/bash
#SBATCH --job-name=b4_canary
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/infres/yinwang/realeeg_feas/b4_stage1/logs/b4_%j.out
# B4 Stage 1 canary shard (development-only, diagnostic). Worker exit 2 (worker error OR fidelity break)=infra fail.
set -uo pipefail
: "${SLICE:?}"
OUTDIR=/home/infres/yinwang/realeeg_feas/b4_stage1
OUT="$OUTDIR/b4_shard_${SLICE/:/_}.jsonl"
mkdir -p "$OUTDIR/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[b4 shard] host=$(hostname) job=${SLURM_JOB_ID:-na} slice=$SLICE"
python -u /home/infres/yinwang/realeeg_feas/realeeg_b4_stage1.py \
  --manifest "$OUTDIR/b4_stage1_manifest.json" --slice "$SLICE" --jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --B 200 --Ka 50 --Kb 200 --out "$OUT"
RC=$?; echo "[b4 slice=$SLICE] rc=$RC"; exit $RC
