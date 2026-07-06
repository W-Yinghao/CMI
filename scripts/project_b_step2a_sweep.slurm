#!/bin/bash
# Project B Step-2A substrate sweep — SINGLE SLURM job with internal bounded parallelism.
# (A 36-task array trips QOSMaxSubmitJobPerUserLimit, so we submit ONE job and fan out with xargs.)
# Writes per-run JSONs to SHARED home storage (NOT /tmp, which is node-local on this cluster).
# Aggregate afterwards with:
#   python scripts/project_b_sweep_synthetic.py --mode aggregate --dir /home/infres/yinwang/project_b_step2a_sweep
#SBATCH --job-name=pb-step2a
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --output=/home/infres/yinwang/project_b_step2a_sweep/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_step2a_sweep/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB
SWEEP=/home/infres/yinwang/project_b_step2a_sweep
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

cd "$WT"
# 12 concurrent runs x 4 threads each = 48 cpus
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PY SWEEP WT

run_task() {
    local world="$1" cfg="$2" seed="$3"
    local log="$SWEEP/logs/${world}_${cfg}_seed${seed}.log"
    if "$PY" scripts/project_b_sweep_synthetic.py --mode run \
            --world "$world" --config "$cfg" --seed "$seed" --out "$SWEEP" > "$log" 2>&1; then
        echo "OK   $world $cfg seed$seed"
    else
        echo "FAIL $world $cfg seed$seed (see $log)"
    fi
}
export -f run_task

echo "driver start $(date) on $(hostname); tasks=$(wc -l < "$SWEEP/tasklist.txt")"
xargs -a "$SWEEP/tasklist.txt" -P 12 -L 1 bash -c 'run_task "$@"' _
echo "driver done $(date); produced $(ls "$SWEEP"/*_seed*.json 2>/dev/null | wc -l) run JSONs"
