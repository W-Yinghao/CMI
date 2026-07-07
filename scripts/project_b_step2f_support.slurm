#!/bin/bash
# Project B Step-2F: nested support calibration (9 world/seed; each trains base + 5 nested models).
# Single SLURM job + xargs. Output to SHARED home storage (/tmp is node-local). Aggregate with:
#   python scripts/project_b_eval_router_support.py --mode aggregate --dir /home/infres/yinwang/project_b_step2f_support
#SBATCH --job-name=pb-step2f
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=40
#SBATCH --mem=48G
#SBATCH --output=/home/infres/yinwang/project_b_step2f_support/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_step2f_support/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB
F=/home/infres/yinwang/project_b_step2f_support
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

cd "$WT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PY F WT

run_task() {
    local world="$1" seed="$2"
    local log="$F/logs/${world}_seed${seed}.log"
    if "$PY" scripts/project_b_eval_router_support.py --mode run \
            --world "$world" --seed "$seed" --out "$F" > "$log" 2>&1; then
        echo "OK   $world seed$seed"
    else
        echo "FAIL $world seed$seed (see $log)"
    fi
}
export -f run_task

echo "driver start $(date) on $(hostname); tasks=$(wc -l < "$F/tasklist.txt")"
xargs -a "$F/tasklist.txt" -P 9 -L 1 bash -c 'run_task "$@"' _
echo "driver done $(date); produced $(ls "$F"/*_router_report.json 2>/dev/null | wc -l) reports"
