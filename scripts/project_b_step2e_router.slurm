#!/bin/bash
# Project B Step-2E: locked-world router evaluation (9 world/seed runs), single SLURM job + xargs.
# Output to SHARED home storage (/tmp is node-local). Aggregate afterwards with:
#   python scripts/project_b_eval_router_synthetic.py --mode aggregate --dir /home/infres/yinwang/project_b_step2e_router
#SBATCH --job-name=pb-step2e
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --output=/home/infres/yinwang/project_b_step2e_router/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_step2e_router/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB
E5=/home/infres/yinwang/project_b_step2e_router
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

cd "$WT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PY E5 WT

run_task() {
    local world="$1" seed="$2"
    local log="$E5/logs/${world}_seed${seed}.log"
    if "$PY" scripts/project_b_eval_router_synthetic.py --mode run \
            --world "$world" --seed "$seed" --out "$E5" > "$log" 2>&1; then
        echo "OK   $world seed$seed"
    else
        echo "FAIL $world seed$seed (see $log)"
    fi
}
export -f run_task

echo "driver start $(date) on $(hostname); tasks=$(wc -l < "$E5/tasklist.txt")"
xargs -a "$E5/tasklist.txt" -P 9 -L 1 bash -c 'run_task "$@"' _
echo "driver done $(date); produced $(ls "$E5"/*_router_report.json 2>/dev/null | wc -l) reports"
