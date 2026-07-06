#!/bin/bash
# Project B Step-2A-NL: nested source-site LODO probe (HF3 only, 25 tasks).
# Single SLURM job + xargs -P fan-out (array trips QOSMaxSubmitJobPerUserLimit).
# Aggregate afterwards with:
#   python scripts/project_b_nested_lodo_probe.py --mode aggregate --dir /home/infres/yinwang/project_b_step2a_nested_lodo
#SBATCH --job-name=pb-nl
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --output=/home/infres/yinwang/project_b_step2a_nested_lodo/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_step2a_nested_lodo/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB
NL=/home/infres/yinwang/project_b_step2a_nested_lodo
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

cd "$WT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PY NL WT

run_task() {
    local seed="$1" u="$2"
    local log="$NL/logs/site_seed${seed}_u${u}.log"
    if "$PY" scripts/project_b_nested_lodo_probe.py --mode site \
            --seed "$seed" --pseudo_site "$u" --out "$NL" > "$log" 2>&1; then
        echo "OK   seed$seed u$u"
    else
        echo "FAIL seed$seed u$u (see $log)"
    fi
}
export -f run_task

echo "driver start $(date) on $(hostname); tasks=$(wc -l < "$NL/tasklist.txt")"
xargs -a "$NL/tasklist.txt" -P 12 -L 1 bash -c 'run_task "$@"' _
echo "driver done $(date); produced $(ls "$NL"/site_seed*_u*.json 2>/dev/null | wc -l) site JSONs"
