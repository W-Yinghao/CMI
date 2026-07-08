#!/bin/bash
# Project B Step-2A HFRAC sweep — source-calibratable harmful world (concept_frac=0.5).
# Single SLURM job + xargs -P fan-out (array trips QOSMaxSubmitJobPerUserLimit).
# Aggregate afterwards with:
#   python scripts/project_b_sweep_synthetic.py --mode aggregate --dir /home/infres/yinwang/project_b_step2a_hfrac
#SBATCH --job-name=pb-hfrac
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --output=/home/infres/yinwang/project_b_step2a_hfrac/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_step2a_hfrac/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB
HF=/home/infres/yinwang/project_b_step2a_hfrac
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

cd "$WT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PY HF WT

run_task() {
    local world="$1" cfg="$2" seed="$3"
    local log="$HF/logs/${world}_${cfg}_seed${seed}.log"
    if "$PY" scripts/project_b_sweep_synthetic.py --mode run \
            --world "$world" --config "$cfg" --seed "$seed" --out "$HF" > "$log" 2>&1; then
        echo "OK   $world $cfg seed$seed"
    else
        echo "FAIL $world $cfg seed$seed (see $log)"
    fi
}
export -f run_task

echo "driver start $(date) on $(hostname); tasks=$(wc -l < "$HF/tasklist.txt")"
xargs -a "$HF/tasklist.txt" -P 12 -L 1 bash -c 'run_task "$@"' _
echo "driver done $(date); produced $(ls "$HF"/*_seed*.json 2>/dev/null | wc -l) run JSONs"
