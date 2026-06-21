#!/bin/bash
# OACI CI on a SLURM CPU node — runs all test modules + both demos in PARALLEL, off the
# contended login node. Submit:  sbatch oaci/slurm_ci.sh   (run from the worktree root).
# Per cluster policy: NO --time/walltime. CPU partition (4-day limit).
#SBATCH --job-name=oaci-ci
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_oaci/logs/oaci-ci-%j.out
set -u
# cap per-process BLAS/torch threads so the parallel jobs don't oversubscribe the node
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
ROOT=/home/infres/yinwang/CMI_AAAI_oaci
cd "$ROOT" || exit 1
mkdir -p logs
WORK=$(mktemp -d)

MODS="test_support_graph test_missing_cell test_leakage_estimate test_leakage_crossfit \
test_leakage_ucb test_train_risk test_train_adversary test_train_primal_dual \
test_train_selector test_rare_cell_sampler"

echo "[oaci-ci] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} commit=$(git rev-parse --short HEAD 2>/dev/null)"
for m in $MODS; do
  ( $PY -m oaci.tests.$m >"$WORK/$m.log" 2>&1; echo $? >"$WORK/$m.rc" ) &
done
( $PY -m oaci.data.sampler_demo >"$WORK/d_sampler.log" 2>&1; echo $? >"$WORK/d_sampler.rc" ) &
( $PY -m oaci.train.synthetic   >"$WORK/d_trainer.log" 2>&1; echo $? >"$WORK/d_trainer.rc" ) &
wait

echo "=== TEST RESULTS ==="
fail=0
for m in $MODS; do
  rc=$(cat "$WORK/$m.rc"); printf "%-26s rc=%s  %s\n" "$m" "$rc" "$(tail -1 "$WORK/$m.log")"
  [ "$rc" != 0 ] && { fail=1; echo "---- $m FAILED ----"; tail -15 "$WORK/$m.log"; }
done
echo "=== sampler_demo (rc=$(cat "$WORK/d_sampler.rc")) ==="; cat "$WORK/d_sampler.log"
echo "=== trainer_demo (rc=$(cat "$WORK/d_trainer.rc")) ==="; cat "$WORK/d_trainer.log"
echo "=== OVERALL: $([ $fail = 0 ] && echo ALL-TEST-MODULES-PASS || echo FAILURES) ==="
