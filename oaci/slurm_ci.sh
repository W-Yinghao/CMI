#!/bin/bash
# OACI CI on a SLURM CPU node — runs all test modules + both demos in PARALLEL, off the
# contended login node, and EXITS NONZERO if any test module OR demo fails.
# Submit from the worktree root:  sbatch oaci/slurm_ci.sh
# Per cluster policy: NO --time/walltime. CPU partition (4-day limit).
#SBATCH --job-name=oaci-ci
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/oaci-ci-%j.out
set -u
# cap per-process BLAS/torch threads so the parallel jobs don't oversubscribe the node
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
# run relative to the worktree root (the dir sbatch was submitted from)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p logs
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

MODS="test_support_graph test_missing_cell test_leakage_estimate test_leakage_crossfit \
test_leakage_ucb test_train_risk test_train_adversary test_train_primal_dual \
test_train_selector test_train_engine test_rare_cell_sampler test_eval test_data_contract \
test_loader_protocol test_sample_mass test_backbone"

echo "[oaci-ci] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} commit=$(git rev-parse --short HEAD 2>/dev/null)"
# every parallel job writes its return code to $WORK/<name>.rc; the final pass/fail folds ALL of them.
for m in $MODS; do
  ( $PY -m oaci.tests.$m >"$WORK/$m.log" 2>&1; echo $? >"$WORK/$m.rc" ) &
done
( $PY -m oaci.data.sampler_demo >"$WORK/demo_sampler.log" 2>&1; echo $? >"$WORK/demo_sampler.rc" ) &
( $PY -m oaci.train.synthetic   >"$WORK/demo_trainer.log" 2>&1; echo $? >"$WORK/demo_trainer.rc" ) &
( $PY -m oaci.eval.synthetic    >"$WORK/demo_eval.log"    2>&1; echo $? >"$WORK/demo_eval.rc" ) &
( $PY -m oaci.data.eeg.smoke    >"$WORK/demo_data.log"    2>&1; echo $? >"$WORK/demo_data.rc" ) &
( $PY -m oaci.data.mass_demo    >"$WORK/demo_mass.log"    2>&1; echo $? >"$WORK/demo_mass.rc" ) &
wait

echo "=== TEST RESULTS ==="
for m in $MODS; do printf "%-26s rc=%s  %s\n" "$m" "$(cat "$WORK/$m.rc")" "$(tail -1 "$WORK/$m.log")"; done
for dem in demo_sampler demo_trainer demo_eval demo_data demo_mass; do
  echo "=== $dem (rc=$(cat "$WORK/$dem.rc")) ==="; cat "$WORK/$dem.log"
done

# fold EVERY rc (tests AND demos) into the job exit status, and dump logs for any failure.
fail=0
for f in "$WORK"/*.rc; do
  rc=$(cat "$f"); name=$(basename "$f" .rc)
  if [ "$rc" -ne 0 ]; then fail=1; echo "---- FAILED: $name (rc=$rc) ----"; tail -25 "$WORK/$name.log"; fi
done
echo "=== OVERALL: $([ "$fail" -eq 0 ] && echo ALL-PASS || echo FAILURES) (exit $fail) ==="
exit "$fail"
