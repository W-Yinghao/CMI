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
test_train_selector test_train_engine test_methods test_leakage_plan test_leakage_parallel test_manifest_artifacts \
test_plan_sampler test_runner_scoring test_runner_contracts test_runner_scope test_runner_plans \
test_runner_train_select test_runner_audit test_runner_finalize test_runner_artifacts test_runner_fake test_runner_fake_artifact test_replay_store test_staged_executor test_scientific_hash test_artifact_pure_science_hash test_optimization_seed_decoupling test_rare_cell_sampler test_eval test_data_contract test_loader_protocol \
test_sample_mass test_backbone test_real_data_blockers test_bnci_loader test_bnci_runner_adapter test_cuda_runtime test_bnci_gpu_compare test_bnci_gpu_script test_confirmatory_adapter test_confirmatory_report"

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

# ---- fake two-level artifact demo + standalone verifier (sequential; needs a CLEAN scientific tree) ----
FAKE_OUT="$WORK/fake-artifact"; mkdir -p "$FAKE_OUT"
$PY -m oaci.runner.demo --manifest oaci/protocol/fake_runner_v1.yaml --output-root "$FAKE_OUT" \
    --model-seed 0 --method-order ERM,OACI,global_lpc,uniform --repo-root "${SLURM_SUBMIT_DIR:-$(pwd)}" \
    >"$WORK/demo_fake.json" 2>"$WORK/demo_fake.log"; echo $? >"$WORK/demo_fake.rc"
if [ "$(cat "$WORK/demo_fake.rc")" -eq 0 ]; then
  ART_DIR=$($PY -c "import json; print(json.load(open('$WORK/demo_fake.json'))['artifact_dir'])" 2>>"$WORK/demo_fake.log")
  $PY -m oaci.artifacts.verify "$ART_DIR" >"$WORK/verify_fake.log" 2>&1; echo $? >"$WORK/verify_fake.rc"
else
  echo 1 >"$WORK/verify_fake.rc"
fi

echo "=== TEST RESULTS ==="
for m in $MODS; do printf "%-26s rc=%s  %s\n" "$m" "$(cat "$WORK/$m.rc")" "$(tail -1 "$WORK/$m.log")"; done
for dem in demo_sampler demo_trainer demo_eval demo_data demo_mass demo_fake verify_fake; do
  echo "=== $dem (rc=$(cat "$WORK/$dem.rc")) ==="; tail -3 "$WORK/$dem.log"
done

# fold EVERY rc (tests AND demos) into the job exit status, and dump logs for any failure.
fail=0
for f in "$WORK"/*.rc; do
  rc=$(cat "$f"); name=$(basename "$f" .rc)
  if [ "$rc" -ne 0 ]; then fail=1; echo "---- FAILED: $name (rc=$rc) ----"; tail -25 "$WORK/$name.log"; fi
done
echo "=== OVERALL: $([ "$fail" -eq 0 ] && echo ALL-PASS || echo FAILURES) (exit $fail) ==="
exit "$fail"
