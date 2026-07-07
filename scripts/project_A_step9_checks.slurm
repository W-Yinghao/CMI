#!/bin/bash
# Project A — Step 9 verification check suite (fast, no training except synthetic epochs=1 plumbing).
# Runs every Project-A check module + the counterexample catalog and fails loudly on the first error.
# Submitted via SLURM CPU per policy (no compute on the login node). NO --time (cluster convention).
#SBATCH --job-name=projA_step9_checks
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/project_A_step9_checks_%j.out
#SBATCH --error=logs/project_A_step9_checks_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

run () { echo "===== $1 ====="; conda run -n icml python $2; echo; }

run "smoke"             "-m h2cmi.tests.test_smoke"
run "counterexamples"   "notes/project_A_observability/counterexamples/run_counterexamples.py"
run "audit"             "-m h2cmi.tests.test_observability_audit"
run "eval-bridge"       "-m h2cmi.tests.test_observability_eval_bridge"
run "harness-contract"  "-m h2cmi.tests.test_eval_bridge_harness_contract"
run "result-index"      "-m h2cmi.tests.test_observability_result_index"
run "text-hygiene"      "-m h2cmi.tests.test_observability_text_hygiene"
run "grid-plumbing"     "-m h2cmi.tests.test_real_audited_grid_plumbing"

echo "ALL STEP-9 CHECKS PASSED"
