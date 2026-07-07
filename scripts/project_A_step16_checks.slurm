#!/bin/bash
# Project A — Step 16 verification check suite (fast; synthetic epochs=1 in grid-plumbing is the only
# training). Runs every Project-A check module (incl. Step-16 benefit-anatomy / sequential-harm-control
# / policy-frontier + updated harm-control + dashboard) + the counterexample catalog + an informational
# forbidden-wording guard grep. SLURM CPU. NO --time.
#SBATCH --job-name=projA_step16_checks
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/project_A_step16_checks_%j.out
#SBATCH --error=logs/project_A_step16_checks_%j.err

set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

run () { echo "===== $1 ====="; conda run -n icml python $2; echo; }

run "smoke"              "-m h2cmi.tests.test_smoke"
run "counterexamples"    "notes/project_A_observability/counterexamples/run_counterexamples.py"
run "audit"              "-m h2cmi.tests.test_observability_audit"
run "eval-bridge"        "-m h2cmi.tests.test_observability_eval_bridge"
run "harness-contract"   "-m h2cmi.tests.test_eval_bridge_harness_contract"
run "result-index"       "-m h2cmi.tests.test_observability_result_index"
run "text-hygiene"       "-m h2cmi.tests.test_observability_text_hygiene"
run "grid-plumbing"      "-m h2cmi.tests.test_real_audited_grid_plumbing"
run "multidataset"       "-m h2cmi.tests.test_observability_multidataset_summary"
run "paper-package"      "-m h2cmi.tests.test_project_A_paper_package"
run "harm-attribution"   "-m h2cmi.tests.test_observability_harm_attribution"
run "harm-predictor"     "-m h2cmi.tests.test_observability_harm_predictor"
run "harm-power"         "-m h2cmi.tests.test_observability_harm_power"
run "harm-control"       "-m h2cmi.tests.test_harm_control_policies"
run "benefit-anatomy"    "-m h2cmi.tests.test_benefit_anatomy"
run "sequential-control" "-m h2cmi.tests.test_sequential_harm_control"
run "policy-frontier"    "-m h2cmi.tests.test_policy_frontier"
run "minimal-paired"     "-m h2cmi.tests.test_minimal_paired_phase_transition"
run "real-minimal-label" "-m h2cmi.tests.test_real_minimal_label_curves"
run "science-dashboard"  "-m h2cmi.tests.test_observability_science_dashboard"

echo "===== forbidden-wording guard (informational; legacy/retraction/registry/no-SOTA only) ====="
grep -RInE "posterior-KL upper bound|genuine concept shift|zero Bayes error.*necessary|target concept shift detected from unlabeled|source-only target safety certified|GLS gives.*target prior.*source-only|target gain identifiable under R1|oracle gain used as predictor|target labels used in R1 diagnostics|full target risk identified|oracle policy deployable" \
  notes/project_A_observability h2cmi/observability h2cmi/tests \
  h2cmi/run_real_audited.py h2cmi/run_real_audited_grid.py h2cmi/eval/harness.py || echo "  (no forbidden-wording hits)"

echo "ALL STEP-16 CHECKS PASSED"
