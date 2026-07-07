#!/bin/bash
# Project A — Step 20 verification check suite (fast; synthetic epochs=1 in grid-plumbing is the only
# training). Runs every Project-A check module (incl. Step-20 closeout) + the counterexample catalog +
# guards: a HARD value-anchored affirmative-claim guard (with negation filter) and INFORMATIONAL
# legacy/A-naming greps. SLURM CPU. NO --time.
#SBATCH --job-name=projA_step20_checks
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/project_A_step20_checks_%j.out
#SBATCH --error=logs/project_A_step20_checks_%j.err

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
run "estimand-consistency" "-m h2cmi.tests.test_estimand_consistency"
run "estimand-frontier"  "-m h2cmi.tests.test_estimand_frontier"
run "harm-mechanisms"    "-m h2cmi.tests.test_harm_mechanisms"
run "prior-stress"       "-m h2cmi.tests.test_prior_stress"
run "prior-uncertainty"  "-m h2cmi.tests.test_prior_uncertainty"
run "prior-robust-policy" "-m h2cmi.tests.test_prior_robust_policy"
run "closeout"           "-m h2cmi.tests.test_closeout"
run "minimal-paired"     "-m h2cmi.tests.test_minimal_paired_phase_transition"
run "real-minimal-label" "-m h2cmi.tests.test_real_minimal_label_curves"
run "science-dashboard"  "-m h2cmi.tests.test_observability_science_dashboard"

DIRS="notes/project_A_observability h2cmi/observability h2cmi/tests"
FILES="h2cmi/run_real_audited.py h2cmi/run_real_audited_grid.py h2cmi/eval/harness.py"

echo "===== HARD affirmative-claim guard (value-anchored booleans; negation-filtered) ====="
AFF=$(grep -RInE "accuracy policy controls b[Aa]cc:? +\*\*[Tt]rue\*\*|deployment prior identified(_under_r1)?:? +\*\*[Tt]rue\*\*|actual target prior identified:? +\*\*[Tt]rue\*\*|tta (is )?safe to deploy:? +\*\*[Tt]rue\*\*|accuracy[- ]gain policy controls balanced[- ]accuracy" \
  $DIRS $FILES 2>/dev/null \
  | grep -viE "never|must not|not be read|not read as|not identif|stand in for|forbidden|denylist|does not|is not|no target functional|# c1[45]|c1[45] must" || true)
if [ -n "$AFF" ]; then echo "$AFF"; echo "FORBIDDEN AFFIRMATIVE CLAIM FOUND"; exit 2; else
  echo "  (no affirmative overclaim after negation filter — good)"; fi

echo "===== legacy forbidden-wording guard (informational; retraction/registry/prior/closeout overclaims) ====="
grep -RInE "posterior-KL upper bound|genuine concept shift|zero Bayes error.*necessary|target concept shift detected from unlabeled|source-only target safety certified|GLS gives.*target prior.*source-only|target gain identifiable under R1|oracle gain used as predictor|target labels used in R1 diagnostics|full target risk identified|oracle policy deployable|actual target prior identified|declared prior-uncertainty set identifies the actual target prior|gain reported as prior-robust without a declared prior-uncertainty set|prior robust without C15|unlabeled offline-TTA is safe to deploy|prior-robust adaptation benefit exists" \
  $DIRS $FILES || echo "  (no legacy forbidden-wording hits)"

echo "===== A0-A6 legacy-naming guard (informational; allowed contexts excluded) ====="
grep -RInE "\bA[0-6]\b" $DIRS | grep -viE "legacy|retract|never|not a[0-6]|instead of a[0-6]|collides|A100|A40" \
  || echo "  (no bare A0-A6 theorem-naming hits)"

echo "ALL STEP-20 CHECKS PASSED"
