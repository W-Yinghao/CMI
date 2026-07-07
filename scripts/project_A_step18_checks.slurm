#!/bin/bash
# Project A — Step 18 verification check suite (fast; synthetic epochs=1 in grid-plumbing is the only
# training). Runs every Project-A check module (incl. Step-18 harm-mechanisms / prior-stress + C14 audit
# + updated dashboard) + the counterexample catalog + guards: a HARD affirmative-claim guard (estimand
# cross-claim + prior-decoupled overclaim) and an INFORMATIONAL legacy/A-naming grep. SLURM CPU. NO --time.
#SBATCH --job-name=projA_step18_checks
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/project_A_step18_checks_%j.out
#SBATCH --error=logs/project_A_step18_checks_%j.err

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
run "minimal-paired"     "-m h2cmi.tests.test_minimal_paired_phase_transition"
run "real-minimal-label" "-m h2cmi.tests.test_real_minimal_label_curves"
run "science-dashboard"  "-m h2cmi.tests.test_observability_science_dashboard"

DIRS="notes/project_A_observability h2cmi/observability h2cmi/tests"
FILES="h2cmi/run_real_audited.py h2cmi/run_real_audited_grid.py h2cmi/eval/harness.py"

echo "===== HARD affirmative-claim guard (estimand cross-claim + prior-decoupled overclaim) ====="
# Only AFFIRMATIVE claims are violations. Labels legitimately render "... : **False**" and the docs /
# registry DEFINE the forbidden phrases as prohibitions ("never read as ...", denylist entries). So we
# (a) match the affirmative forms, then (b) DROP any hit sitting in a negation/denial context. A real
# affirmative claim has none of those negation cues and survives -> hard fail (exit 2).
AFF=$(grep -RInE "accuracy policy controls b[Aa]cc:? +\*\*[Tt]rue\*\*|accuracy[- ]gain policy controls balanced[- ]accuracy|deployment prior identified(_under_r1)?:? +\*\*[Tt]rue\*\*|declared (deployment )?prior identifies the (actual )?target prior|target prior identified source[- ]only|prior[- ]weighted gain (holds|identified) under all (target )?priors" \
  $DIRS $FILES 2>/dev/null \
  | grep -viE "never|must not|not be read|not read as|not identif|stand in for|forbidden|denylist|does not|is not|no target functional|# c14|c14 must" || true)
if [ -n "$AFF" ]; then echo "$AFF"; echo "FORBIDDEN AFFIRMATIVE CLAIM FOUND"; exit 2; else
  echo "  (no affirmative overclaim after negation filter — good)"; fi

echo "===== legacy forbidden-wording guard (informational; retraction/registry/no-SOTA) ====="
grep -RInE "posterior-KL upper bound|genuine concept shift|zero Bayes error.*necessary|target concept shift detected from unlabeled|source-only target safety certified|GLS gives.*target prior.*source-only|target gain identifiable under R1|oracle gain used as predictor|target labels used in R1 diagnostics|full target risk identified|oracle policy deployable|actual target prior identified" \
  $DIRS $FILES || echo "  (no legacy forbidden-wording hits)"

echo "===== A0-A6 legacy-naming guard (informational; allowed contexts excluded) ====="
grep -RInE "\bA[0-6]\b" $DIRS | grep -viE "legacy|retract|never|not a[0-6]|instead of a[0-6]|collides|A100|A40" \
  || echo "  (no bare A0-A6 theorem-naming hits)"

echo "ALL STEP-18 CHECKS PASSED"
