# Step 17 — Estimand Consistency: Results

Scope: scientific exploration on the Step-13 raw diagnostics grids (BNCI2014_001 4-class + BNCI2014_004
binary, 54 runs). **Not SOTA. No new datasets, no retraining.** Oracle target labels are used only to
form R2 labeled slices and to score evaluation-only oracle gains; they are never a predictor feature.

## What was run

- `estimand_consistency.py` — for each run, oracle full-target **accuracy gain** and **balanced-accuracy
  gain**; then policies `{plugin_sign, ci_adapt_only, ci_three_way}` under both estimands and both
  sampling contracts (iid vs class-balanced C13), over `k ∈ {0,1,2,4,8,16,32,64,128,256,full}`,
  `τ ∈ {0,0.01,0.02,0.05}`, 500 repeats.
- `estimand_frontier.py` — one harm/coverage frontier per `(estimand, sampling)` group, kept separate.
- Step-17 dashboard — cross-estimand diagnostics + claim-boundary conjunction.

## Headline finding — the Step-16 "mismatch" was a **threshold artifact**, not an estimand divergence

On this grid **all 54 target sets are perfectly class-balanced** (BNCI2014_001: 144/144/144/144;
BNCI2014_004: 360/360). When classes are balanced, balanced accuracy **equals** ordinary accuracy, so:

- per-run **`max |accuracy_gain − bacc_gain| = 0.0`** across all 54 runs (identical, not merely close);
- `estimand_relationship = identical_on_grid`; `cross_estimand_sign_agreement = 1.0`; 0 sign-disagreement
  runs (neither `runs_accuracy_benefit_bacc_harm` nor `runs_bacc_benefit_accuracy_harm` fires).

The Step-16 gap (0.1481 vs 0.0926) came entirely from an **inconsistent benefit threshold** in
`benefit_anatomy.py`: bAcc benefit used `gain > 0.005` (→ 0.0926) while accuracy benefit used
`gain > 0` (→ 0.1481). At a **shared** threshold the two coincide:

| threshold | accuracy benefit-rate | bAcc benefit-rate |
|---|---:|---:|
| `> 0`     | 0.1481 | 0.1481 |
| `> 0.005` | 0.0926 | 0.0926 |

So the number the Step-17 directive asked us to "fix" was a reporting artifact on class-balanced data,
not a genuine accuracy-vs-bAcc divergence. We under-claim accordingly: **we did not find an
accuracy/bAcc estimand divergence in this real grid.**

## Does the Step-15/16 negative depend on the estimand? — No, it is estimand-invariant here

Because the two gains coincide per run, the beneficial set is identical under both estimands, and the
Step-15/16 negative **persists under bAcc-consistent control**: no minimal-label policy meets
`harm ≤ 0.10` at `coverage ≥ 0.05` for **either** estimand. A policy only "meets" the harm bound at
`k = full` — i.e. using (near-)full target labels, which is oracle-equivalent, not a minimal-label win
(`best_*_control_at_minimal_labels = false` for both). This reproduces the Step-15/16 conclusion:
**safe adaptation on this grid needs (near-)full target labels; minimal labels do not enable safe
adaptation selection**, and switching the estimand from accuracy to balanced accuracy does not change
that.

## The machinery is still correct and necessary — validated where the estimands truly diverge

Class balance is a property of *this* grid, not a law. The separation machinery is validated on
imbalanced data in `test_estimand_consistency.py`:

- `test_accuracy_and_bacc_gain_can_disagree` — a 90/10-imbalanced run where the adapter predicts the
  majority class: **accuracy gain +0.12, bAcc gain −0.29** (accuracy-benefit ∧ bAcc-harm). Here the
  estimands genuinely disagree on sign, and reporting an accuracy-gain policy as a bAcc control would be
  a real error.
- `test_iid_bacc_estimator_abstains_when_class_missing` — a small iid slice that omits the rare class
  yields `bacc_slice_status = missing_class` and the policy abstains (no bAcc gain is fabricated).
- `test_k_less_than_n_classes_bacc_abstains` — class-balanced (C13) with `k < n_classes` →
  `under_class_budget` → abstain.

## Hard rules enforced (tests + dashboard)

- `accuracy_policy_controls_bacc = false` — accuracy-gain and bAcc-gain are different target functionals;
  a policy licensed for one is never reported as controlling the other.
- `no_overall_best_across_estimands = true` — frontiers by `(estimand, sampling)` are never merged into a
  single winner.
- Class-balanced bAcc-gain estimation **requires contract C13**
  (`balanced_accuracy_gain:class_balanced → requires_contract = C13`); iid small-`k` bAcc abstains on a
  missing class rather than guessing.
- k=0 is R1 non-identifiable (abstain); k>0 is an R2 labeled slice under a sampling/design contract, NOT
  R1 target-gain identifiability. Oracle gains are evaluation-only.

## What remains unknown

- Whether the accuracy/bAcc estimands diverge on genuinely class-imbalanced clinical EEG (they must, by
  construction — this grid simply does not exercise it).
- Whether a class-balanced (C13) calibration protocol is feasible in real cued BCI workflows.
- Whether a bAcc-consistent policy could control harm at higher coverage under active (non-iid) sampling.

> Balanced accuracy and ordinary accuracy are distinct target functionals in general, but coincide on
> this class-balanced grid; the Step-16 benefit-rate gap was a threshold artifact; the Step-15/16
> minimal-label negative is estimand-invariant here; k>0 slices are R2 under a sampling/design contract,
> not R1 identifiability. No SOTA.
