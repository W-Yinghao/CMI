# Step 15 — Coverage-Aware Harm-Control Policies

## Motivation

Step 14 showed the real minimal-label curves are **coverage-limited, not inaccurate**: a small k of
labeled target trials produces **high-precision** harm-sign calls when decisive (conditional accuracy
0.81 at k=8 → 0.998 at k=256), but most cells remain **undecided** (coverage 0.32 at k=256).

This motivates a harm-control framing: do NOT force every target to be adapted. Use minimal target
labels to choose one of three actions per target:
- **adapt** — use the offline-TTA adapted model;
- **identity** — keep the source model / no adaptation;
- **abstain** — do not auto-decide; request more labels or manual calibration.

## Regime boundary

- **k = 0** — R1 target-unlabeled. The gain sign is **non-identifiable**; a label-based policy must
  **abstain** (it cannot license adapt/identity from a target quantity it cannot observe).
- **k > 0** — R2 labeled-slice evidence under an **iid sampling / coverage contract**. A policy may
  estimate the labeled-slice gain and decide. It does NOT identify full target risk without that
  contract.

## Estimator

For a sampled labeled slice `S_k`, the paired per-trial accuracy gain is
`delta_i = 1[adapt_pred_i == y_i] - 1[identity_pred_i == y_i]`, `gain_hat = mean(delta_i)`, with a
finite-sample CI (paired normal approximation by default; bootstrap optional). For `k < 2` the CI is
`[-1, 1]` (not decisive). This is empirical; it is a population statement only under the declared iid
sampling contract.

## Policy families

- **P0 `always_identity`** — never adapt. No adaptation harm, misses all benefit.
- **P1 `always_adapt`** — always adapt. The benchmark that drove the high harm-rate in the audited grids.
- **P2 `plugin_sign`** — adapt if `gain_hat > tau` else identity (k=0 → abstain). High variance; unsafe baseline.
- **P3a `ci_adapt_only_abstain`** — adapt if `ci_low > tau`, else **abstain** (ask for more labels).
- **P3b `ci_adapt_only_identity`** — adapt if `ci_low > tau`, else **identity** (safety-first fallback).
- **P4 `ci_three_way`** — adapt if `ci_low > tau`; identity if `ci_high < -tau`; else abstain.
- **P5 `oracle_full_target`** — adapt if full-target gain > 0 else identity. **Evaluation-only upper
  bound; FORBIDDEN as a deployment policy** and never selected as best-deployable.

## Metrics (per policy / k / tau)

- `adaptation_coverage = P(action == adapt)`
- `decision_coverage = P(action in {adapt, identity})`
- `abstention_rate = P(action == abstain)`
- `harm_rate_among_adapt_decisions = P(oracle_gain < 0 | action == adapt)`
- `prevented_harm_rate_vs_always_adapt` = among harmful cells, fraction NOT adapted
- `missed_benefit_rate` = among beneficial cells, fraction NOT adapted
- `conditional_action_accuracy = P(action == oracle_best_action | action != abstain)`
- `expected_oracle_acc_of_chosen_action` (non-abstain)

## Best deployable policy (predeclared selection rule)

Among **deployable** (non-oracle) policies × (k, tau) cells with `adaptation_coverage > 0`:
1. keep only cells with `harm_rate_among_adapt_decisions <= 0.05`;
2. maximize `adaptation_coverage`;
3. tie-break: minimize `missed_benefit_rate`.
If none qualifies → `best_policy = null`, reason `no deployable policy meets harm constraint`.
The oracle policy is never selected.

## Claim boundary

These are R2 policy-evaluation experiments under an iid sampling contract. They are NOT R1 target-gain
identification, NOT source-only adaptation claims, and NOT a SOTA claim. Oracle full-target labels are
used only for evaluation and for the forbidden upper-bound policy.
