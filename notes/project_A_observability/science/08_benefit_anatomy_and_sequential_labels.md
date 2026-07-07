# Step 16 — Benefit Anatomy and Sequential Label-Acquisition Frontier

## Motivation

Step 15 showed no deployable minimal-label policy safely adapts under harm ≤ 0.05, while the oracle
full-label policy shows safe adaptation is possible only if beneficial targets can be identified.

Step 16 asks:
- **Where are the beneficial cells?** Are they stable by dataset, target subject, or seed?
- **Sequential labels:** if we may keep acquiring labels batch by batch under an R2 iid sampling
  contract, is there a budget at which harm becomes controllable with useful coverage?
- Is the Step-15 failure caused by rare benefit, noisy slice estimates, or lack of target-level
  separability?
- Is harm ≤ 0.05 simply too strict? (Policy frontier over harm thresholds.)

## Regime boundary

- All full-target gain labels are **oracle/evaluation-only** (benefit anatomy). Not R0/R1 observable.
- Sequential policies with k > 0 are **R2 labeled-slice** procedures under iid sampling / coverage
  contracts. They do NOT identify target gain under R1.
- `oracle_full_target` is an evaluation-only upper bound, never deployable. A deployable sequential
  policy at `budget = full` is a full-label **calibration** policy (marked `calibration_burden = full`),
  distinct from the oracle policy.

## Modules

- `benefit_anatomy.py` — oracle-only rarity / per-(dataset,target) sign-consistency / gain distribution.
- `sequential_harm_control.py` — seq_ci_three_way / seq_ci_adapt_only / seq_plugin_confirm over label
  budgets; predeclared best rule (harm ≤ 0.05, coverage ≥ 0.05, minimize mean labels, tie-break max
  coverage then min missed benefit); oracle excluded.
- `policy_frontier.py` — combines Step-15 static and Step-16 sequential cells; reports whether any
  deployable policy meets harm thresholds 0.05 / 0.10 / 0.20 / 0.50 at min coverage 0.01 / 0.05 / 0.10.

## Non-goals

- no new dataset · no retraining · no SOTA claim · no manuscript writing.
