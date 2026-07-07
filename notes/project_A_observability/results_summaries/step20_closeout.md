# Step 20 — Final Observability-Contract Closeout and Claim Ledger

Scope: final observability-contract closeout and claim ledger; no new data / retraining / rescue; not SOTA.

## Final verdict

> **Observability contracts expose why unlabeled offline-TTA cannot be safely controlled under honest prior uncertainty.**

## Evidence ledger (Step 12-19)

| step | question | verdict | headline metric | value |
|---|---|---|---|---|
| Step 12 | Can source-only (R0) diagnostics predict TTA harm? | **REFUTES** | R0 harm-predictor bAcc | 0.3342 |
| Step 13-14 | Do richer R1 target-unlabeled diagnostics predict harm? | **REFUTES** | harm-predictor verdict | above_baseline_but_within_permutation_null_overfitting_artifact |
| Step 14 | Are the real minimal-label curves accurate or merely low-coverage? | **CHARACTERIZES** | k=0 status | not_identified_R1 |
| Step 15 | Can R2 minimal labels safely control harm (static policies)? | **REFUTES** | best deployable policy | None |
| Step 16 | Does sequential acquisition rescue it, and is benefit real? | **REFUTES** | oracle benefit-rate (bAcc) | 0.0926 |
| Step 17 | Is the failure an accuracy-vs-bAcc estimand mismatch? | **REFUTES** | max |acc-bAcc gain| | 0.0 |
| Step 18 | Is TTA harm global or class/prior-dependent? | **CHARACTERIZES** | prior-dependent-sign fraction | 0.963 |
| Step 19 | Is any run robustly beneficial under bounded prior uncertainty? | **REFUTES** | robust safe adaptation exists (margin) | False |

## Deployment decision ladder

No rung licenses a deployable adaptation: **True**.

| observability level | contracts | licensed decision | deployable adapt? |
|---|---|---|:--:|
| R0 (source-only) | — | identity | no |
| R1 (target-unlabeled) | — | identity/abstain | no |
| R1 + C14 (declared point prior) | C14 | abstain | no |
| R1 + C15 (declared prior-uncertainty set) | C15 | block/abstain | no |
| R2 (minimal paired labels) | C13 | abstain | no |
| R2 (full / oracle labels) | — | adapt (evaluation-only) | no |

## Forbidden headline claims (none is made)

- **unlabeled offline-TTA is safe to deploy** — FORBIDDEN (made: False). Step 15/16 (minimal-label policies fail) and Step 19 (no robust benefit under prior uncertainty) refute it.
- **the target prior is identified from R0/R1** — FORBIDDEN (made: False). TU-1 boundary: identification needs C1 AND C2 AND C3; C14/C15 only DECLARE a prior / prior set.
- **prior-robust adaptation benefit exists under honest prior uncertainty** — FORBIDDEN (made: False). Step 19: with a harm margin tau>=0.05 no run is robustly beneficial at any rho.

## Manuscript-ready science summary

1. Under R0/R1 the sign of the offline-TTA target gain is not deployably identifiable; the source-only harm predictor is null (TOS-1 ceiling) and richer R1 diagnostics are an overfitting artifact once permutation-controlled.
2. R2 minimal-label control fails: neither static nor sequential label-acquisition policies safely select adaptation; safe selection requires (near-)full target labels (oracle, non-deployable). The failure is estimand-invariant (accuracy == balanced accuracy on this class-balanced grid; the Step-16 gap was a benefit-threshold artifact).
3. Mechanistically the harm is class-specific and prior-dependent, not global; the gain sign is fragile (median L1 flip-radius ~0.165 from uniform).
4. Under any honest declared prior uncertainty (contract C15), robust adaptation benefit is UNATTAINABLE with a usable harm margin (tau>=0.05); the only positive is a zero-margin sign-level artifact on a vanishing minority. The robustly justified actions are abstain or block, never a deployable adapt.
5. C14 (declared point prior) and C15 (declared uncertainty set) support counterfactual prior analysis only; neither identifies the actual target prior (Prior-Decoupled boundary).

> Terminal closeout: no target functional is claimed identifiable under R0/R1; no deployable adaptation is licensed at any honest observability level; C14/C15 are declared/counterfactual, never identified target priors; the oracle gain is evaluation-only. No SOTA.
