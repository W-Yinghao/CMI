# Step 20 — Final Observability-Contract Closeout and Claim Ledger

Terminal synthesis of the Step 12–19 science arc. **No new data, no retraining, no new rescue policy.**
This step only freezes the scientific conclusion, the claim boundaries, the evidence ledger, and the
manuscript-ready wording constraints. The machine-auditable form is `closeout.py` →
`results_summaries/step20_closeout.{json,md}` (headline metrics are pulled LIVE from the tracked
step-digests, not hand-copied).

## Final verdict

> **Observability contracts expose why unlabeled offline-TTA cannot be safely controlled under honest
> prior uncertainty.**

## Evidence ledger (Step 12–19)

| step | question | verdict | headline |
|---|---|---|---|
| 12 | Can source-only (R0) diagnostics predict TTA harm? | **REFUTES** | R0 harm-predictor ≈ chance (TOS-1 ceiling) |
| 13–14 | Do richer R1 target-unlabeled diagnostics predict harm? | **REFUTES** | within permutation null (n_perm=1000) → overfitting artifact |
| 14 | Are the real minimal-label curves accurate or low-coverage? | **CHARACTERIZES** | high precision, coverage-limited; k=0 non-identifiable |
| 15 | Can R2 minimal labels safely control harm (static)? | **REFUTES** | best deployable policy = none |
| 16 | Does sequential acquisition rescue it; is benefit real? | **REFUTES** | best sequential = none; benefit rare/small/unstable |
| 17 | Is the failure an accuracy-vs-bAcc estimand mismatch? | **REFUTES** | class-balanced ⇒ acc==bAcc; Step-16 gap = threshold artifact |
| 18 | Is TTA harm global or class/prior-dependent? | **CHARACTERIZES** | class-specific + prior-dependent (96.3%); only 3.7% harmful-under-all |
| 19 | Any run robustly beneficial under bounded prior uncertainty? | **REFUTES** | no robust benefit at τ≥0.05; median flip-radius ≈0.165 |

Two `CHARACTERIZES` rows (14, 18) are mechanism findings; the six `REFUTES` rows are rescue hypotheses
that failed. Every rescue attempt across the arc failed; the two mechanism findings explain *why*.

## Deployment decision ladder

For each honestly-achievable observability level, the licensed adaptation decision. **No rung licenses a
deployable `adapt`** — adaptation is only ever licensed by the oracle full-target gain, which is an
evaluation-only upper bound.

| observability level | contracts | licensed decision | deployable adapt? |
|---|---|---|:--:|
| R0 (source-only) | — | identity | no |
| R1 (target-unlabeled) | — | identity / abstain | no |
| R1 + C14 (declared point prior) | C14 | abstain | no |
| R1 + C15 (declared prior-uncertainty set) | C15 | block / abstain | no |
| R2 (minimal paired labels) | C13 | abstain | no |
| R2 (full / oracle labels) | — | adapt (evaluation-only) | no |

The honest deployment default is therefore **identity / abstain / block, never adapt**.

## Forbidden headline claims

The project must never assert any of these; the machine registry (`FORBIDDEN_CLAIMS`) and the closeout
flags encode them as `made: false`:

1. **"unlabeled offline-TTA is safe to deploy"** — refuted by Steps 15/16 (minimal-label policies fail)
   and Step 19 (no robust benefit under prior uncertainty).
2. **"the target prior is identified from R0/R1"** — TU-1 boundary: identification needs C1 ∧ C2 ∧ C3;
   C14/C15 only *declare* a prior / prior set.
3. **"prior-robust adaptation benefit exists under honest prior uncertainty"** — Step 19: with a harm
   margin τ ≥ 0.05, no run is robustly beneficial at any ρ.

A note on wording (reviewer): the Step-19 dashboard's `prior_robust_safe_adaptation_exists_at_rho_0_10 =
true` is a **τ=0, sign-level** artifact on a 3.7% minority and must not be used as a headline. The
manuscript-usable statement is the τ≥0.05 result: **no robust safe adaptation exists**.

## Manuscript-ready science summary

1. Under R0/R1 the sign of the offline-TTA target gain is not deployably identifiable; the source-only
   harm predictor is null (TOS-1 ceiling) and richer R1 diagnostics are an overfitting artifact once
   permutation-controlled.
2. R2 minimal-label control fails: neither static nor sequential label-acquisition policies safely select
   adaptation; safe selection requires (near-)full target labels (oracle, non-deployable). The failure is
   estimand-invariant (accuracy == balanced accuracy on this class-balanced grid; the Step-16 gap was a
   benefit-threshold artifact).
3. Mechanistically the harm is class-specific and prior-dependent, not global; the gain sign is fragile
   (median L1 flip-radius ≈ 0.165 from uniform).
4. Under any honest declared prior uncertainty (contract C15), robust adaptation benefit is **unattainable**
   with a usable harm margin (τ ≥ 0.05); the robustly justified actions are abstain or block, never a
   deployable adapt.
5. C14 (declared point prior) and C15 (declared uncertainty set) support counterfactual prior analysis
   only; neither identifies the actual target prior (Prior-Decoupled boundary).

## Claim boundary

No target functional is claimed identifiable under R0/R1; no deployable adaptation is licensed at any
honest observability level; C14/C15 are declared/counterfactual, never identified target priors; the
oracle gain is evaluation-only. Not SOTA. No manuscript is written or modified by this step.
