# Step 18 — Prior-Stress Results

Scope: counterfactual deployment-prior stress on the Step-18 class-wise recall deltas (54 runs).
**Not SOTA. No new datasets, no retraining.** Priors are **declared** (contract C14), not identified;
class deltas are oracle/evaluation-only.

## The exact result

For a class prior `π`, the prior-weighted gain is `gain(π) = Σ_c π_c · recall_delta_c`. Over the simplex
the extreme gains are the extreme class deltas, so the sign is prior-dependent iff
`min_c recall_delta_c < 0 < max_c recall_delta_c`. No model is retrained; this is read off the deltas.

## How many runs are harmful under all priors? — Very few

| category | fraction of 54 runs |
|---|---:|
| **prior-dependent sign** (`min_δ < 0 < max_δ`) | **0.963** |
| **harmful under all priors** (`max_δ ≤ 0`) | **0.037** |
| **beneficial under all priors** (`min_δ ≥ 0`) | **0.000** |
| mean prior-sign-width (`max_δ − min_δ`) | **0.442** |

Under the benchmark **uniform** prior, TTA looks like broad harm (net −0.042, §14). But only **2 / 54**
runs are harmful under *every* deployment prior. For **~96%** of runs there exists a declared class prior
under which the adaptation gain sign flips — the sign-width (0.44) is large relative to the uniform gain
(−0.04).

## Does uniform bAcc hide prior-specific effects? — Both directions

| category | fraction |
|---|---:|
| uniform-**harm** but some-prior-**benefit** | **0.815** |
| uniform-**benefit** but some-prior-**harm** | **0.148** |

- **81.5%** of runs are net-harmful under the uniform prior yet have a class prior under which they help
  — the benchmark-uniform bAcc masks a niche-class benefit.
- **14.8%** are net-beneficial under uniform yet harmful under some deployment prior — a bAcc-positive
  adaptation is *not* prior-robust.

## What this implies for C14 and prior-decoupled reporting

The Step-15/16 measurement→control negative is **not** rescued by this: you still cannot safely select
adaptation, because (a) the class deltas are oracle, and (b) the *actual* deployment prior is
unidentified under R0/R1. What Step 18 establishes is *why* the aggregate looks negative and *how
fragile* that sign is:

- TTA harm on this grid is **class-specific and prior-dependent**, not global (only 3.7% global harm).
- Reporting a single uniform-prior gain — bAcc or accuracy — hides that the sign is a function of the
  deployment prior. This is exactly the **Prior-Decoupled boundary**: without declaring the operating
  prior/utility (contract **C14**), the sign of adaptation gain is under-determined.
- C14 is a *declared* operating condition. It does **not** identify the actual target prior (that needs
  TU-1); the audit engine rejects a `prior_weighted_gain` claim that carries neither C14 nor TU-1, and
  never lets C14 stand in for "the target prior identified source-only".

## Claim boundary

Prior-weighted gains are counterfactual evaluations under declared priors (C14); the actual target prior
is not identified; class deltas are oracle/evaluation-only. `deployment_prior_identified = false`,
`deployment_prior_identified_under_R1 = false`. No SOTA.
