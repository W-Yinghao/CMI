# Step 19 — Prior-Robust Policy Results

Scope: worst-case adaptation decision rule over declared L1 prior-uncertainty balls, on the Step-19
frontier (54 runs). **Not SOTA. No new datasets, no retraining.** Oracle/evaluation-only; not a
deployable selector.

## The rule

For a declared uncertainty ball `U_ρ` and harm margin `τ`:

```
adapt   if robust_lower(ρ) > τ      (robustly beneficial over the whole set)
block   if robust_upper(ρ) < −τ     (robustly harmful -> keep identity)
abstain otherwise                   (sign ambiguous under the declared uncertainty)
```

## Can declared prior-uncertainty sets support safe adaptation? — No

| ρ | τ | adapt_cov | robust_harm_block | abstain |
|---:|---:|---:|---:|---:|
| 0.05 | 0.05 | **0.000** | 0.204 | 0.796 |
| 0.05 | 0.10 | 0.000 | 0.074 | 0.926 |
| 0.05 | 0.20 | 0.000 | 0.000 | 1.000 |
| 0.10 | 0.05 | 0.000 | 0.167 | 0.833 |
| 0.10 | 0.10 | 0.000 | 0.056 | 0.944 |
| 0.20 | 0.05 | 0.000 | 0.148 | 0.852 |
| 0.50 | 0.05 | 0.000 | 0.037 | 0.963 |

**Adaptation coverage is 0.0 at every (ρ, τ) with harm margin τ ≥ 0.05.** No run is robustly beneficial
enough to clear even a 0.05 margin under an L1 ≥ 0.05 ball, so the best prior-robust policy is **none**.

## What is the adaptation coverage under robust-positive criteria?

Zero, with a harm margin. (At the bare zero-margin sign level, 3.7% of runs are robustly beneficial at
ρ = 0.10 — see [18_prior_uncertainty_results.md](18_prior_uncertainty_results.md) — but that vanishes once
a harm margin τ ≥ 0.05 is required and once ρ ≥ 0.20.)

## Does prior robustness mostly abstain? — Yes

As ρ or τ grows, decisions move overwhelmingly to **abstain** (up to 100% at τ = 0.20). Robust **block**
(identity is robustly safest) covers a meaningful minority at small margins (20.4% at ρ = 0.05, τ = 0.05)
and shrinks as the declared uncertainty grows.

## Consistency check

A robust `adapt` requires the worst-case gain over `U_ρ` (which contains the uniform prior) to exceed
τ ≥ 0, so it can never be harmful under the benchmark uniform prior. The summary reports
`robust_adapt_never_uniform_harmful = true` (trivially satisfied here, since no run adapts).

## What does this imply for C14/C15 deployment reporting?

- Reporting a single-prior gain (C14 point prior, e.g. uniform bAcc) overstates safety: the sign is
  fragile (median flip radius ≈ 0.165).
- Robustness must be declared as a **set** (C15). Once it is, offline TTA cannot be certified beneficial
  here — the only robustly justified actions are **block** (for the persistently-harmful minority) or
  **abstain**.
- Neither C14 nor C15 identifies the actual target prior. This is not a deployable selector: the class
  deltas are oracle and the true operating prior is unidentified under R0/R1. It quantifies how a
  prior-decoupled, honest robustness accounting turns the Step-15/16 measurement→control gap into an
  explicit *abstain-or-block* conclusion under declared uncertainty.

## Claim boundary

Worst-case decision rule over declared L1 prior-uncertainty sets (C15); class deltas are
oracle/evaluation-only; not a deployable selector; does not identify the actual target prior. No SOTA.
