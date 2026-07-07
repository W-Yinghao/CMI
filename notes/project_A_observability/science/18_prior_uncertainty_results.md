# Step 19 — Prior-Uncertainty Frontier Results

Scope: exact L1-ball robust gain bounds on the Step-18 class-wise recall deltas (54 runs). **Not SOTA.
No new datasets, no retraining.** Class deltas are oracle/evaluation-only; prior sets are declared (C15).

## What is the median L1 radius to flip sign from uniform?

| statistic | L1 radius from uniform |
|---|---:|
| q25 | **0.073** |
| **median** | **0.165** |
| q75 | **0.304** |
| runs unflippable over the whole simplex | **2 / 54** |

The gain sign is fragile: for half the runs an L1 prior shift of ~0.165 (out of a maximum of 2) already
flips the sign. Only 2 runs (the ones harmful under all priors — all class deltas ≤ 0) cannot flip.

## How many runs flip under small prior shifts?

| flip within L1 ≤ | fraction of runs |
|---|---:|
| 0.10 | **0.278** |
| 0.20 | **0.630** |
| 0.50 | **0.870** |

By L1 ≤ 0.20, nearly two-thirds of runs have flipped; by ≤ 0.50, seven in eight.

## Robust harm / benefit / ambiguity by radius

| ρ | robust_harm | ambiguous | robust_benefit |
|---:|---:|---:|---:|
| 0.00 | 0.852 | 0.000 | 0.148 |
| 0.05 | 0.759 | 0.167 | 0.074 |
| 0.10 | 0.685 | 0.278 | 0.037 |
| 0.20 | 0.370 | 0.630 | 0.000 |
| 0.30 | 0.296 | 0.704 | 0.000 |
| 0.50 | 0.130 | 0.870 | 0.000 |
| 1.00 | 0.037 | 0.963 | 0.000 |
| 2.00 | 0.037 | 0.963 | 0.000 |

Two things stand out:

- **robust_benefit collapses to 0 by ρ = 0.20.** The 14.8% that look beneficial at the exact-uniform
  prior all have a nearby admissible prior that flips them; none survive a declared L1 ≥ 0.20 ball.
- **robust_harm decays but persists.** Even at ρ = 0.5, 13.0% of runs are harmful under *every*
  admissible prior — for those, blocking adaptation (keep identity) is robustly justified. As ρ grows the
  bulk moves into *ambiguous* (96.3% at ρ = 2), matching the Step-18 full-simplex prior-dependence.

## Are runs mostly robust harm, robust benefit, or ambiguous?

Under small uncertainty (ρ ≤ 0.10) most runs are still **robust harm** (68.5% at ρ = 0.10). As the
declared uncertainty grows, they move into **ambiguous**, never into **robust benefit**. Robust benefit is
essentially unreachable under any non-trivial declared prior uncertainty.

## Claim boundary

Robust bounds are exact over declared L1 prior-uncertainty sets (C15); class deltas are
oracle/evaluation-only; the actual target prior is not identified. No SOTA.
