# Step 19 — Prior-Robustness Results (overview)

Scope: overview of the Step-19 prior-uncertainty robustness frontier on the Step-18 class-wise recall
deltas (54 runs). **Not SOTA. No new datasets, no retraining.** Class deltas are oracle/evaluation-only;
prior-uncertainty sets are declared (contract C15), not identified.

## Headline

The offline-TTA gain sign is **fragile** to the deployment prior, and **no run is robustly beneficial**
under even a modest declared prior-uncertainty ball:

- median L1 flip-radius from uniform ≈ **0.165** (q25 0.073, q75 0.304); **62.96%** of runs flip within
  L1 ≤ 0.20; only **2/54** cannot flip over the whole simplex.
- robust-benefit fraction: **14.8%** at the point-uniform prior → **3.7%** at ρ=0.10 → **0%** at ρ≥0.20.
- with a harm margin τ≥0.05, **no** (ρ, τ) yields any robustly-beneficial run — the best prior-robust
  policy is **none**.
- robust-harm (identity/block robustly justified) stays meaningful: **13.0%** even at ρ=0.5; under
  bounded prior uncertainty most decisions become **abstain**.

Details: [18_prior_uncertainty_results.md](18_prior_uncertainty_results.md) (frontier),
[19_prior_robust_policy_results.md](19_prior_robust_policy_results.md) (policy).

## Interpretation

This sharpens the Step-18 negative. Step 18 said "some prior can flip the sign"; Step 19 quantifies *how
little* prior shift is needed (median L1 ≈ 0.165) and shows that robust benefit is unattainable once the
declared uncertainty is non-trivial. It does not make adaptation safe — the class deltas are oracle and
the true operating prior is unidentified under R0/R1 — but it establishes that under any honest
declaration of prior uncertainty, offline TTA cannot be certified beneficial, while abstention/identity is
often the only robustly justified action.

## Claim boundary

Robust gain bounds are over declared prior-uncertainty sets (C15); class deltas are oracle/evaluation-only;
the actual target prior is not identified. No SOTA.
