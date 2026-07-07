# Step 19 — Prior-Uncertainty Robustness Frontier

## Motivation

Step 18 showed that most offline-TTA gain signs are prior-dependent over the full simplex. But the full
simplex is an extreme uncertainty set.

Step 19 asks:
- how far from the benchmark-uniform prior must the operating prior move to flip the gain sign?
- which runs are robustly harmful or robustly beneficial under bounded prior uncertainty?
- does declared-prior robustness produce a useful frontier?

## Regime boundary

All class-wise recall deltas are oracle/evaluation-only. Prior-uncertainty sets are declared external
operating assumptions (contract **C15**). They do not identify the actual target prior under R0/R1.

## Prior sets

```
U_L1(ρ) = { π ∈ simplex : ||π − u||_1 ≤ ρ }
```

where `u` is the benchmark-uniform prior. We report, for a class-delta vector `d`:

- worst-case gain over `U_L1(ρ)`: `robust_lower(ρ) = min_{π∈U_ρ} <π,d>`
- best-case gain over `U_L1(ρ)`: `robust_upper(ρ) = max_{π∈U_ρ} <π,d>`
- whether the gain sign is robust over `U_L1(ρ)`
- minimal L1 radius from uniform needed to flip the sign

## Exact optimizer

Because `<π,d>` is linear and `||π − u||_1 = 2·(mass moved)`, the extrema are found EXACTLY by greedy
mass transfer: to minimize, move up to `ρ/2` mass from the highest-delta classes (capacity `u_c` each)
to the lowest-delta classes (capacity `1 − u_c` each), while the transfer still lowers the objective;
mirror for the maximum. This is validated in `test_prior_uncertainty.py` against the binary closed form
and a brute-force simplex grid (3 classes). The minimal flip radius is where `robust_lower` (or
`robust_upper`, for a negative uniform gain) crosses zero; `None` if the sign cannot flip even over the
whole simplex (`ρ = 2`).

## Claim boundary

- **C14** declares a *point* prior or utility.
- **C15** declares a *prior-uncertainty set* / robustness criterion.
- Neither C14 nor C15 identifies the actual target prior under R0/R1 (that needs TU-1). The machine audit
  gates `robust_prior_weighted_gain` on C15; a point prior (C14) does not certify robustness over a set.

## Results

See [17_prior_robustness_results.md](17_prior_robustness_results.md) (overview),
[18_prior_uncertainty_results.md](18_prior_uncertainty_results.md) (flip radii / sign fractions), and
[19_prior_robust_policy_results.md](19_prior_robust_policy_results.md) (robust policy).

## Non-goals

- no new datasets
- no retraining
- no SOTA claim
- no manuscript writing
