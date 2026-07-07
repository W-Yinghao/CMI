# Step 14 — Metric Semantics and Power

## Why Step 14 exists

Step 13 produced two results that were easy to over-read:
- the retrospective **R1 harm-predictor bAcc jumped to 0.652**, but only marginally above its
  permutation-null p95 (~0.64); and
- the real minimal-label **"accuracy" at k=256 looked low (0.32)**, but that number conflated *how
  often the slice commits to a sign* (coverage) with *how accurate it is when it commits*.

Step 14 separates these, and does NOT rerun training or add datasets.

## Metric decomposition (real minimal-label curves)

For each k, instead of a single ambiguous "harm_sign_accuracy" we report:
- **decisive_rate (coverage)** — P(the k-slice CI excludes 0, so a sign is called);
- **unconditional_correct_rate** — P(decisive AND correct sign);
- **conditional_accuracy_given_decisive** — P(correct | decisive), null when never decisive;
- **abstention_rate** — 1 − decisive_rate.

At **k = 0** there is NO estimator licensed under R1: accuracy is **NULL** (not 0.5), decisive_rate 0,
abstention 1.0. Reporting 0.5 would falsely imply a chance-level sign estimator exists.

**Reading (real data).** The burden turns out to be **coverage, not accuracy**: when a slice is
decisive it can be highly accurate, but even k=256 rarely commits. So the practical requirement is
*coverage of decisive harm calls*, which needs either more labels or a decision rule that tolerates
abstention.

## Robustness + power (harm prediction)

- The permutation null is reported at **p90 / p95 / p99** with a larger `--n-perm`; a bAcc is called
  **robust** only if it clears **perm_null_p95 + margin** (default 0.03).
- `harm_power.py` reports the minority-class limitation (n, fraction), the **minimum detectable bAcc**
  (≈ perm p95 + margin) below which a result cannot be distinguished from the overfitting-inflated
  null, and an `underpowered` flag.

## Claim boundary

- Permutation-null significance is still **empirical retrospective** evidence. It does **not** make
  target gain identifiable under R1 (TOS-1 / TU-2).
- k-label curves are **R2 labeled-slice** analyses under iid sampling / coverage contracts, never
  full-target identification.
- Duplicating targets to inflate n is NOT evidence and is not performed.
