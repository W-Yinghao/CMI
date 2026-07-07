# Step 16 — Benefit Anatomy Results

Numbers live in `results_summaries/step16_benefit_anatomy.{json,md}` and the Step-16 dashboard. This
doc frames the questions and the reading.

## Questions

- **How rare are beneficial cells?** `benefit_rate = n_beneficial / n_runs` (oracle bAcc gain > 0.005).
- **Are they target-stable?** For each `(dataset, target)`, is the benefit sign the same across all
  seeds (`sign_consistent`)? `target_sign_consistency_rate` aggregates this.
- **Are gains small or large?** `beneficial_gain_distribution_bacc` (mean / q10 / q50 / q90 / max).
- **Does rarity explain the Step-15 false positives?** With a high harm base-rate, a confident-positive
  labeled slice is more likely a noise false-positive on a harmful cell than a true beneficial cell.

## Reading

The consistent picture from Steps 14–15 predicts, and the anatomy confirms: beneficial cells are
**rare**, their gains are **small** (hard to separate from zero with a modest slice), and they are only
**partially stable** across seeds — so a minimal-label CI test that fires on "confident positive" is
dominated by false positives from the majority harmful class. This is *why* Step-15 policies fail, not
a tuning artifact.

## Claim boundary

Benefit anatomy uses the oracle full-target gain and is **evaluation-only**. It is not observable under
R0/R1 and is never used as a predictor feature or a deployable signal. It characterises the problem; it
does not solve it.
