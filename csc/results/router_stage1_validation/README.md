# Router Stage R1 — fresh calibration + held-out validation (diagnostic-only)

```
Scope: Router R1 fresh calibration + held-out validation | threshold calibration only
  not a concept certificate | not confirmatory tag | not deployable claim
  no method/h0/h1/statistic/feature change | no oracle feature use
  primary safety evaluated kind-specifically (NULL_cov & NULL_cov_plus_label NOT pooled)
```
Rule (MONOTONE): ALLOW if B3 method_confirm AND observed_T>=tau, else ABSTAIN (allow-set is a strict SUBSET of
B3 confirms). Fresh seeds: calibration base 80e6 (NULL_cov, NULL_cov_plus_label, null-only), held-out validation
base 90e6 (all 6 conditions). tau_R1 locked on calibration (smallest tau with both primary nulls <=3/300); the
LOCKED tau_R1 is evaluated once on the held-out block. Commit order: protocol -> calibration -> tau lock ->
validation -> red-team -> results. No tag.
