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

## Held-out RESULT @ locked tau_R1 = 0.004587196 (merged sha256 065c681b…)
allow counts /300: NULL_cov=5 (CP95u 0.0347), NULL_cov_plus_label=4 (0.0303), NULL_label=1, random_label_control=0,
POS_concept=52, POS_concept_plus_cov=53. PRIMARY safety PASS (both primary nulls <=7 cap, CP95u<5%); SECONDARY ok;
utility genuinely concept-driven (observed_T separates POS from covariate-null at AUC~0.90).

## Red-team (3 lenses; `r1_redteam_checks.json`) — COMMIT-CLEAN, but down-scoped
- **nopeek/tau-lock PASS:** tau re-derives byte-exact (hexfloat 3f72ca060cce336e, rank 81/88); calibration null-only;
  eval loads locked tau (never re-derives); monotonicity holds (allow strict subset of confirm); invalid accounting
  conservative; no_retune honored; all sha256 pass.
- **safety_real / utility_real MINOR_ISSUE:** the SAFETY-PASS is REAL but SINGLE-BLOCK and THIN — verified 2x
  calibration->validation drift (allow 3->5), tau*0.90 breaches the cap, tested only against a SOFT covariate
  (session_auc~0.52); utility is genuine concept-detection but ~17% recall on an EASY linear concept.
- **Down-scoped verdict (supersedes the auto-printed 'STRONG->eligible' string):** *development-diagnostic;
  single-block SAFETY-PASS with a thin, drift-exposed margin and ~17% power against a soft covariate — candidate for
  frozen-protocol DESIGN DISCUSSION, not a confirmatory result.* Path to confirmatory: multiple independent validation
  blocks + a strong-covariate null + a second dataset / nonlinear concept. See notes/router_stage1_validation.md.
