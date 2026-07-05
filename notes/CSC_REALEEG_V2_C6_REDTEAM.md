# CSC real-EEG v2 — C6 red-team independent re-aggregation

**Verdict: CONFIRMS the reported package = FAIL.** 4 independent adversarial lenses, each re-derived the entire
verdict from the raw 801 `per_cohort` records (NOT the reported verdict block). **All 4 recompute FAIL; all 4
match the reported numbers byte-exactly; 0 serious issues.**

## Lenses (all `matches_reported_verdict = true`, confidence high)
1. **Independent re-aggregation.** Reproduced TIER1/2/3 exactly. Cohort-bootstrap uppers matched to
   **|diff| = 0.00e+00** (byte-identical to the hash-pinned engine's `cohort_bootstrap_upper` AND to an
   independent re-implementation): NULL_cov 0.21 FAIL, NULL_label 0.0 PASS, NULL_cov_plus_label 0.10 FAIL,
   random_label_control 0.0 PASS. Seed formula recovered (seed0 = base+900e6 = 920000000; gating seed0+i by
   list index; TIER2 seed0+100+i). **Seed-robust**: the alternative seed-index interpretation (condition_index)
   yields the same 0.21/0.10, so the FAIL is not seed-fragile.
2. **Denominator + bootstrap integrity (anti-gaming).** DECIDED-only denominator verified (all 400 gating
   records DECIDED; n_confirmed == CONCEPT_CONFIRMED count). invalid_frac = 0.0 < 0.20 for all gating → no
   INCONCLUSIVE masking in either direction. **Counterfactual**: under a naive all-cohorts denominator the
   gating result is IDENTICAL (0 abstentions in gating conditions), so the DECIDED rule neither helped nor hurt
   the FAIL. NULL_label + random_label PASS at exactly 0/100 → the machinery is not trivially failing everything.
3. **Interpretation honesty / overclaim.** NULL_cov ground_truth = "NO_CONCEPT, real COVARIATE shift present" →
   the failure is a genuine type-I false-confirmation of the paired B3 certifier under REAL Lee2019 covariate
   structure with injected null concept. TIER2 power + TIER3 Route A correctly flagged non-gating (cannot flip
   the verdict); genuine contrast descriptive-only; no clinical/PD claim implied. Fair summary confirmed:
   "B3 real-feature safety does NOT hold on Lee2019 — primary NULL_cov type-I gate fails, upper 0.21."
4. **Provenance / completeness.** 801/801 records unique, complete (8×100 cohorts 0..99 + genuine ×1); all 801
   seeds satisfy `base + condition_index*stride + cohort_index` (0 violations); `execution.task_table_sha256`
   recomputed identical; n_tasks==n_tasks_expected==801; resumed==false; schema_version==v2; mode==cohort_parallel;
   BLAS all '1'; bank file sha256 214b97de matches recorded `bank_manifest_sha256`; engine file sha256 d5aaaf78
   matches manifest pin; annotated tag `csc-realeeg-v2` dereferences to `5cce744` == HEAD == `frozen_refs.git_head`,
   tree clean; synthetic tags untouched.

## Consensus
`c6_consensus_package_verdict = FAIL`; `all_lenses_match_reported = true`; `serious issues = 0`.
The result is correctly derived, honestly scoped, and provenance-complete.
