# CSC Route B3-P2.4 — calibrated `pc_centered_calibrated`: real improvement, but residual leakage is DIFFUSE

DEVELOPMENT, simulator-only, `csc/mininfo/`. Does NOT touch the frozen A tag. NO freeze, NO confirmatory,
NO real EEG. No finite-sample type-I control is claimed. Artifact `csc/results/b3_p24_dev_map.json`
(`method_lock` + `calibration_version = p24_pair_integrity_classbalanced_crossfit`). Same grid as P2.3 (10
kinds × 6 scenarios × m{0,10,20,30} × 24 clusters). Method LOCKED; P2.4 added ONLY the four calibration
fixes.

> **Disclosure note (two overconfident drafts, both caught by the independent red-team).** An earlier
> draft framed the residual as "confined to label_noise" and the controls as "clean/covariate clean" and
> "passes all 7". The red-team showed that is NOT supported (below). This note is the corrected, fully
> disclosed version. The pooled rate is genuinely conservative, but the per-cell residual is DIFFUSE.

## The four fixes and what each genuinely closed (P2.3 → P2.4)

| fix | P2.3 failure | P2.4 result |
|---|---|---|
| batch **pair-integrity** guard (≥0.95) | `missing_pair` false-confirmed **2/24** (fail-closed BREACH) | `missing_pair` **0 across ALL cells** ✓ |
| **eligible-complete-pair** (min 8 epochs/cell) | `unequal_epochs_extreme` up to 0.12 | **0 across ALL cells** ✓ |
| **class-balanced** loss + weights | `paired_label` 13 FC (m≥20) | `paired_label` **4 FC** |
| **cross-fitted** T (3 folds) | `random_label` 15 FC; `pure_conditional` 0.00 @ high_subject_tau | `random_label` **8 FC**; `pure_conditional` **0.54** @ high_subject_tau |

These four are real, verified wins: the two **safety breaches are closed**, pooled control FC (m≥20) fell
**0.030 → 0.0119** (below α), and the high_subject_tau `pure_conditional` collapse is resolved.

## Honest assessment vs the 7 promotion criteria (red-team-corrected)

| # | criterion | result |
|---|---|---|
| 1 | `missing_pair` → 0 confirms all cells | **PASS** — 0 total (verified) |
| 2 | no **broad** kind-structured creep | **PARTIAL** — broad pattern gone (pooled 0.0119 < α), BUT per-kind FC (m≥20) is **random_label 8, paired_covariate_plus_label 6 (worst), paired_label 4, paired_covariate 4, clean 2**, unequal 0, missing 0 — creep is now **diffuse**, not eliminated |
| 3 | clean & paired_covariate clean @ m≥20 | **FAIL (strict)** — `clean` false-confirms **2** cells (few_epochs m20 & m30); `paired_covariate` **4** cells (baseline, high_nuisance, imbalanced×2). Not "clean" |
| 4 | concept & concept_plus_cov high power @ m20 | **PARTIAL** — `paired_concept` 1.00 in 5/6; **`paired_concept_plus_cov` 1.00 in only 4/6** (label_noise 0.96); `few_epochs` 0.29 (eligibility abstention — verified NEED_MORE_LABELS, invalid_pair 0, NOT a leak) |
| 5 | `pure_conditional` nonzero @ m30 | **PARTIAL** — 0.54–0.67 in 4 scenarios (incl. high_subject_tau **0.54**, resolved), BUT **label_noise 0.375** and **few_epochs 0.00** (the fragility is *relocated*, not gone) |
| 6 | decomposition interpretable, no silent failures | **PASS** — max mean bootstrap-invalid frac 0.000; all states in the 5-set; power drops are NEED_MORE_LABELS/INVALID |
| 7 | safeguards predeclared + machine-readable | **PASS** — METHOD_LOCK + calibration_version embedded |

## The residual is DIFFUSE (the key correction)

Control false-confirms at m≥20 appear in **all 6 scenarios** (label_noise 10, high_subject_tau 4,
imbalanced 3, few_epochs 3, high_nuisance 2, baseline 2), and contaminate the **purest** controls:
`clean` (2 cells, few_epochs) and `paired_covariate` (4 cells). Worst cells: `label_noise |
paired_covariate_plus_label` 4/24 (CP-up **0.342**), `label_noise | paired_label` 3/24, `high_subject_tau
| random_label` 3/24. Label-bearing controls (paired_label + random_label + covariate_plus_label) total
**18/24** of the m≥20 false-confirms, and a label-control reaches CP-up 0.34 — well above the α=0.05 target.

Whether these per-cell elevations (1–4/24) are **real** or **24-cluster multiple-comparison noise** is
**unresolved at this n** — the pooled rate (0.0119) is consistent with α-level conservatism, but individual
cells (CP-up 0.18–0.34) cannot be ruled above α. This ambiguity is exactly what more clusters would settle.

## Verdict (corrected)

P2.4 is a **substantial, real improvement** — both P2.3 safety breaches closed, pooled control FC halved
to below α, `pure_conditional` high-subject-variance fragility resolved, eligibility guard correctly
abstaining on short records. But it is **NOT a clean pass** under a strict reading of the promotion
criteria: the residual control leakage is **diffuse across all scenarios** (including `clean` and
`paired_covariate`), `concept_plus_cov` is not 1.00 everywhere, and `pure_conditional` fragility is
relocated (label_noise 0.375, few_epochs 0.00) rather than eliminated. Real-vs-noise on the per-cell
residual is undetermined at 24 clusters.

## Recommendation (reviewer's call) — still NO freeze/confirmatory/real-EEG

Given two overconfident drafts and a diffuse residual, the conservative path is **not** to jump to a freeze
on a "passes all 7" claim that the strict reading does not support. Options:

- **(2a) Resolve real-vs-noise first.** Re-run the controls at **48 clusters/cell** (reviewer offered
  this) to test whether the per-cell elevations (incl. `clean`/`paired_covariate`) are genuine or
  small-sample noise. Cheap, decisive, and directly answers the one open question. If pooled stays <α and
  no control cell is robustly >α, that materially strengthens a freeze case.
- **(2b) + label-control null fix.** Additionally add a prior-robust / label-composition-robust null for
  the label-bearing controls if 48 clusters confirm a real label_noise elevation.
- **(1) Proceed to freeze-candidate anyway** with FULL disclosure of the diffuse residual (pooled <α; some
  per-cell control cells up to CP-up 0.34) and let the unseen confirmatory bound it — defensible only
  because the pooled rate is conservative, but it freezes an undisclosed-real-or-noise residual.

My lean: **(2a)** — one 48-cluster controls run cleanly separates "conservative on average with noisy
cells" from "genuinely leaky", which is the crux the freeze decision hinges on, and avoids a third
overconfident call. Then freeze if it comes back clean.
