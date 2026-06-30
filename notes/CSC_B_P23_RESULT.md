# CSC Route B3-P2.3 — expanded development map (method LOCKED = `pc_centered`)

DEVELOPMENT, simulator-only, `csc/mininfo/`. Does NOT touch the frozen A tag. NO freeze, NO confirmatory,
NO real EEG. No finite-sample type-I control is claimed. Artifact `csc/results/b3_p23_dev_map.json`
(`METHOD_LOCK` embedded). 10 kinds (7 controls + 3 positives) × 6 star-grid scenarios × m∈{0,10,20,30} ×
24 clusters/cell, single locked method `pc_centered` (h1_basis=pc, condition_coding=centered, rank=3,
C=0.5, min_confirm_pairs=20).

## Assessment vs the pre-registered promotion criteria

| # | criterion | result |
|---|---|---|
| 1 | m=0 → no CONCEPT_CONFIRMED (any kind) | **PASS** — 0.00 everywhere |
| 2 | clean/cov/label/random controls don't inflate @ m≥20 | **PARTIAL (broader than first stated — see Red-team correction)** — pooled control FC rate 0.030, but **35/84 control cells @ m≥20 have ≥1 FC and 16/84 have ≥2/24**; creep is **kind-structured** (random_label 15, unequal_epochs_extreme 14, paired_label 13) not merely scenario-confined |
| 3 | `concept` & `concept_plus_cov` high @ m=20 | **PASS (strong)** — **1.00 (CP-low 0.88) in ALL 6 scenarios** (concept_plus_cov 0.96 once, label_noise m30) |
| 4 | `pure_conditional` nonzero & interpretable @ m=30 | **FRAGILE** — 0.75 baseline/high_nuisance, 0.62 imbalanced, 0.42 noise, **0.00 under high_subject_tau** |
| 5 | missing-pair / invalid → fail closed | **FAIL (corrected)** — `missing_pair` **DOES false-confirm 2/24** (`high_subject_tau` m20 1/24; `few_epochs` m30 1/24): a CONCEPT_CONFIRMED slipped the pair audit. All-unpaired still → INVALID (test_b3 #2), but the guard is **not** a clean fail-closed invariant under stress |
| 6 | failure decomposition interpretable, no silent failures | **PASS** — max mean bootstrap-invalid frac = **0.000**; class-coverage failures = **0**; all states in the defined 5-set |
| 7 | `pc_centered` is the only method into freeze | **PASS** — locked; full_z diagnostic-only |

### Red-team correction (independent verifier audit of this note vs the artifact)

An independent agent re-aggregated all 240 cells and corrected three things in an earlier draft of this
note (now fixed above): (1) criterion 5 is a **FAIL, not PASS** — `missing_pair` false-confirms in 2 cells
(2/24 total), so the advertised pair-audit "fail-closed" invariant is breached under stress; (2) criterion
2's creep is **kind-structured** (random_label / unequal_epochs_extreme / paired_label leak across
scenarios — `unequal_epochs_extreme` leaks 2/24 even at *baseline*/high_nuisance), so a scenario-keyed
intake guard would NOT fully fix it; (3) the verdict's lean toward "(A) scope-and-freeze, primary is
freeze-ready" was **overconfident** — the 24-cluster CP intervals are wide (power CP-low 0.88, not ≈1.00;
clean-control CP-up 0.117–0.342), and freezing a certificate with a breached fail-closed guard repeats the
project's recurring silent-safety-gap failure mode. Claims 1 (concept power 1.00 all scenarios) and 3
(pure_conditional 0.00 under high_subject_tau) were verified exactly.

## Concern 1 — controls creep under heavy stress (the freeze blocker to weigh)

Worst control false-confirm cells (24 clusters; `0/24` CP-up = 0.117):

| scenario | kind | m | count | CP-upper |
|---|---|---|---|---|
| few_epochs | random_label | 30 | 4/24 | 0.342 |
| label_noise | paired_label | 20 & 30 | 3/24 | 0.292 |
| label_noise | paired_covariate_plus_label | 20 & 30 | 3/24 | 0.292 |
| label_noise | unequal_epochs_extreme | 30 | 3/24 | 0.292 |
| high_subject_tau | random_label | 30 | 3/24 | 0.292 |
| few_epochs | random_label | 20 | 3/24 | 0.292 |

At **baseline / high_subject_tau / high_nuisance / imbalanced** controls are clean (≤2/24, mostly 0). The
creep is confined to **`label_noise` (10% symmetric flips)** and **`few_epochs` (6–12 epochs)** — i.e. the
test is **not finite-sample-conservative under heavy label noise / very short records**. Per the
pre-registered rule ("if label/random controls rise materially, do not freeze — revisit null
calibration"), this is the item to resolve or scope around before a freeze.

NB the P2.3 **canary** flagged `unequal_epochs_extreme` at 0.25 (4 clusters) — at 24 clusters it is
0.04–0.08 at baseline (small-sample noise), confirming the value of the larger map.

## Concern 2 — `pure_conditional` is fragile

`pure_conditional` (invisible relabel, no mean shift) is recovered well only in low-noise, low-subject-
variance regimes (0.75 @ m30 baseline/high_nuisance) and **fails entirely (0.00) under high_subject_tau**
(all 24 → NO_CONCEPT_EVIDENCE — it abstains, does not false-confirm). The within-subject pairing cannot
rescue this subtle signal when the subject random effect is large. → `pure_conditional` is at most a
**development secondary**, valid only in the low-noise envelope; it must NOT anchor a confirmatory headline.

## What is solid

The **primary claim is robust**: paired minimal labels (m=20) move `UNIDENTIFIABLE → CONCEPT_CONFIRMED`
for mean-shift / boundary concept change (`paired_concept`, `paired_concept_plus_cov`) at power 1.00
(CP-low 0.88) across **every** stress scenario — high subject variance, high nuisance, class imbalance,
label noise, few epochs — while the matched controls at baseline/moderate stress show no false
confirmations and the machinery never silently fails (0 invalid bootstraps, 0 coverage failures).

## Verdict & recommendation

`pc_centered` is a genuine, development-validated paired concept-shift certificate **for the mean-shift /
boundary case**, robust to a wide difficulty grid. It is **not yet** uniformly clean: (a) controls creep
to 0.12–0.17 under heavy label-noise / very-few-epochs, and (b) `pure_conditional` is fragile.

Two honest paths (reviewer's call), still NO freeze/confirmatory/real-EEG until decided:

- **(A) Scope-and-freeze.** Freeze the **concept/concept_plus_cov** claim @ m=20 with a pre-registered
  operating envelope + record-length/label-quality intake guard; `pure_conditional` development-secondary.
  **Weakness (per red-team):** the creep is kind-structured (random_label / unequal_epochs_extreme leak
  even at baseline), so a regime-only envelope is incomplete, AND the `missing_pair` fail-closed breach
  would be frozen unaddressed.
- **(B) P2.4 calibration round (now the better-supported path).** Before any freeze: (i) **fix the
  `missing_pair` pair-audit breach** (a CONCEPT_CONFIRMED must not slip when pairing is degraded); (ii)
  tighten the null so the **kind-structured** control creep (random_label, unequal_epochs_extreme,
  paired_label) stays conservative — e.g. an effective-sample / pairing-integrity guard or a noise-robust
  null; then re-map and, if clean, freeze the broader claim.

My (revised) lean: **(B)**. The independent red-team showed the earlier "(A) freeze-ready" lean was
overconfident — it rested on point estimates while the 24-cluster CP intervals are wide, treated a
**breached fail-closed guard** (`missing_pair` 2/24) as a PASS, and proposed an envelope that does not
cover the kind-structured component. The concept signal is genuinely strong, but freezing a *certificate*
whose advertised safety guard demonstrably leaks under stress would repeat the project's recurring
silent-safety-gap failure mode. Calibrate + fix the pair-audit breach first, then freeze.
