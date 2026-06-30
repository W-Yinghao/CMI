# CSC Route B3-P2.4c — studentized subject-consistency gate: IMPROVEMENT over P2.4b, NOT a clean pass

DEVELOPMENT diagnostic, simulator-only. Method LOCKED (P2.4b fixed-margin null + all guards); the ONLY
change is the confirmation rule: CONCEPT_CONFIRMED now requires fixed-margin mean-T p≤α AND studentized
subject-consistency p≤α AND 95% LCB(Δ_s)>0. Same grid (controls 48/cell, power 24/cell, 6 scenarios,
m{0,20,30}). Artifact `csc/results/b3_p24c.json` (SLURM 877548). Red-team-verified; presented as evidence.
**No finite-sample claim; NO freeze/confirmatory/real-EEG.**

> **Two overconfidence corrections the red-team forced (and my own recompute confirms).** An earlier draft
> said "ALL control kinds ≤ α on point" and "the gate removes noise, not concept." Both are misleading:
> (1) that's true only under **m≥20 pooling** — at the **m=30 budget alone `clean` = 15/288 = 0.052 > α**;
> (2) the gate removed **10 PRIMARY concept** confirmations, not only secondary. Corrected below.

## Control false-confirm — POOLED (m≥20) AND per-budget (the honest stratified view)

| control kind | pooled m≥20 | m=20 | m=30 | CP-up (m30) |
|---|---|---|---|---|
| `missing_pair` | 0.000 | 0.000 | 0.000 | 0.010 |
| `unequal_epochs_extreme` | 0.000 | 0.000 | 0.000 | 0.010 |
| `paired_covariate` | 0.009 | 0.0035 | 0.014 | 0.032 |
| `paired_covariate_plus_label` | 0.023 | 0.021 | 0.024 | 0.045 |
| `clean` | 0.028 | 0.0035 | **0.052 > α** | 0.079 |
| `paired_label` | 0.031 | 0.035 | 0.028 | 0.050 |
| `random_label` | 0.047 | 0.049 | 0.045 | 0.071 |
| pooled | 0.0196 | — | — | (CP-up 0.024) |

Two residual >α signals: **`clean` @ m=30 point 0.052** (spread across baseline/high_nuisance/high_subject_tau/
imbalanced, not one cell), and **`random_label` CP-up 0.064 (pooled) / 0.071–0.075 (per budget)** — its
point (0.045–0.049) is itself only a hair under α. NO hard-fail flags fired (worst cell 4/48 < 6).

## Power — and the PRIMARY-power cost (corrected)

| positive | m30 NEW | m30 OLD | note |
|---|---|---|---|
| `paired_concept` | 0.889 | 0.889 | m30 unchanged, BUT m20 label_noise 24→20 |
| `paired_concept_plus_cov` | **0.875** | 0.889 | gate removed primary confirms |
| `paired_pure_conditional` | 0.368 | 0.451 | secondary; larger loss |

**Gate removals (old-confirm → new-not-confirm, m≥20): 27 controls (good) + 36 positives.** The 36 positives
are NOT all secondary: **`paired_concept` 4 + `paired_concept_plus_cov` 6 = 10 PRIMARY** (all `label_noise`,
killed by studentized_p 0.055–0.099 — genuine borderline-power losses), `paired_pure_conditional` 26. So
**primary concept power is NOT fully retained** (concept @ m20 1.00 in 4/6 scenarios; `label_noise` 20/24).

## What is solid

vs P2.4b, the studentized gate **reduced every control kind** (pooled 0.026→0.020; `random_label`
0.066→0.047; `paired_label` CP-up 0.054→0.046), with **0 sampler failures**, guards permanent (0/576), and
concept @ m20 still 1.00 in baseline/high_nuisance/high_subject_tau/imbalanced. The gate's removals are
interpretable (`SUBJECT_CONSISTENCY_GATE_NOT_MET` 38 / `STUDENTIZED_FIXED_MARGIN_NULL_NOT_SIG` 25).

## My read (the reviewer judges) — IMPROVEMENT, not a freeze-candidate

P2.4c is a **real improvement over P2.4b** — the right direction (a finite-sample consistency requirement
suppresses noise-label over-confirmation). But it is **not a clean pass**:
1. **`clean` @ m=30 point 0.052 > α** (per-budget) — the cleanest, shift-free control exceeds α at the
   higher decision budget; the m≥20 pooled view (0.028) masks it.
2. **`random_label` CP-up still > α** (0.064), point borderline (~0.047).
3. **The gate cost ~10 PRIMARY concept confirmations** (label_noise) → primary power not fully retained.

So **not a freeze-candidate**. The residual is now small and at the **m=30 / label_noise** edge, and the
gate trades a little primary power for control tightening. Options for the reviewer (no freeze yet):
- accept that the certifier is **conservative-on-average but not per-budget-clean at m=30**, and decide
  whether the residual (`clean`@m30 0.052, `random_label` CP-up 0.064) is within tolerance for a
  freeze-with-disclosure, **or**
- a small P2.4d targeting the m=30 / borderline-power edge (e.g. a finite-sample df correction to the
  studentized statistic, or a Bonferroni-across-budget α) — **not** another null/gate family — then
  re-run this control-resolution. Still NO freeze/confirmatory/real-EEG.
