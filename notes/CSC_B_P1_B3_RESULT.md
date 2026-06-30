# CSC Route B3 — paired minimal-information certificate (B-P1 + B3-P2.1 hardened)

DEVELOPMENT, simulator-only, new namespace `csc/mininfo/`. Does NOT touch the frozen A tag
`csc-confirmatory-v1`. NO freeze, NO confirmatory, NO real EEG. Per-cluster logging from day one
(artifact `csc/results/b3_dev_map.json`).

## Mechanism

**Paired within-subject (ON/OFF) conditional-change test** — target-INTERNAL reference (the subject's
own other condition), so NO source posterior is used. Tests whether `P(Y|Z)` depends on condition
beyond a condition intercept (which absorbs a condition covariate offset AND a condition class-prior):
`h0: [Z, condition]` vs `h1: [Z, condition, condition×Z_pc(r)]` (low-rank); `T = vote(NLL_h0) −
vote(NLL_h1)`; parametric-bootstrap null `Y*~h0`; subject-condition vote. Certifier states:
`CONCEPT_CONFIRMED / NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT / NEED_MORE_LABELS / INVALID_PAIR_STRUCTURE /
UNIDENTIFIABLE` (no `COVARIATE_COMPATIBLE`).

## B3-P2.1 inference-contract hardening (done)

- **Subject-condition-WEIGHTED fits + standardisation + PCs** (raw weights `1/(|U_s|·n_su)`, Σ=#subjects)
  → the observed `T` is **exactly invariant to epoch duplication** (`dT = 1.1e-16`; test_b3 #5).
- **Per-condition class coverage fails closed** — a condition with <2 classes → `INVALID` (test_b3 #6).
- **Null invalid-accounting** — a bootstrap replicate is valid only if it preserves the audit's
  pair-validity; degenerate/fit-failed replicates are charged extreme (conservative); >20% invalid →
  test INVALID (fail closed).
- **Explicit label-budget accounting** logged: `n_queried_subjects`, `n_labeled_subject_conditions`,
  `n_labeled_epochs`, `n_pairs`, `classes_by_condition`, `n_boot_invalid`, `false_confirm_flag`,
  `power_flag` per cluster.

## Development map — CONCEPT_CONFIRMED rate (12 clusters/cell, B3-P2.1 hardened)

| kind | truth | m=0 | m=5 | m=10 | m=20 |
|---|---|---|---|---|---|
| clean | NO_CONCEPT | 0.00 | 0.33 | 0.17 | 0.08 |
| paired_covariate | NO_CONCEPT | 0.00 | 0.08 | 0.25 | 0.08 |
| paired_label | NO_CONCEPT | 0.00 | 0.00 | 0.08 | 0.08 |
| random_label | NO_CONCEPT | 0.00 | 0.00 | 0.17 | 0.00 |
| **paired_concept** | CONCEPT | 0.00 | **1.00** | **1.00** | **1.00** |
| paired_pure_conditional | CONCEPT | 0.00 | 0.08 | 0.00 | 0.00 |
| **paired_concept_plus_cov** | CONCEPT | 0.00 | **1.00** | **1.00** | **1.00** |

Hardening vs the unhardened B-P1 map: **power UP on mean-shift concept** (paired_concept / concept_plus_cov
now 1.00 from m=5, was 0.83 / 0.75); **two honest regressions** surfaced — pure_conditional power *lost*
and small-m controls *noisier* — see Limitations.

## Kill-criteria assessment (DEVELOPMENT evidence — NOT error control)

> **Wording discipline (post-A lesson):** 12 clusters/cell cannot establish finite-sample control — a
> `0/12` control still has CP-UB ≈ 0.221. So this section reads as *development behaviour*, NOT a
> type-I-control claim. Finite-sample control remains **unestablished** until a B3 freeze→unseen
> confirmatory.

- **m=0 reproduces the impossibility boundary** — 0.00 CONCEPT_CONFIRMED for *every* kind (no labels →
  the certifier abstains). ✓
- **Strong power on the mean-shift concept** — paired_concept & concept_plus_cov confirm at **1.00 from
  m=5** (covariate does not break it). [development]
- **At the robust operating point m=20, controls confirm at ≤ 0.08** (1/12) — substantially below B2's
  covariate 0.75, but at 12 clusters this is **development behaviour, not finite-sample type-I control**.
- **Invalid pair structure fails closed** — all-unpaired target → `INVALID_PAIR_STRUCTURE`. ✓

Sanity + contract tests (`csc/tests/test_b3.py`, standalone, not in the audited A TEST_MODULES): **6/6** —
m=0 abstains, invalid-pair, concept-confirmed (4/4), covariate not-confirmed (0/4), epoch-duplication
invariance (dT=1.1e-16), per-condition class coverage fail-closed.

## Limitations / honest findings from the hardened run

- **`paired_pure_conditional` power LOST (0.00) under the variance-PC basis** — hardening dropped it from
  0.50 (unhardened) to 0.00. Root cause: the invisible relabeling moves no class-conditional *mean*, so
  its discriminative direction is a **low-variance** direction; the top-r *variance* PCs in `h1`'s
  `condition×Z_pc` simply do not span it. This is not a regularisation artifact — it shows the variance-PC
  inductive bias is **wrong for pure-conditional shift**. → Direct motivation for **P2.2 (richer h1:
  RFF / discriminative-direction interaction)**, run jointly with controls.
- **Small-m controls are noisy** — clean 0.33 @ m=5, paired_covariate 0.25 @ m=10 (3/12); they settle to
  ≤0.08 @ m=20. With only ~5 paired subjects the subject-vote `T` and its bootstrap are high-variance.
  → The decision-relevant regime is **m≥20**; a `min_confirm_pairs` guard (no CONCEPT_CONFIRMED below a
  minimum audit size) is a candidate P2 contract addition.
- 12 clusters/cell → CP bounds loose (`0/12` is CP-UB ≈ 0.221); DEVELOPMENT map, **not** error control.

## Status & next

B3 is the **better Route-B primary candidate**: minimal paired labels move `UNIDENTIFIABLE →
CONCEPT_CONFIRMED` on genuine concept change while **substantially reducing (in development)** the
covariate/label/clean false-confirmation that sank the source-reference B2 — finite-sample control is
**not** claimed (that is the freeze→unseen confirmatory's job). B2 (`few_label_test.py`) stays a
**diagnostic/unpaired fallback**.

Progress: **B3-P2.1 inference-contract hardening DONE** (weighted fits/standardise/PCs → exact
epoch-invariance; per-condition coverage; null invalid-accounting; budget logging; 6/6 tests). It
*raised* power on mean-shift concept but exposed the two findings above.

Next (NOT done; needs authorization):
- **P2.2 — richer `h1`** (the priority finding): RFF / discriminative-direction `condition×Z`
  interaction to recover pure_conditional power, run **jointly with all controls** (no best-of-K; one
  pre-declared replacement), plus a candidate `min_confirm_pairs` guard for small-m noise.
- **P2.3 — expanded development map** (more m, more controls incl. covariate+label & missing-pair, more
  axes), still development, still no error-control claim.
- only then a **new** freeze (new tag, e.g. `csc-b3-confirmatory-v1`) → fail-closed audit →
  unseen-cluster confirmatory — its own cycle, never reusing `csc-confirmatory-v1`.
- Real EEG (PD ON/OFF) only after a passing B3 synthetic confirmatory.
