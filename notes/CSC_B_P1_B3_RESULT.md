# CSC Route B3 — B-P1 result (paired minimal-information certificate; the robust path)

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

## Development map — CONCEPT_CONFIRMED rate (12 clusters/cell)

| kind | truth | m=0 | m=5 | m=10 | m=20 |
|---|---|---|---|---|---|
| clean | NO_CONCEPT | 0.00 | 0.00 | 0.17 | 0.00 |
| paired_covariate | NO_CONCEPT | 0.00 | 0.00 | 0.17 | 0.17 |
| paired_label | NO_CONCEPT | 0.00 | 0.00 | 0.00 | 0.00 |
| random_label | NO_CONCEPT | 0.00 | 0.00 | 0.17 | 0.00 |
| **paired_concept** | CONCEPT | 0.00 | 0.83 | 0.92 | **1.00** |
| paired_pure_conditional | CONCEPT | 0.00 | 0.08 | 0.08 | 0.50 |
| **paired_concept_plus_cov** | CONCEPT | 0.00 | 0.75 | 0.92 | **1.00** |

## Kill-criteria assessment (all the hard ones PASS)

- **m=0 reproduces the impossibility boundary** — 0.00 CONCEPT_CONFIRMED for *every* kind (no labels →
  the certifier abstains). ✓
- **Controls do not false-confirm** — clean / paired_covariate / paired_label / random_label all ≤ 0.17
  (mostly 0.00). The **decisive win over B2**: the same covariate confound that B2 false-confirmed at
  **0.75** is **0.17** here, because the within-subject reference + condition intercept absorb it. ✓
- **Power monotone & strong on genuine concept** — paired_concept 0.83→0.92→1.00; concept_plus_cov
  0.75→0.92→1.00 (covariate does not break it). ✓
- **Paired benefit over unpaired B2** — large on the type-I side (covariate 0.17 vs 0.75) and clean
  (≤0.17 vs 0.62). ✓
- **Invalid pair structure fails closed** — all-unpaired target → `INVALID_PAIR_STRUCTURE`. ✓

Sanity tests (`csc/tests/test_b3.py`, standalone, not in the audited A TEST_MODULES): 4/4 — m=0 abstains,
invalid-pair, concept-confirmed (4/4), covariate not-confirmed (0/4).

## Limitations (honest)

- **`paired_pure_conditional` power is modest** (0.08→0.50): the invisible relabeling moves no
  class-conditional *mean*, so a low-rank boundary-perturbation `h1` has little to grab; rises with m but
  far below the boundary-move case. A richer `h1` (kernel / higher-rank) or more labels would be the next
  lever. This is the subtlest positive and the honest soft spot.
- 12 clusters/cell → CP bounds are loose (a `0/12` control is only CP-UB ≈ 0.221); this is a DEVELOPMENT
  map, **not** error control. A confirmatory claim needs the freeze→unseen cycle.
- Random-label control at small m is noisy (0.00–0.38 in earlier probes); stabilises to 0.00 by m=20.

## Status & next

B3 is the **robust primary** Route-B mechanism: minimal paired labels move `UNIDENTIFIABLE →
CONCEPT_CONFIRMED` on genuine concept change while controlling the covariate/label/clean false-confirm
that sank the source-reference B2. B2 (`few_label_test.py`) stays a **diagnostic/unpaired fallback**.

Next (NOT done; needs authorization): harden the B3 protocol (richer `h1` for the pure-conditional tail;
pre-registered decide_n / m-ladder), then a **new** freeze (new tag, e.g. `csc-b3-confirmatory-v1`) →
fail-closed audit → unseen-cluster confirmatory — its own cycle, never reusing `csc-confirmatory-v1`.
Real EEG (PD ON/OFF) only after a passing B3 synthetic confirmatory.
