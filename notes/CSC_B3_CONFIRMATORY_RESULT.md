# CSC Route B3 — CONFIRMATORY RESULT (unseen synthetic block) — PASS (C1–C6)

**Scope of this result.** ONE pre-registered, frozen, unseen-cluster synthetic confirmatory of the B3 paired
within-subject concept-shift certificate. This is a **development-to-confirmatory success on synthetic data
only.** It is **not** a real-EEG claim; real EEG / PD ON–OFF remains blocked pending separate authorization.

## Provenance (infra-clean)

| field | value |
|---|---|
| frozen tag | `csc-b3-confirmatory-v1` |
| tag / HEAD commit | `0595f64cde0857c53fff2f9fc36347d663a65331` |
| tree clean | `True` |
| manifest hash (canonical sha256 over frozen_payload) | `a96dc0de55917c0f6dadf39807ea3cdd7539df1802f57917549ae1ddbb65cf62` |
| base_seed | `3000000` (cell_stride 100000, target_offset 10000) |
| n_clusters | 5376 (112 cells × 48 replicates) |
| seed range / disjoint | [3000000, 14100047]; disjoint from A src 900000–65, A tgt 1800000–65, B dev, smoke <100000 |
| SLURM job / node | 878107 / cpu-high nodecpu02 |
| artifact | `csc/results/b3_confirmatory_result.json` (1.67 MB) |
| artifact sha256 | in `csc/results/b3_confirmatory_result.json.sha256` (re-verified OK after copy) |
| runner exit | rc=0 (scientific verdict inside JSON) |
| A tag `csc-confirmatory-v1` / `dee8958` | UNTOUCHED |

The frozen manifest was NOT modified after freeze; the runner re-verified manifest hash + all 3 code hashes +
HEAD==tag + clean tree + seed disjointness before writing; the sbatch wrapper re-verified freshness before
`mv` to the final artifact.

## Verdict

- **preliminary_scientific_verdict_excluding_C6 = PASS** (runner artifact).
- **C6 independent red-team (separate agent, own scipy Clopper-Pearson) CONFIRMS without correction** —
  every gating number reproduced to machine precision (CP max |Δ| = 2.8e-17); no discrepancy, denominator
  inflation, hidden failure, or overclaim found. See `CSC_B3_CONFIRMATORY_C6_REDTEAM.md`.
- **FINAL C6-inclusive verdict = PASS (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6).**

| criterion | result | headline number |
|---|---|---|
| C1 guards (no false confirm) | PASS | missing_pair 0/576, unequal_epochs_extreme 0/576 (both 100% INVALID_PAIR_STRUCTURE) |
| C2 control type-I (kind×budget, n=288, CP-up ≤ .05) | PASS | worst `clean`\|m30 = 3/288, CP-up **0.0267** |
| C3 no hot cell / no kind-leak | PASS | max control cell **2/48**; 0 cells ≥6; 0 cells ≥3 |
| C4 primary power (kind×budget, n=192, CP-lo ≥ .60; per-cell ≥ .50) | PASS | all four = **1.000**, CP-lo **0.9845** |
| C5 no silent failure | PASS | sampler_failures 0, boot_invalid 0, 0 out-of-set states |
| C6 red-team reproduces w/o correction | PASS | independent re-aggregation matches exactly |

## C2 — control false-confirmation, ALL 14 cells disclosed (n=288 each, pooled over all 6 scenarios)

Both guard kinds are included in the 7 control kinds. Denominators conservative: NEED_MORE_LABELS and
INVALID_PAIR_STRUCTURE are retained (not dropped). **Stratified by budget — not pooled across m20+m30.**

| control kind | m20 | CP-up | m30 | CP-up |
|---|---|---|---|---|
| clean | 1/288 | 0.0164 | **3/288** | **0.0267** |
| paired_covariate | 0/288 | 0.0103 | 1/288 | 0.0164 |
| paired_label | 0/288 | 0.0103 | 0/288 | 0.0103 |
| random_label | 0/288 | 0.0103 | 0/288 | 0.0103 |
| paired_covariate_plus_label | 1/288 | 0.0164 | 0/288 | 0.0103 |
| missing_pair (guard) | 0/288 | 0.0103 | 0/288 | 0.0103 |
| unequal_epochs_extreme (guard) | 0/288 | 0.0103 | 0/288 | 0.0103 |

All 14 CP-uppers ≤ 0.05. Worst is `clean`@m30 (0.0267), well under α. Total control confirms = **6**.

## C3 — every control cell with ≥1 confirm disclosed (scenario×kind×budget, n=48)

| scenario | kind | budget | count |
|---|---|---|---|
| baseline | clean | m30 | 2/48 |
| baseline | paired_covariate_plus_label | m20 | 1/48 |
| high_nuisance | clean | m20 | 1/48 |
| imbalanced | clean | m30 | 1/48 |
| imbalanced | paired_covariate | m30 | 1/48 |

max cell = 2/48; #cells ≥6 = 0; #cells ≥3 = 0; no kind×budget with ≥2 cells at ≥3. (Sum = 6, reconciles with C2.)

## C4 — primary power, ALL 4 gating cells (n=192 each, pooled 4 strong scenarios)

| kind | budget | power | CP-lo |
|---|---|---|---|
| paired_concept | m20 | 192/192 = 1.000 | 0.9845 |
| paired_concept | m30 | 192/192 = 1.000 | 0.9845 |
| paired_concept_plus_cov | m20 | 192/192 = 1.000 | 0.9845 |
| paired_concept_plus_cov | m30 | 192/192 = 1.000 | 0.9845 |

Every per-scenario cell (n=48) = 1.00, so the ≥0.50 floor holds everywhere.

## Secondary (reported, NON-gating) — paired_pure_conditional (the subtlest, known-weak shift)

| kind | budget | power | CP-lo |
|---|---|---|---|
| paired_pure_conditional | m20 | 42/288 = **0.146** | 0.113 |
| paired_pure_conditional | m30 | 81/288 = **0.281** | 0.238 |

This is weak, as pre-registered — the invisible pure-conditional relabel is a genuine limitation of the method
and is **not** part of the primary claim. Reported here for completeness, not counted toward the verdict.

## State distribution (C5)

`NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT` 2984, `CONCEPT_CONFIRMED` 897, `NEED_MORE_LABELS` 343,
`INVALID_PAIR_STRUCTURE` 1152. `UNIDENTIFIABLE` unused but valid. All within the 5-state set; 0 sampler
failures; 0 boot-invalid. Confirmed accounting: 6 control + 768 primary + 123 secondary = 897 (reconciles).

## Honest limitations (ship with the positive claim)

1. **Synthetic only.** No real-EEG / PD ON–OFF claim; that requires a passing synthetic confirmatory (this)
   AND a separate real-data authorization.
2. **Development-informed operating envelope.** The primary claim is scoped to 4 strong scenarios
   (baseline / high_nuisance / high_subject_tau / imbalanced) × m∈{20,30}. Heavy label-noise (~10% flips)
   and very-short-record (<8 epochs/condition → eligibility abstention) regimes were **pre-registered OUT** of
   the primary power claim, not discovered-after-the-fact. Controls, however, WERE evaluated over all 6
   scenarios incl. those two.
3. **pure_conditional is weak** (0.146 / 0.281) — the subtlest invisible relabel remains hard even with the
   full machinery. Secondary, non-gating, disclosed.
4. C2 gates are **pointwise Clopper-Pearson** at the kind×budget reporting unit; the headline is a conjunction
   of pointwise gates, not a simultaneous familywise CI over all cells.

## What this establishes / what it does not

- **Establishes:** on paired within-subject targets with sound pairing (integrity ≥0.95, ≥8 epochs/condition)
  and ≥20 eligible labelled paired subjects, the `pc_centered_calibrated` certifier confirms genuine
  class-conditional/boundary concept shift at power 1.00 while controlling false confirmation at α — within the
  pre-registered synthetic envelope, verified on unseen clusters and independently red-teamed.
- **Does NOT establish:** any real-EEG result, any Z-only (label-free) confirmation (theory: must abstain),
  or power on the excluded label-noise / short-record regimes.
