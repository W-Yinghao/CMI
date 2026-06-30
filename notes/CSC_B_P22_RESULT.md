# CSC Route B3-P2.2 — R1 (full-Z interaction) is a PRE-DECLARED NEGATIVE

DEVELOPMENT, simulator-only, `csc/mininfo/`. Does NOT touch the frozen A tag `csc-confirmatory-v1`. NO
freeze, NO confirmatory, NO real EEG. This round tested ONE pre-declared `h1` replacement (per the
reviewer's "one pre-declared replacement, no best-of-K" rule) and records its outcome honestly.

## What was tested (single pre-declared change)

P2.1's diagnosis: the low-rank **variance-PC** `h1` (`condition × Z_pc(r)`) misses `pure_conditional`
because the invisible relabel lives on a **low-variance** discriminative direction. P2.2's pre-declared
fix (**R1**): replace it with `condition × FULL Z_std` (keeps all directions) under a fixed strong L2
`C_full = 0.5·3/d = 0.125` (the rank-3 `C=0.5` interaction budget scaled to full-Z; NO sweep). Plus a
`min_confirm_pairs = 20` guard (no `CONCEPT_CONFIRMED` below a 20-subject audit). Everything else
(target-internal reference, subject-condition weights, weighted standardise, parametric-bootstrap null,
pair-validity gate, null invalid-accounting, per-cluster logging) unchanged from P2.1.

Pre-registered reject criteria (reviewer): R1 is rejected if controls at m≥20 jump materially, or
`pure_conditional` stays ≈0.

## Result — R1 REJECTED on BOTH criteria

Raw would-confirm (p≤0.05) at m=20, 16 fresh seeds (focused probe):

| kind | truth | **full_z (R1)** | pc (P2.1) |
|---|---|---|---|
| clean | NO_CONCEPT | **0.94** | 0.00 |
| paired_covariate | NO_CONCEPT | **1.00** | 0.00 |
| paired_label | NO_CONCEPT | 0.31 | 0.12 |
| paired_concept | CONCEPT | 1.00 | 1.00 |
| paired_pure_conditional | CONCEPT | **0.12** | 0.00 |

- **Type-I CATASTROPHE** — full_z rejects NULL data at 0.94 (clean) and 1.00 (covariate); a calibrated
  test must reject NULL at ≈α=0.05. (Criterion 1: controls jump materially → fail.)
- **`pure_conditional` NOT recovered** — 0.12, essentially still ≈0; the full-Z basis does not buy the
  invisible-relabel power the diagnosis hoped for. (Criterion 2 → fail.)

Full 24-cluster m-ladder map (`csc/results/b3_p22_dev_map.json`; full_z primary-under-test + pc baseline
on the SAME data). CONFIRM = guarded (m≥20 only); for full_z the RAW would-confirm at m=5/10 is also
shown to expose the label-efficiency behaviour:

| kind | truth | full_z confirm m20 / m30 | full_z raw m5 / m10 | **pc confirm m20 / m30** |
|---|---|---|---|---|
| clean | NO_CONCEPT | **0.96 / 0.88** | 0.92 / 0.96 | 0.04 / 0.00 |
| paired_covariate | NO_CONCEPT | **1.00 / 1.00** | 0.96 / 1.00 | 0.04 / 0.12 |
| paired_label | NO_CONCEPT | 0.25 / 0.21 | 0.58 / 0.58 | 0.08 / 0.00 |
| random_label | NO_CONCEPT | **0.04 / 0.04** | 0.08 / 0.04 | 0.00 / 0.04 |
| paired_concept | CONCEPT | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| paired_pure_conditional | CONCEPT | **0.12 / 0.04** | 0.50 / 0.38 | 0.00 / 0.00 |
| paired_concept_plus_cov | CONCEPT | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

Two further confirmations of the mechanism: (a) **full_z false-confirms only on SEPARABLE null data**
(clean 0.96, covariate 1.00) but NOT on `random_label` (0.04) — the asymmetric extra capacity only
inflates `T` when there is a real boundary to fit *sharper*, exactly as hypothesised. (b) full_z's
`pure_conditional` raw rate *falls* with more labels (0.50→0.04 from m5→m30) — its small early "power"
was the same overfitting artifact washing out, not genuine signal. The **pc baseline at 24 clusters is
tighter than the 12-cluster P2.1 read** (clean 0.04→0.00, covariate 0.04 @ m20).

## Why R1 fails (hypothesis, to be tested only in a future authorized fork — NOT salvaged here)

The condition is coded **0/1**, so `condition × Z` adds Z-capacity to **condition B only** — an
**asymmetric** interaction that lets h1 fit condition B's (shared) boundary *sharper* than condition A
even with NO condition-dependent change. Combined with the strong `C=0.125` that **shrinks h0**, the
parametric-bootstrap labels `Y*~h0` are *less separable* than the real labels, so h1 recovers more on
the real data than under the null → `T > T*` → spurious rejection. The P2.1 `pc` basis avoids this (only
3 interaction directions + weaker `C=0.5` → h0 less shrunk → calibrated). This points at two candidate
levers for a NEW fork (reviewer's call): **centered/symmetric condition coding** (`±1` so the interaction
is antisymmetric and 0 is optimal under H0) and/or a **cross-fitted `T`** (so in-sample overfitting does
not enter). NOT implemented this round — recorded as a single pre-declared negative.

## Standing state after P2.2

- The **`pc` (P2.1 low-rank) basis is the shipping default** again (`certify_paired(h1_basis="pc")`):
  type-I-controlled at m≥20 in development, strong on mean-shift concept (1.00), but **blind to
  `pure_conditional`** (0.00).
- `full_z` is RETAINED in code only as the documented failed experiment (clearly flagged).
- `min_confirm_pairs=20` guard kept (sound regardless of basis; kills small-m instability).
- Tests `csc/tests/test_b3.py` 8/8 (sanity on the controlled pc basis + both-basis epoch-invariance +
  guard + full_z feature-count).

## Next (needs authorization) — a NEW fork, not an in-round retry

Neither basis is satisfactory (`pc` controlled-but-blind to pure_conditional; `full_z` powerful-but-
uncalibrated). The reviewer chooses the next single pre-declared fork, e.g.: (i) symmetric-coding +
cross-fit `T` on the full/low-rank interaction; (ii) R2 (RFF interaction); (iii) R3 (score-space
residual test); or (iv) accept `pc` and scope B3's claim to mean-shift concept only (pure_conditional
declared out-of-scope / secondary). Still NO B freeze/confirmatory, NO real EEG.
