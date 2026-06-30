# CSC Route B3-P2.2 R1c — the coding fix WORKS, and the winner is `pc_centered` (not full-Z)

DEVELOPMENT, simulator-only, `csc/mininfo/`. Does NOT touch the frozen A tag `csc-confirmatory-v1`. NO
freeze, NO confirmatory, NO real EEG. Single pre-declared change this round (no best-of-K): **condition
coding 0/1 → symmetric ±0.5 ("centered")**, applied to BOTH h0 and h1; `min_confirm_pairs=20` kept; no
cross-fit-T, no RFF, no score-space, no C sweep. R1c primary = `full_z` + centered; the SAME-data
baseline = `pc` + centered (isolates the basis given identical coding).

## Result — 24-cluster development map (`csc/results/b3_p22_r1c_dev_map.json`, SLURM)

CONFIRM rate (guarded, m≥20 only):

| kind | truth | full_z_centered m20 / m30 | **pc_centered m20 / m30** |
|---|---|---|---|
| clean | NO_CONCEPT | 0.00 / 0.00 | 0.00 / 0.00 |
| paired_covariate | NO_CONCEPT | 0.00 / 0.00 | 0.00 / 0.00 |
| paired_label | NO_CONCEPT | 0.12 / 0.08 | 0.00 / 0.08 |
| random_label | NO_CONCEPT | 0.08 / 0.04 | 0.00 / 0.04 |
| paired_concept | CONCEPT | 1.00 / 1.00 | 1.00 / 1.00 |
| paired_pure_conditional | CONCEPT | 0.12 / 0.17 | **0.33 / 0.75** |
| paired_concept_plus_cov | CONCEPT | 1.00 / 1.00 | 1.00 / 1.00 |

## Verdict

1. **The diagnosed coding fix is VALIDATED at scale.** Centered ±0.5 coding collapses the R1 (0/1)
   type-I catastrophe: full_z clean **0.96 → 0.00**, covariate **1.00 → 0.00**. The mechanism (0/1 coding
   gave condition B free asymmetric Z-capacity) is confirmed — symmetric coding removes it.
2. **R1c's full-Z BASIS is NOT promoted.** It does not beat the centered-pc baseline: a mild control
   residual (`paired_label` 0.12, `random_label` 0.08 @ m20) and weak `pure_conditional` (0.12/0.17). The
   *coding*, not the *basis*, was the lever.
3. **The winner is `pc_centered`** (the controlled low-rank basis under the new coding): type-I controlled
   in development (clean/covariate 0.00; label/random ≤0.08 = ≤2/24, noise), `concept` &
   `concept_plus_cov` power 1.00, and — the headline — **`pure_conditional` 0.00 → 0.33 (m20) → 0.75
   (m30)**. The previously Z-only-*undetectable* invisible-relabel case is now **detectable with paired
   labels**, on the type-I-controlled basis. This is the B3 thesis realised for the *hardest* shift.

## Why centered coding unlocks `pure_conditional` (principled, not a fluke)

`pure_conditional` is a condition-**antisymmetric** boundary change (the boundary moves in B relative to
A, with no class-conditional mean shift). Symmetric ±0.5 coding makes `condition × Z` an antisymmetric
basis whose coefficient is exactly 0 under H0 and directly represents an antisymmetric boundary move —
so it has power for `pure_conditional`. The old 0/1 coding conflated that change with the condition main
effect (interaction supported on condition-1 rows only), giving 0 power. The same asymmetry that *broke
calibration* for full_z also *killed pure_conditional power* for pc; fixing the coding cures both.

## Caveats (honest)

- 24 clusters → CP bounds still moderate (`0/24` is CP-UB ≈ 0.117; `2/24` higher). "Type-I controlled"
  remains **development-level**, NOT finite-sample-proven — that is the freeze→unseen confirmatory's job.
- `pure_conditional` needs the larger budget (0.33 @ m20 vs **0.75 @ m30**); mean-shift `concept` is
  saturated already at m20. So the label-efficiency story differs by shift type.
- `pc_centered` label/random sit at ≤0.08 (≤2/24) — watch these in the confirmatory.

## Standing state & recommendation

`certify_paired` default is already **`h1_basis="pc", condition_coding="centered"` = `pc_centered`** — the
winner. `full_z` retained only as the documented experiment (R1 0/1 = type-I catastrophe; R1c centered =
fixes calibration but basis not promoted). `min_confirm_pairs=20` kept. Tests `csc/tests/test_b3.py` lock
the coding fix.

**Recommendation:** adopt `pc_centered` as the B3 method. It now delivers the full B3 claim — minimal
paired labels move `UNIDENTIFIABLE → CONCEPT_CONFIRMED` for BOTH mean-shift concept (m20) AND invisible
pure-conditional relabel (m30), with development type-I control, where Z-only must abstain. Next
(needs authorization): pre-register endpoints (primary `paired_concept`/`concept_plus_cov`; `pure_conditional`
now a *covered secondary* with m=30 budget) → NEW freeze tag (e.g. `csc-b3-confirmatory-v1`) → fail-closed
audit → unseen-cluster confirmatory. Real EEG (PD ON/OFF) only after that passes. Still NO B
freeze/confirmatory, NO real EEG until authorized.
