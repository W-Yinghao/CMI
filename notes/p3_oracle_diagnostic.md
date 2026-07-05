# CSC-realEEG P3 forensics — the v2 FAIL is a plug-in fitted-h0 NULL problem (diagnostic-only)

```
diagnostic_only = true
not_used_for_certificate = true
oracle_uses_true_DGP = true
not_deployable = true
no_method_change = true
no_B4 = true
```

```
Scope:
  P3.0b/c + P3.0d forensic archive
  development diagnostic only
  no method change · no certifier change · no h1/statistic/feature/montage change
  no B4 canary · no confirmatory tag · no deployment claim
```

**Safe headline:** *P3.0d supports fitted-h0 / plug-in null under-dispersion as the dominant mechanism of the
real-feature B3 NULL_cov false-confirm failure; an oracle-generator fixed-margin null restores safety on the
exposed semi-synthetic bank but is not deployable and has modest POS power.*

**Not confirmatory. Not a method. Not merged with the frozen `csc-realeeg-v2` scientific result. No tag.**
Development forensics that localize WHY the frozen paired B3 certificate FAILed on real Lee2019/OpenBMI
SM16_no_FCz log-bandpower features (see [CSC_REALEEG_V2_RESULT.md](CSC_REALEEG_V2_RESULT.md)).

## Headline (refined conclusion for the v2 FAIL)
> **B3's plug-in fitted-h0 null is anti-conservative under real EEG cross-session covariate structure. The B3
> statistic itself is NOT the dominant failure mode.** When the null is drawn from the true data-generating
> P(Y|Z) instead of the fitted h0, the NULL_cov false-confirmation collapses from **18.7% → 1.3%** (≈ the alpha
> budget), while ~50% of the true-concept power survives. The failure is the NULL GENERATOR, not the statistic,
> and not covariate overlap.

## P3.0 — overlap hypothesis REFUTED
Reconstructed the exact frozen v2 cohorts and measured raw-Z session separability / covariate overlap. The
NULL_cov false-confirms do NOT concentrate in high-session-AUC / low-overlap cohorts (sep-AUC ≤ 0.635, mostly
~0.5; POS_concept at chance), and an overlap/ESS gate is non-specific (catching the false-confirms requires
proportionally destroying true power). On these features session drift is weak (session AUC ~0.53, overlap
~0.99). → **not a raw-Z covariate-support/overlap problem; no overlap gate.**

## P3.0b/c — certifier-internal forensic (1800 cohorts, 3 seed blocks × 6 conditions)
Reconstructed each cohort and captured the certifier's full internal log (T, fixed-margin & standard nulls,
studentized stat/p, subject deltas, null mean/sd). Findings (reproducible across all 3 blocks):
- **False-confirm is robust:** NULL_cov 18/19/20 per 100 (~19%), NULL_cov_plus_label 6/11/15; controls clean.
- **The mean-T bootstrap null is under-dispersed & SATURATED:** `fixed_margin_p` at the 0.005 floor for 100% of
  FC and 91% of clean; the standard (non-fixed-margin) null too. The mean-T gate is non-functional on real Z.
- **The studentized subject-consistency gate is the only live discriminator** and false-confirms ~19%.
- **NOT few-subject dominance** (frac_delta_pos 0.77, top1_share 0.14 — many subjects weakly consistent).
- False-confirms look nearly identical to true concepts on studentized/subject-consistency axes, differing mainly
  in effect size (T ~4× smaller).

## P3.0d — ORACLE-generator fixed-margin diagnostic (1200 cohorts, all fidelity-exact)
The decisive test. For every cohort, the SAME statistic path (query, folds, h0/h1 fits, subject-condition
weighting, T, studentized stat, LCB, invalid accounting) is recomputed with ONLY the null LABEL GENERATOR
swapped: instead of the fitted h0, Y* is drawn from the bank's TRUE generator `p_oracle(Y|Z) =
_pooled_clf(coh_Z, coh_y)` (session-independent), under the SAME observed condition×class margins (fixed-margin).
Strict fidelity: the re-derived observed T equals the in-process method T exactly (dT=0). Oracle uses the true
DGP — it is a DIAGNOSTIC, not deployable.

### Table A — all-cohort calibration (method → oracle)
| condition | n | n_invalid | method confirm | **oracle confirm** |
|---|---|---|---|---|
| NULL_cov | 300 | 0 | 0.187 | **0.013** |
| NULL_cov_plus_label | 300 | 0 | 0.107 | **0.007** |
| POS_concept | 300 | 9 | 0.267 | **0.133** |
| POS_concept_plus_cov | 300 | 7 | 0.250 | **0.113** |

### Table D — decision matrix (method × oracle)
| condition | m+/o+ | m+/o− | m−/o+ | m−/o− |
|---|---|---|---|---|
| NULL_cov | 4 | **52** | 0 | 244 |
| NULL_cov_plus_label | 2 | **30** | 0 | 268 |
| POS_concept | 40 | 40 | 0 | 220 |
| POS_concept_plus_cov | 34 | 41 | 0 | 225 |

### Table B — NULL_cov mechanism
Of 56 method false-confirms, oracle confirms only **4 (7%)**; oracle null_sd **0.00049 > method null_sd 0.0003**
(fitted null too narrow by ~1.6×). No `m−/o+` cells anywhere (oracle creates no hidden positives).

### Table C — POS separation
Oracle keeps ~**50%** of POS_concept true-confirms, ~**45%** of POS_concept_plus_cov. Oracle p's are NOT
floor-saturated (NULL_cov oracle studentized-p median 0.55) → B=1000 resolution not needed.

### P3.0d verdict
```
dominant mechanism  = fitted-h0 / plug-in null under-dispersion (Case 1)
rejected            = raw-Z overlap failure
rejected            = statistic hopeless under the true generator
caveat              = oracle-calibrated power is modest (~13%)
```

## Updated scientific arc
```
A. Z-only route            : non-identifiable (Prop.1) + frozen synthetic FAIL.
B. Paired minimal-label    : synthetic B3 PASS (C1-C6).
C. Real-feature bridge     : plug-in B3 FAIL under Lee2019/OpenBMI covariate structure.
D. P3 forensics            : overlap NOT responsible.
E. Oracle diagnostic       : statistic not hopeless; plug-in fitted-h0 null under-dispersed;
                             oracle null restores safety (1.3% FC) but reveals modest power (~13%).
F. Next open problem       : construct an ESTIMABLE null generator approaching the oracle operating point.
```

## B4-dev — OPENED as "better estimable null" only (no method/h1/feature/statistic change yet)
**Research question:** can an ESTIMABLE null generator approximate the oracle-null operating point WITHOUT using
the true semi-synthetic DGP? Target = recover most of the oracle's calibration while retaining some real (modest)
power. **Not** to beat the oracle.

**Oracle reference point (SM16):** NULL_cov FC ≈ 1.3% · NULL_cov_plus_label ≈ 0.7% · POS_concept ≈ 13.3% ·
POS_concept_plus_cov ≈ 11.3%.

**Candidate nulls (test 3, not a fishing expedition):**
- **B4a — bagged / fold-local h0 null:** draw the null h0 from a subject-cluster BAGGED ensemble (draw at the
  replicate/subject-cluster level, not per-trial), keep margins/folds/weighting/statistic; widen the null to
  reflect nuisance uncertainty. First candidate (most direct).
- **B4b — nested calibration / full-refit null:** each null replicate replays the fold-local h0/h1 estimation
  pathway the observed statistic depends on (statistic unchanged; null includes estimation uncertainty). Canary
  first — runtime may expand.
- **B4c — richer SHARED nuisance h0 (not richer h1):** h0 = f(Z)+c, h1 = f(Z)+c+c×Z_PC, with f(Z) a fixed
  regularized basis (PCs / quadratic PC / low-rank spline-RBF). Enrich the trunk, never the interaction first.
- **B4-cal — variance inflation ONLY as a locked calibration layer:** estimate an sd-floor / studentized
  inflation / empirical p-recalibration on calibration null blocks, LOCK the map, evaluate on FRESH held-out
  seed blocks. Forbidden: choosing the multiplier after seeing the held-out endpoint.

**B4-dev success criterion (n=300):** safety NULL_cov ≤ 7/300 and NULL_cov_plus_label ≤ 7/300 (95% CP upper
< 5%; 8/300 crosses it); advisory power POS_concept ≥ ~10%, POS_cov ≥ ~8%. Integrity: no oracle generator, no
overlap gate, no feature rescue, no pooled-control hiding, no method-invalid denominator padding.

**Protocol:** Stage 0 forensic commit (this package) → Stage 1 B4 canary (40 FC + 40 clean + POS) → Stage 2
exposed dev replay on the 1200 P3 cohorts (DEVELOPMENT-EXPOSED; pick ≤1 candidate) → Stage 3 FRESH seed block
(base 80e6/90e6, all 6 conditions) → only then discuss a frozen B4 confirmatory.

**Not authorized / not done here:** overlap gate, richer h1, feature/montage swap, oracle-as-method, confirmatory
tag, papering over weak oracle power, any freeze.

## Verification (independent red-team, 3 lenses)
**All 3 lenses MATCHES, 0 serious issues.** (1) Re-aggregation: every Table A–D number reproduced exactly from
the raw per-cohort rows (confirm defs recomputed from raw fields, 0/1200 mismatches vs stored flags); `m−/o+ = 0`
in every condition — the oracle only ever REMOVES confirms, never adds. (2) Integrity: **oracle diagnostic =
1200/1200 cohort rows unique; internal forensic array = 1800/1800 cohort rows unique; no duplicate cohort key in
either merged artifact.** All applicable fidelity_dT exactly 0, oracle p ∈ [1/201,1], margins preserved, 16 method-invalid
(POS 9+7) correctly `oracle_applicable=false` + non-confirm, both merged sha256 recomputed and match. Method
`fixed_margin_p` sits at the 1/201 floor for 93% of NULL_cov (8 unique values) while oracle p has 162 unique
values spread [0.005,1.0] — the plug-in null collapses, the oracle null is estimable-spread. (3) Implementation:
the oracle draws Y* from `clf_oracle = _pooled_clf(coh_Z, coh_y)`, **byte-identical to the clf `build_cohort` uses
to generate the labels** (true session-independent `p_oracle(Y|Z)`), fixed-margin, same statistic path, only
`logp0` swapped; no clearing/dropping bug — the 1.3% collapse is honest size, the ~50% POS drop is the
correctly-specified null being stricter than the over-fit fitted-h0 null.

### Known numerical tolerances / non-impacting discrepancies (NOT selective denominator handling)
- **Table B null_sd provenance is cross-artifact by design.** Table B's "method null_sd ≈ 0.0003" is the
  internal-forensic `null_sd_T` (median 0.000284, mean 0.000283) over the same NULL_cov FC cohorts — the
  oracle-only merged file stores only the oracle-side `null_sd_T` (0.00049 median / 0.00050 mean over those
  cohorts). The inequality *oracle null wider than method null* holds under both mean and median.
- **NULL_cov method-FC count 56 (oracle file) vs 57 (internal file)** is a **1-cohort cross-node `lbfgs(tol=1e-4)`
  difference**, not selective denominator handling. Both use the full 300 denominator; the oracle worker
  recomputes the method **in-process** to neutralize cross-node solver noise (within-oracle `fidelity_dT = 0`).
  All-cohort Tables A/D verdicts are robust to ±1.

**Provenance gap CLOSED.** The first-array NULL oracle shards were numerically red-team-verified, **but were
replaced** by re-running with the committed worker so all 12 oracle shards come from ONE git-tracked code version
(uniform schema; identical scientific values modulo the cross-node ±1 above). The replacement is for uniform
provenance, **not** because any number was overturned.

## Package (diagnostic-only)
`csc/results/p3_forensics/`: `p3_internal_forensics_merged.jsonl` (1800), `p3_oracle_diagnostic_merged.jsonl`
(1200), `p3_oracle_tables_A_D.json`, `p3_redteam_report.json`, `SHA256SUMS`, and `scripts/` (the exact dev
workers/mergers/sbatch that produced them — diagnostic tooling, not method code). Post-rerun merged basis:
internal sha256 `ad8f5f7b…`, oracle `83b45c8c…` (all 12 oracle shards from the committed worker). This is a
plain diagnostic-only commit — **no tag**. Per-file sha256 in `SHA256SUMS`.
