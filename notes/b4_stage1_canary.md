# CSC-realEEG B4 Stage 1 canary — bagged/nested h0 null does NOT approximate the oracle (diagnostic-only)

```
Scope: B4 Stage 1 canary | development diagnostic only | NULL-GENERATION only
  observed T / studentized / folds / features / margins / invalid accounting UNCHANGED (fidelity_dT=0)
  no overlap gate | no richer h1 | no feature/montage change | no new statistic
  no oracle generator in any candidate arm | not deployable | no confirmatory tag | no method promotion
```

**Safe headline:** *Uncertainty-aware estimation (subject-cluster bagging / nesting) of the CURRENT fitted-h0
null does NOT approximate the oracle null dispersion; it captures h0 estimation variance (~20–25% of the gap)
but not the h0 misspecification bias the oracle exposes. B4a and B4b both FAIL the Stage-1 hard-safety screen;
neither is eligible for Stage 2.*

This is a development eligibility screen (not confirmatory). Follows [p3_oracle_diagnostic.md](p3_oracle_diagnostic.md):
P3.0d showed the v2 real-feature FAIL is a fitted-h0 / plug-in null under-dispersion problem; the oracle
(true-generator) null restores safety. B4 tests whether an ESTIMABLE null can approach that oracle operating
point WITHOUT the true DGP. Stage 1 = a bagged/nested version of the SAME h0.

## Selection (deterministic, logged)
206 cohorts from the committed P3 oracle strata (`csc/results/p3_forensics/`), sorted by (seed_block, cohort)
within condition×stratum. Manifest sha256 `9541b1d6…`. Arms: **method** (existing fixed-margin full-audit-h0
null), **B4a** (bagged h0, K=50 subject-cluster bootstraps, rotated per replicate), **B4b** (nested h0, K=B=200,
fresh bag per replicate), + a variance-inflation diagnostic baseline. **NULL-ONLY**: observed T computed once and
shared across arms (fidelity_dT=0 on all 206 → observed statistic provably unaltered).

## Hard safety screen → B4a FAIL, B4b FAIL
| stratum (n=20) | method | B4a | B4b | oracle | cap |
|---|---|---|---|---|---|
| NULL_cov m+/o− | 20 | **17** | **16** | 0 | ≤5 |
| NULL_cov_plus_label m+/o− | 20 | **14** | **14** | 0 | ≤5 |
| NULL_cov m−/o− (clean) | 0 | 1 | 1 | 0 | ≤1 |
| NULL_cov_plus_label m−/o− | 0 | 1 | 0 | 0 | ≤1 |

NULL m+/o− removal: B4a 3/20 & 6/20, B4b 4/20 & 6/20 — vs oracle 20/20 & 20/20. Clean false-add at/under cap.

## POS usefulness (m+/o+, n=20) — retained but moot (unsafe)
POS_concept method 20 / B4a 20 / B4b 20 ; POS_concept_plus_cov 20 / 19 / 19.

## Null-dispersion screen (ratio-of-medians, consistent convention)
| | null_sd median | / method | gap closed |
|---|---|---|---|
| method | 0.000408 | 1.00 | — |
| B4a | 0.000461 | 1.13 | **20%** |
| B4b | 0.000475 | 1.16 | **25%** |
| oracle | 0.000677 | 1.66 | 100% |

NULL_cov m+/o− studentized-p median: method 0.012 → B4a/B4b 0.015 → oracle 0.264. `fixed_margin_p` stays at the
0.005 floor for 19–20/20 under B4a — **the collapse is not broken.**

## Stage-1 verdict
**Question:** can uncertainty-aware estimation of the CURRENT h0 null approximate the oracle null dispersion?
**Answer: NO.** Bagging/nesting the fitted logistic h0 widens the null only 13–16% (closes ~20–25% of the
oracle gap). The fitted-h0 ↔ true-generator gap is **bias-dominated** (systematic misspecification of the
logistic h0 under real covariate structure), not estimation variance — so resampling the SAME h0 family cannot
close it. Neither B4a nor B4b is eligible for Stage 2.

## Recommended next branch (reviewer decision tree)
- **NOT** advance to the 1200-cohort replay (both fail); **NOT** hold safe-but-powerless (they are unsafe).
- **Revise → B4c (richer SHARED nuisance h0), as a new logged Stage-1b canary** — the bias-not-variance finding
  points here: a more flexible fixed regularized trunk f(Z) (PC quadratic / low-rank spline-RBF) reduces the
  misspecification the oracle's extra width reflects. Enrich the trunk, never h1. (Held until now; now motivated.)
- Alternative: variance-inflation / p-recalibration under an explicitly authorized **split-null calibration**
  design (target ratio ≈ 1.66×) — but a single multiplier risks overfitting Lee2019 SM16; needs held-out.
- Alternative: retire B4, pivot to an abstention / risk-control router.
**No decision taken here.** Awaiting reviewer authorization of one branch.

## Verification (independent red-team, 2 lenses)
0 serious issues. Re-aggregation: all confirm counts + null_sd medians reproduce EXACTLY (0/206×3 mismatches);
B4a/B4b FAIL confirmed. Null-only invariant: MATCHES — 206/206 fidelity_dT=0 (observed T unaltered), 206 unique
task_ids = manifest, all arm p ∈ [1/201,1], only the h0 log-prob generator differs between arms, no oracle/true-DGP
generator used in B4a/B4b. Lows: ratio-convention cosmetic (now fixed to consistent ratio-of-medians: oracle
1.66×); shared `prep` fold object is null-only and cannot alter the observed scalar (fidelity_dT=0 confirms).
Full: `csc/results/b4_stage1_canary/b4_stage1_redteam_checks.json`.

## Package (diagnostic-only, no tag)
`csc/results/b4_stage1_canary/`: `b4_stage1_manifest.json`, `b4_stage1_results.jsonl` (206), `b4_stage1_tables.json`,
`b4_stage1_redteam_checks.json`, `SHA256SUMS`, `scripts/` (worker/merge/tables/sbatch/watcher + README).
Merged canary sha256 `26685ac9…`.
