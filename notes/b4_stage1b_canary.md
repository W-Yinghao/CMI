# CSC-realEEG B4c Stage-1b canary — richer shared nuisance trunk HARD FAILS (regresses safety); B4 closed

```
Scope: B4c Stage-1b canary | development diagnostic only | RICHER SHARED NUISANCE TRUNK only
  interaction c x PC1:3 UNCHANGED | estimand CE(h0)-CE(h1) unchanged | SM16 unchanged
  no overlap gate | no richer interaction | no feature/montage change | no new statistic
  no oracle generator in fitting or null | not deployable | no confirmatory tag | no method promotion
```

**Safe headline:** *A richer FITTED shared nuisance trunk (B4c-Q3) does NOT approximate the oracle null; it
HARD-FAILS the safety screen (removes 0/20 NULL_cov false-confirms) and REGRESSES safety (adds false-confirms on
clean cohorts). Enriching the fitted family makes the null narrower and more overfit, not less biased. Combined
with Stage 1a (bagged/nested h0 also fail), no estimable fitted-h0-family null approximates the oracle — the B4
"better estimable null" direction is closed.*

Follows [b4_stage1_canary.md](b4_stage1_canary.md) (Stage 1a: bagged/nested h0 FAIL, under-widen ~20-25% of the
gap) and [p3_oracle_diagnostic.md](p3_oracle_diagnostic.md) (the oracle true-generator null restores safety but
is not deployable). Same 206-cohort manifest, same oracle strata.

## B4c-Q3 candidate (predeclared, no tuning)
Shared trunk = `[Z1:16, c, u1^2,u2^2,u3^2, u1u2,u1u3,u2u3]` (u = top-3 PC of weighted-standardized Z, same PCs as
the B3 interaction) in BOTH h0 and h1; h1 adds the UNCHANGED `c x PC1:3` interaction. Estimand `CE(h0)-CE(h1)`
subject-vote unchanged; SM16 features / folds / condition-class margins / invalid accounting unchanged; C=0.25;
B=200. Observed T may change (logged; `observed_T_repro_ok=true` for all 206 → deterministic).

## Hard safety screen → HARD FAIL + safety regression
| stratum (n=20) | method | **B4c** | oracle | cap |
|---|---|---|---|---|
| NULL_cov m+/o− | 20 | **20** | 0 | ≤5 (≥10 hard-fail) |
| NULL_cov_plus_label m+/o− | 20 | **20** | 0 | ≤5 |
| NULL_cov m−/o− (clean) | 0 | **4** | 0 | ≤1 |
| NULL_cov_plus_label m−/o− | 0 | **3** | 0 | ≤1 |

B4c removes **0/20 & 0/20** false-confirms (oracle removes 20/20 & 20/20) and **adds 4 & 3** on clean cohorts
(safety regression — B4c confirms cohorts the method did not).

## POS usefulness (m+/o+): retained but moot
POS_concept method 20 / B4c 16 ; POS_concept_plus_cov 20 / 19.

## Observed-T + null-dispersion — the mechanism
NULL_cov m+/o−: B3 observed_T median 0.00223 → **B4c 0.00331 (+49%, grew — no shrinkage)**.
null_sd median: method 0.000408 · B4a 0.000461 · B4b 0.000475 · **B4c 0.000359 (NARROWER than method)** · oracle
0.000677. `fixed_margin_p` floor 20/20; studentized_p median 0.005. **Enriching the fitted shared trunk lets h0
overfit the permuted-label null (narrower null) while the c×PC channel still exploits the real covariate–label
structure (larger observed T) → the collapse worsens.** The oracle's width comes from NOT fitting (the true
generator); no fitted-family enrichment (bagging, nesting, or richer trunk) replicates it.

## Verdict + B4 direction closed
**Stage-1b answer: NO.** No estimable fitted-h0-family null (variance-widened B4a/B4b, or capacity-enriched B4c)
approximates the oracle on real EEG covariate structure. Per the pre-registered stop rule, **no broad B4 model
search is opened.** The two remaining options are:
1. **Formally design a split-null calibration** (variance-inflation / p-recalibration estimated on locked
   calibration null blocks, evaluated on fresh held-out seed blocks) — a calibration PATCH, not a mechanism, and
   at risk of being Lee2019/SM16-specific; needs explicit authorization.
2. **Pivot to a selective risk-control / abstention router** — since no estimable null can be made safe, the
   deployable path is to ABSTAIN where the certifier cannot be trusted (the oracle diagnostic can help *label*
   unsafe regimes even though it cannot deploy). (Reviewer's stated default.)
**Neither taken here — awaiting reviewer authorization of one.**

## Verification (independent red-team, 2 lenses)
See `csc/results/b4_stage1b_canary/b4_stage1b_redteam_checks.json` (re-aggregation from raw + B4c-invariant /
implementation faithfulness: interaction channel unchanged, only the shared quad trunk added to both h0/h1, no
oracle/DGP leak, observed-T reproducible).

## Package (diagnostic-only, no tag)
`csc/results/b4_stage1b_canary/`: `b4_stage1b_manifest.json`, `b4_stage1b_results.jsonl` (206),
`b4_stage1b_tables.json`, `b4_stage1b_redteam_checks.json`, `SHA256SUMS`, `scripts/`. Merged sha256 `3eee5a49…`.
