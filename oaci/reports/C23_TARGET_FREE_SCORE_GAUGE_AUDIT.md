# C23 — Target-Free Score Calibration / Gauge Audit (frozen C19 `664007686afb520f`)

> Read-only MECHANISM audit: can the per-target score OFFSET that breaks the C20/C22 pooled cross-target estimand be explained/reduced by TARGET-FREE, SOURCE-ONLY, TARGET-ANONYMOUS gauge summaries — without target identity, target labels, target-wise centering, source subject IDs, or checkpoint selection? NOT a selector, NOT an OACI rescue, NOT deployable calibration.

- **CASE: `G5_offset_source_unobservable`**
- the per-target offset is NOT predictable from the allowed target-free source summaries -> within-target competence is source-visible, but cross-target calibration is not identifiable from tested source evidence.
- next: strong mechanism boundary: cross-target calibration is not identifiable from tested source summaries; package as final.

## HARD GATE — target-identity-leakage audit (reported FIRST, gates any positive claim)

- 9-way target-id accuracy from raw source features **+0.541** vs chance **+0.111** (ceiling 0.35) → source features identity-separable: **True**
- If source features predict target id far above chance (1/9), the per-target gauge is identity-laden; a positive calibration then only counts if the offset relationship GENERALIZES leave-one-target-out (offset_model.loto), else it is G3.

> In LOSO the source composition ≈ target identity. If source features carry target id, a per-target gauge can only count as target-free calibration if the offset relationship GENERALIZES leave-one-target-out (offset LOTO below); otherwise it is G3.

## Calibration ceiling ladder (pooled AUC; oracle rungs use target identity = NON-deployable ceilings)

| rung | pooled AUC | deployable |
|---|---:|:--:|
| raw (no calibration) | +0.543 | yes |
| regime-centered | +0.543 | yes |
| **source-gauge LOTO** | **+0.468** | **yes (target-free)** |
| target-centered ORACLE | +0.634 | no (uses target id) |
| target-rank ORACLE | +0.648 | no |
| within-target ceiling | +0.659 | no |

- source-gauge AUC improvement over raw: **-0.075** (success ≥ 0.03)
- oracle gap closed by source gauge: **-0.825** (success ≥ 0.4)

## Offset model (fixed ridge L2=1.0, leave-one-target-out; no grid search, no feature selection)

- LOTO offset R² **-2.354** (in-sample +0.956); permutation p **+0.642** (null mean -1.816) → LOTO beats permutation: **False**
- residual offset std **+0.283** of true offset std **+0.155** (residual/true +1.827)

## Secondary — risk-family gauge (R_src per-target mean ONLY; static training scalar)

- R_src-only LOTO offset R² **-0.427**, source-gauge AUC **+0.521**, gap closed **-0.245** — SECONDARY: R_src is a training-realized static scalar, not a robust-core deletion-robust observable.

## Epoch/order residual carry-forward (gauge excludes epoch/order)

- corr(offset, mean epoch) **-0.049**, corr(offset, mean order) **-0.130**, corr(gauge LOTO prediction, mean epoch) **-0.257** → offset is an epoch proxy: **False**

## Boundary of the claim

> DIAGNOSTIC-ONLY mechanism audit. The target-centered/rank rungs are ORACLE ceilings (they use target identity) and are NOT deployable. No selector is produced; no per-candidate checkpoint is selected; the C19/C20 estimand boundary is characterized, not rescued.