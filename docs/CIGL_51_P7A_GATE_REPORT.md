# CIGL_51 P7a ERM-first gate — logvar vs cov_tangent, full-LOSO seed0, fusion_floor=0.05

42/42 folds (21 × {logvar, cov_tangent}), 0 NaN. Same branch/runner/CLI/floor → apples-to-apples. Primary
independently recomputed (−0.1233, matches aggregate). One fold (2015 t10 cov_tangent) hit an NFS stale-handle
from a submitter double-submit race; re-run clean to `..._rr.json` (compute was fine, `rc=0` ×3).

## Result

| metric (2a / BNCI2014_001) | logvar | cov_tangent | Δ(cov−log) |
|---|---|---|---|
| **CSP-decodable {1,3,8,9} mean** | 0.4852 | 0.3620 | **−0.1233** ⟵ PRIMARY (need ≥ +0.02) |
| full mean bAcc | 0.3789 | 0.3133 | −0.0656 (need ≥ 0) |
| source_bacc | 0.6706 | 0.4913 | −0.1793 |
| gate_spatial_mean | 0.4535 | 0.2197 | −0.2338 |
| zero_spatial ablation | 0.2743 | 0.2963 | +0.0220 (spatial NOT load-bearing under cov_tangent) |
| per-subj decodable | s1 .524→.344, s3 .549→.377, s8 .497→.349, s9 .372→.378 | | −.181/−.172/−.148/+.007 |

| metric (2015 / BNCI2015_001) | logvar | cov_tangent | Δ |
|---|---|---|---|
| full mean bAcc | 0.5818 | 0.6176 | **+0.0358** |
| source_bacc | 0.7503 | 0.7766 | +0.0263 |

cov_tangent conditioning: 2a eig_min pinned to the shrinkage floor α/C=0.0023 (near-rank-deficient covariances);
2015 eig_min 0.0041 (just above its floor 0.0038). feat_norm finite (78–124), no blow-up.

## Verdict: **FAIL** (pre-registered rule)

- PRIMARY 2a decodable Δ = **−0.1233** ≪ +0.02 → cov_tangent is **12pp worse** on the subjects that carry the
  4-class signal (3 of 4 crater, s9 flat). SECONDARY 2a full Δ = −0.066 < 0. → **cov_tangent does not beat
  logvar; treat like P6.**
- **NOT the memorization failure we guarded against** — source_bacc *dropped* (0.671→0.491). This is an
  **underfit**: the 2a band covariances are near-rank-deficient (eig_min at the shrinkage floor), so
  `vech(logm(S))` is dominated by shrinkage-noise directions and the linear head can't recover the 4-class CSP
  contrast. The floor=0.05 worked (gate_spatial 0.22, not 0), so this is a fair feature test — the gate then
  correctly routed *away* from the bad feature (graph 0.30→0.48), and `zero_spatial` shows the cov_tangent
  spatial branch is no longer load-bearing (drop +0.017 vs logvar +0.105).

## Honest nuance (not a rescue)

cov_tangent **helps 2015** (2-class, +0.036 full / +0.026 source), where covariances are better conditioned.
So the covariance-tangent geometry is not useless in general — it underfits the **near-rank-deficient 4-class
2a** covariances specifically. That could motivate a *new* conditioning-first variant (higher shrinkage, or
rank-reduction / spatial projection before the covariance) — but that is a **new hypothesis**, not P7a. Per the
pre-registered gate, P7a as specified FAILS.

## Consequences (PI-gated)

- **Do NOT advance to P7b** (TaskNullProjector was gated behind P7a passing — it did not).
- **Do NOT spend seeds 1/2** on cov_tangent.
- Standing kills unchanged: decodability-adaptive gating (killed), PCMI-TIF (monitor only), FDR (later).
- Provenance note: single-seed GPU results carry run-to-run nondeterminism (the logvar 2a-decodable baseline
  here is 0.485 vs P6's 0.466 for the identical config, within seed noise); the **same-run** cov−log delta is
  the valid comparison and it is decisively negative.
