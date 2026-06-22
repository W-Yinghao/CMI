# ACAR v3 — DESIGN DRAFT

**Status:** `DRAFT / NOT FROZEN / NO LOCKBOX ENDPOINT ACCESSED`  
**Date:** 2026-06-22  
**Parent result:** ACAR v2 `MEASUREMENT_ONLY` (`acar-v2-protocol` @ `9b2f0c1`; result `1528a94`)  
**Purpose:** Close the **measurement → risk-regression → calibrated-control** gap without rewriting the v2 endpoint.

This file is a design-stage document. It is **not** `ACAR_FROZEN_v3.md` and must not authorize a binding held-out run.

> **⚠ RATIONALE-ONLY; ALL NORMATIVE RULES SUPERSEDED (Amendments 1+2, 2026-06-22).** This entire draft is retained
> for scientific motivation only. **Every normative rule is defined by `notes/ACAR_V3_FREEZE_SKELETON.md`** (folding in
> `ACAR_V3_AMENDMENT_1.md` + `ACAR_V3_AMENDMENT_2.md`); where this draft disagrees, the skeleton governs. Known stale
> spots: **V3.0** frames the predictor as mean/scale only — the candidate set is **C0/C1/C2/C3 (CQR added)**; **V3.2–
> V3.6** predate candidate-specific disease×action calibration, the random subject-hash split with exchangeable
> same-site coverage (not "historical/future"), the two-site rule + single-site contingency, C2 `q⁺=max(q,0)`,
> disease-specific models, and the harmful-rate single-statistic; **V3.8** lists only `σ` guards — see skeleton §S8/§S13
> for the full set-contract + heteroscedastic guards; **V3.10** lists only mean/scale implementation — CQR (C3) and the
> two-phase lock apply. Do not generate a frozen protocol from this draft.

## V3.0 Scientific question

Can an action-conditioned, permutation-invariant predictor of the **conditional mean and scale** of batch incremental risk produce tighter subject-level simultaneous upper bounds than the v2 batch-summary HGB, while preserving label-free deployment and the same operating point (`alpha=0.10`, `delta=0`)?

The target remains

\[
\Delta R_a(B)=R_B(f_a)-R_B(f_0),
\]

for `a ∈ {matched_coral, spdim, t3a}` relative to `identity`.

## V3.1 Fixed scope

Unchanged from v2 during the initial v3 line:

- Primary risk: NLL.
- Actions: identity, matched-CORAL, SPDIM, T3A.
- Natural recording-ordered batches; `MIN_BATCH=8`; small batches retained and forced to identity.
- Label-free deployment inputs only.
- CMI/density coordinates are context only; no fixed risk direction.
- `alpha=0.10`, `delta=0`.
- Batch routing is primary; sample abstention remains a separate future model and endpoint.

## V3.2 Proposed method: Heteroscedastic Set-Conformal Router (HSCR)

### Per-window paired set

For each batch `B`, action `a`, and window `i`, compute label-free paired features from `(p0_i, pa_i, z_i, ztilde_{i,a})`, including pre/post entropy and margin, their differences, flip indicator, per-window JS, embedding displacement, confidence change, and pseudo-label agreement. Batch-level Bures, post-separation, effective sample size, and source-context features are appended after pooling.

No target label may enter feature construction, feature normalization, architecture selection, or deployment.

### Action-conditioned DeepSets predictor

\[
h_a(B)=\rho\left(\operatorname{pool}_{i\in B}\psi(x_{i,a}), c_a(B), e_a\right),
\]

where `e_a` is an action embedding and pooling is permutation-invariant. A shared trunk with action-specific output heads predicts

\[
\hat\mu_a(B),\qquad \hat\sigma_a(B)>0.
\]

All preprocessing statistics, model parameters, and scale clipping constants are fit on FIT only.

### Standardized group conformal score

For CAL subject `s`:

\[
S_s=\max_{B\in\mathcal B(s)}\max_a
\frac{\Delta R_a(B)-\hat\mu_a(B)}{\max(\hat\sigma_a(B),\sigma_{min})}.
\]

With the finite-sample corrected subject quantile `q`, define

\[
U_a(B)=\hat\mu_a(B)+q\max(\hat\sigma_a(B),\sigma_{min}).
\]

Route to `argmin_a U_a` only when `U_a<0`; otherwise identity. The score remains simultaneous over actions and the fixed finite batch set of an exchangeable subject because `mu`, `sigma`, and the score function are frozen before CAL.

## V3.3 DEV versus lockbox

### DEV cohorts — method development only

- PD: ds002778, ds003490, ds004584.
- SCZ: ds003944, ds003947, ds004000, ds004367.

These seven cohorts have already informed v2 and cannot serve as v3 confirmatory evidence.

### Candidate held-out cohorts — metadata shortlist only

PD:

- OpenNeuro ds007020, EEG Mortality Dataset in Parkinson's Disease.
- OpenNeuro ds007526, Resting-State & Walking EEG in Parkinson's Disease; use only the pre-specified resting condition.

SCZ:

- Zenodo 10.5281/zenodo.14808296, 38 schizophrenia + 39 controls, 64-channel resting EEG.
- Zenodo 10.5281/zenodo.14178398 (ASZED), 76 schizophrenia + 77 controls; use only the pre-specified resting condition.

Before lockbox designation, perform a metadata-only audit of license, raw-signal availability, montage, sampling rate, task condition, subject identity, overlap, class labels, sample size, and preprocessing compatibility. Do not run ACAR inference or inspect adaptation outcomes.

## V3.4 Development gate

Use nested subject-disjoint CV on DEV only to compare:

1. frozen v2 HGB + raw joint residual;
2. DeepSets mean-only + raw residual;
3. DeepSets mean/scale + standardized residual (primary candidate).

The lockbox is not consumed unless the selected model meets all pre-lock criteria:

- At least one PD action has risk-regressor harm AUROC `>=0.60`.
- SCZ risk regression is not worse than the corresponding v2 predictor.
- Disease-macro upper-bound width falls by at least 30% relative to v2.
- Adaptation coverage at `alpha=0.10, delta=0` is at least 15%.
- Router NLL reduction is positive and not below the frozen v2 router.
- All leakage, split-isolation, set-invariance, serialization, and double-run determinism guards pass.

Failure closes v3 at development stage without reading held-out endpoint labels.

## V3.5 External evaluation arms

### Arm A — zero-shot external robustness

Predictor and calibration are learned from DEV only. Held-out cohorts are used only for empirical external utility and coverage diagnostics. No cross-cohort finite-sample theorem is claimed.

### Arm B — site-local conformal safety, proposed primary arm

The predictor is frozen from DEV. Each held-out cohort is deterministically split by subject hash into a historical CAL subset and a future EVAL subset. CAL labels may be used **only** to compute the site-local conformal quantile; they may not retrain the predictor, choose hyperparameters, alter actions, or tune `alpha/delta`. Deployment on EVAL remains label-free.

The coverage claim is restricted to exchangeable future subjects from that site and the fixed finite batching protocol.

## V3.6 Proposed binding endpoints

Exact thresholds must be frozen before lockbox evaluation. Current proposal, per disease:

### G1 — external measurement/regression

- At least one action-specific held-out harm AUROC `>=0.60`.
- Continuous risk prediction improves over v2 by a frozen subject-clustered error metric.

### G2 — useful calibrated control

All must hold:

- `red_router > 0`.
- `red_router > red_bestfixed`, where best-fixed is selected on DEV only.
- `red_router > red_v2_router` on the identical held-out EVAL subjects.
- Oracle-benefit retention `>=0.50`.
- Adaptation coverage `>=0.20`.
- Each held-out cohort improves over identity, or at most one cohort fails when at least three cohorts are available for that disease.
- Harmful adapted-batch rate is lower than best-fixed under a frozen subject-clustered test.

### Coverage diagnostic

Report exact site and disease subject-event coverage. Do not use the brittle rule “observed coverage must be at least 0.90.” Pre-register a one-sided exact binomial lower-tail undercoverage test per site, with Holm correction across sites, while retaining the formal theorem's explicit exchangeability assumptions. Failure of the diagnostic is not silently ignored.

## V3.7 Proposed decision taxonomy

- `PROCEED_SAFE_ROUTER`: G1, G2, and coverage diagnostic pass.
- `UTILITY_ONLY`: G2 passes but the coverage diagnostic fails.
- `MEASUREMENT_ONLY`: G1 passes but G2 fails.
- `TERMINATE`: held-out G1 fails.
- `RUN_QUARANTINED / PROTOCOL_INVALID`: guards, split isolation, provenance, or preregistration integrity fails.

## V3.8 New hard guards

Retain all v2 guards and add:

- Set permutation invariance of `mu`, `sigma`, `U`, and action.
- Action-order invariance.
- FIT-only normalization and scale-floor derivation.
- Strict positivity and finiteness of `sigma`.
- CAL-label changes affect EVAL only through the calibrated standardized quantile.
- EVAL-label permutation leaves `mu`, `sigma`, `q`, `U`, and actions bit-identical.
- Serialization round-trip for the set encoder and all preprocessing state.
- Record-level hash includes per-window paired inputs or their canonical digest, `mu`, `sigma`, subject score, `q`, `U`, and chosen action.

## V3.9 Provenance repairs

The v3 manifest must record distinct units:

- `n_fit_subjects`, `n_fit_batches`
- `n_cal_subjects`, `n_cal_batches`
- `n_eval_subjects`, `n_eval_batches`

It must store full 64-character SHA-256 digests for every raw/derived dump, exact dataset version/DOI, subject-list hashes, split assignments, source-state hash, protocol commit, immutable tag, environment lock, and double-run hash.

## V3.10 Immediate work plan

1. Add this file to `notes/` as a non-frozen draft.
2. Run metadata-only feasibility checks on candidate lockboxes.
3. Implement per-window paired-set extraction and synthetic invariance tests.
4. Implement the action-conditioned mean/scale set predictor and standardized subject conformal score.
5. Run the DEV gate only.
6. If the gate passes, select one model and one lockbox list, write `ACAR_FROZEN_v3.md`, commit/tag a clean worktree, and run the binding external evaluation once.

## Source pointers

- v2 protocol: https://github.com/W-Yinghao/CMI/commit/9b2f0c1
- v2 frozen result: https://github.com/W-Yinghao/CMI/commit/1528a94
- v2 audit note: https://github.com/W-Yinghao/CMI/commit/6a0c3d0
- coverage wording fix: https://github.com/W-Yinghao/CMI/commit/ce5c330
- PD candidate ds007020: https://openneuro.org/datasets/ds007020/versions/1.0.0
- PD candidate ds007526: https://openneuro.org/datasets/ds007526/versions/1.0.2
- SCZ candidate: https://zenodo.org/records/14808296
- SCZ candidate ASZED: https://zenodo.org/records/14178398
