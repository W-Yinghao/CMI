# Project B Step-2 Synthetic Router Report

> Frozen synthetic evidence for the Project B refusal-first EEG adaptation router. All numbers are
> reproduced from the locked Step-2E (`/tmp/project_b_step2e_router`) and Step-2F
> (`/tmp/project_b_step2f_support`) outputs by `scripts/project_b_step2g_report.py`
> (tables in `/tmp/project_b_step2g_report/`). No model is trained here; nothing is re-run; no
> `h2cmi/**` code is changed. Branch `project-b-refusal-router`.

## 1. Frozen protocol
- **Substrate:** the `h2cmi` EEG mechanism simulator; a hierarchical site→subject→session DAG. Full
  training (30 epochs, `--fast` disabled). Held-out **target = one unseen site**; the evaluation unit
  is the **subject** (4 target subjects per world-seed).
- **Three locked worlds** (established in Step-2A):
  - **R2** (recoverable): `cov1.2/prior0.4/montage0.2/concept0`, seeds 0,1,2.
  - **HF3** (harmful / concept-shift): `cov0.8/prior0.4/montage0.2/concept1.2/concept_frac0.5`, seeds 3,4,7,8,10.
  - **H_OOD** (target-only stress): `cov0.8/prior0.4/montage0.2/concept1.0/concept_frac0.17`, seed 32.
- **Label safety:** the router consumes target `X` only. Target `y` is used **only** for post-hoc
  balanced accuracy, after each `RouterDecision` is produced. All thresholds are **source-only**.
- **Two support-calibration modes** (Step-2F): `in_source_subject_q95` (Step-2E baseline) and
  `nested_site_excess_q95` (primary; base_source_q95 + scale-normalised nested held-out-site excess).

## 2. Router components under test
- **Actions:** `REFUSE / IDENTITY / OFFLINE_TTA` (offline mode). `OFFLINE_TTA` only ever estimates an
  affine `(A,b)` on frozen embeddings + a target prior; the encoder/classifier are frozen.
- **Support / TOS:** `n_target`, effective sample size (min-class responsibility ESS), prior-decoupled
  density NLL under source vs (estimated) target prior. Support blocker = target-prior density NLL over
  a **source-calibrated** threshold; low-ESS is a separate blocker.
- **ACAR harm:** conformal harm bound from source pseudo-target gains; when the source calibration set is
  degenerate/unavailable it yields **no bound** (never a fabricated one).
- **Action-specific blockers:** TTA-evidence / ACAR-harm failures block `OFFLINE_TTA` but **not** an
  otherwise support-valid `IDENTITY`. Selection is safe-beneficial-then-identity.

## 3. Main result

| world | strict bAcc | raw ΔTTA | base cov. | **nested cov.** | nested acc-bAcc | off-TTA rate | low-ESS refused | ACAR-harm |
|---|---|---|---|---|---|---|---|---|
| R2 | 0.855 | +0.071 | 0.00 | **0.83** | 0.880 | 0.00 | 2 | degenerate |
| HF3 | 0.568 | −0.071 | 0.00 | **0.60** | 0.569 | 0.00 | 7 | degenerate |
| H_OOD | 0.579 | −0.231 | 0.00 | **0.50** | 0.522 | 0.00 | 2 | degenerate |

The Step-2E baseline (`in_source_subject_q95`) **refuses everything** (coverage 0 in all worlds) because a
threshold calibrated on in-distribution source subjects (q95 ≈ 4–5) is systematically below every held-out
target's density NLL (≈ 8–11). The nested source-site excess calibration (primary) **restores support-valid
IDENTITY coverage** while keeping `OFFLINE_TTA` blocked everywhere (ACAR-harm degenerate).

## 4. R2 recoverable world
- The base model transfers well (strict bAcc 0.855; raw TTA would add +0.071).
- **Baseline: all-refuse (coverage 0).** **Nested: coverage 0.83**, `accepted_bacc` 0.880 — recoverable
  targets are accepted as IDENTITY. `SUPPORT_MISMATCH` domains 12→0.
- **`OFFLINE_TTA` stays blocked** (ACAR-harm degenerate), so raw TTA's +0.071 is a **missed benefit**
  (mean 0.075). This is the intended conservative v1 posture, not a policy bug.

## 5. HF3 harmful / concept-shift world
- Raw offline TTA is **harmful** on average (ΔTTA −0.071; e.g. seeds 4/8 at −0.14/−0.15).
- **Nested: coverage 0.60, `OFFLINE_TTA` blocked in every seed** (`tta_block` = `ACAR_HARM_CALIBRATION_DEGENERATE`
  + `TTA_NEGATIVE_EVIDENCE`) — the harmful adaptation is avoided.
- **Honest caveat:** the identity outputs that pass support are **concept-degraded** (`accepted_bacc` 0.569).
  Source-only support **cannot** detect concept-shift accuracy loss; the router blocks the harmful *TTA* and
  emits the base prediction, without claiming the base is accurate.

## 6. H-OOD target-only stress world
- Raw offline TTA is strongly **harmful** (ΔTTA −0.231).
- **Nested: coverage 0.50.** The nested threshold widens to 14.4 (a genuinely-OOD source site inflates the
  fold excess), so **`SUPPORT_MISMATCH` no longer fires** (4→0). **`LOW_ESS` remains the active blocker on
  2/4 subjects** (ESS 5.8 / 7.8) → those refuse; `OFFLINE_TTA` blocked throughout.
- So for H_OOD the *density* threshold is no longer discriminative after nested widening; the
  *effective-sample-size* check is what still catches the low-support subjects. Reported as a **PARTIAL**.

## 7. Reason-code audit
`table3_reason_code_audit.csv` separates **top-level decision** reasons from **action-level** reasons/blockers.
Key invariant, confirmed: `OACI_ACAR_HARM_CALIBRATION_DEGENERATE` appears as an **offline-TTA blocker**
(count 20 on HF3) with **identity-action count 0** — the TTA blocker never masquerades as "identity is unsafe".
Refusals are attributable: `SUPPORT_MISMATCH` (baseline over-refusal, 12/20/4 → 0/1/0 under nested),
`LOW_EFFECTIVE_SAMPLE_SIZE` (persistent, the real support signal), `ACAR_HARM_CALIBRATION_DEGENERATE`
(TTA block), `TTA_NEGATIVE_EVIDENCE` (TTA block), `LEAKAGE_RESIDUAL_UNAVAILABLE` (audit), `PRIOR_SHIFT_ONLY_INFO`
(audit — prior shift never refuses).

## 8. What the router can claim
1. Default Project B v1 **never selects `OFFLINE_TTA`** when ACAR-harm calibration is degenerate/unavailable.
2. Nested source-site support **excess** calibration **fixes** the Step-2E all-refuse failure on R2.
3. Project B can **allow support-valid IDENTITY** while still refusing low-ESS targets.
4. OACI reason codes **expose** whether a refusal came from support, ESS, ACAR degeneracy, or TTA evidence.

## 9. What the router cannot claim
1. It **cannot** claim source-only ACAR-harm is generally identifiable.
2. It **cannot** claim to recover R2's raw TTA benefit — default v1 blocks TTA under harm-calibration degeneracy.
3. It **cannot** claim support-valid identity is accurate under concept shift; HF3 shows concept-degraded
   identity can pass support checks.
4. It **cannot** claim the density support threshold alone catches H-OOD after nested widening; `LOW_ESS`
   is the active blocker for only part of H-OOD.
5. It **cannot** claim target-label-tuned thresholds; thresholds are source-only and label-safe.

Machine-readable: `claim_boundary.json` (and `notes/PROJECT_B_CLAIM_BOUNDARY.md`).

## 10. Next engineering step
Project B v1 is positioned as a **high-precision, support-valid IDENTITY router** — not a high-coverage
adapter, not a complete concept-shift detector, not a guaranteed OOD detector under all calibration modes.
The two parallel exits from the frozen synthetic package:
- **Step-3A** — real-EEG bridge (existing `cmi/` MOABB loaders or an h2cmi-compatible wrapper).
- **Step-3B** — paper section draft (method, protocol, ablations, limitations, claims), grounded in the
  frozen tables and the claim boundary above.
