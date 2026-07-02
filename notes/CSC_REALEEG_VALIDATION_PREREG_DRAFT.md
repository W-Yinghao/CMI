# CSC on REAL EEG — validation PRE-REGISTRATION (SIGNED-OFF DESIGN v3, not frozen, not run)

Status: **v3 — reviewer-signed-off choices frozen (2026-07-02).** Red-team-hardened (v1→v2 fixed 2 blockers)
and now with the reviewer's §11 sign-off locked into the body. This document still creates no tag, writes no
manifest, builds no adapter, and runs no validation bank. Authorized next steps (this design + build PLAN +
dry-run/manifest build + a cache feasibility check that reads metadata and computes `Z`/rank) are bounded in
§13. The frozen synthetic tags `csc-confirmatory-v1`/`dee8958` (A) and `csc-b3-confirmatory-v1`/`0595f64` (B3)
are untouched; both certifier method locks are **byte-unchanged**.

## 0. Motivation, ceiling, bridge, and Route-A framing

All CSC evidence so far is **synthetic**; real EEG can differ sharply (1/f, non-stationarity, artifacts,
heteroscedasticity, subject heterogeneity). We test whether the two headline conclusions **transfer**:
- **Route B3 (positive):** a paired within-subject minimal-label certificate controls false confirmation and
  has power.
- **Route A (negative):** we **evaluate** the frozen $Z$-only certificate on the *same* real-feature injected
  bank to test whether the synthetic A-negative transfers. **A is not required to fail for the study to PASS.**
  If A performs well on real-feature data, that is an important finding, not a protocol failure — but it does
  not retroactively change any B3 claim.

**Ceiling:** genuine real EEG has **no ground-truth concept-shift label** → on genuine contrasts we can only
describe, never measure power. **Bridge:** *semi-synthetic on real EEG features* — keep real covariate/noise/
subject structure, inject a **known** shift onto the real features → ground truth restored → both
false-confirmation and power measurable for A and B3 on realistic data.

## 1. What a PASS does and does not mean (quotable guardrail — reuse verbatim downstream)

> **A PASS means the frozen certifiers satisfy the pre-registered false-confirmation and validity gates on
> real-EEG features with injected, known ground truth. Power on POS_concept / POS_concept+cov, and the
> A-vs-B3 gap, are pre-registered REPORTED outcomes, NOT PASS/FAIL gates. A PASS does NOT mean any real
> session-contrast `CONCEPT_CONFIRMED` verdict is validated truth, does NOT report power on genuine real
> shifts, and does NOT establish any clinical or PD result. Genuine-contrast findings are descriptive only.**

## 2. Data and mapping (feasibility verified by red-team)

**`Lee2019_MI` (OpenBMI)** via `cmi/data/moabb_data.py`, offline datalake cache
(`configure_offline_moabb()`, no network): 54 subjects × 2 sessions × Left/Right MI, 62 channels, 8–30 Hz,
128 Hz, window 0.5–3.5 s. **Exact trials/class/session are read at build time into the manifest** (do not
hard-code "~100"). Mapping to the certifier tuple `(s,c,e,Y,Z)`: subject `s`; condition `c=±½` = session 1/2;
epoch `e` = MI trial; label `Y` = Left/Right; features `Z` = §3.

## 3. Feature `Z` — FROZEN (signed off: 1A, exact montage frozen now)

```
Z_ch      = log( var_t(x_ch) )        # per-trial/epoch log band-power
x         = UN-normalized 8-30 Hz trial signal   (load(..., normalize=None))
normalize = None                       # NEVER the loader trial_zscore (v1 blocker: makes log-var == 0)
feature   = log-bandpower              # primary family, single, frozen
```

**Frozen montage (17 sensorimotor channels), exact:**
`FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4`.
→ `Z ∈ R^{17}` per trial.

**Fail-closed feature rule (no substitution, ever):** if **any** frozen channel is absent from
Lee2019/OpenBMI, or `Z`-dim < 6, or `rank(Z) < 3`, the primary feature **FAILS CLOSED**: the manifest build /
cache build **stops and reports**; we re-freeze the montage. **No intersection, no auto-substitute montage, no
switching to covariance tangent-space within this protocol.** No external PCA (certifier has no in-fold hook;
redundant with its internal weighted-SVD rank-3 PC basis).

**Robustness-only secondary (pre-declared):** covariance tangent-space — **cannot change PASS/FAIL**, reported
regardless of outcome, never in the main endpoint.

## 4. Method locks (UNCHANGED) + pinned hashes + bootstrap terminology

- **Route B3:** `pc_centered_calibrated`, `p24d_cross_budget_alpha_spending_studentized_fixed_margin`;
  `h0:[Z,c]` vs `h1:[Z,c,c×Z_pc(rank 3)]`, centered ±½, `C=0.5`; condition-matched fixed-margin `h0`
  bootstrap; studentized gate `LCB_{0.975}(Δ_s)>0`; family `α=0.05`, budgets `{20,30}`, `α_budget=0.025`;
  `n_folds=3`; subject-grouped cross-fit; subject-condition class-balanced weights.
- **Route A:** the frozen `dee8958`-line 3-state certifier, unchanged; hashes pinned at build.
- **Pinned hashes (B3):** `sha256(paired_calibrated.py)=26e505ed…` AND
  `sha256(paired_conditional_test.py)=1263f672…`, plus `paired_certifier.py`.
- **Two distinct bootstraps (split terminology, no ambiguity):**
  - `B_certifier = 200` — the certifier's INTERNAL fixed-margin null bootstrap. **Fixed by the method lock; do
    NOT change for real EEG.**
  - `B_subject = 2000` — the AGGREGATE subject-clustered bootstrap used only for REPORTED cohort bounds (R1/R5).
    Not part of the method lock.
- **α (frozen):** each null cohort's certifier decision uses `α_budget=0.025`; the AGGREGATE false-confirmation
  bound target is family `0.05`. Both intentional, frozen, not a post-hoc relaxation.

## 5. Eligibility guards (contract)

Pair integrity ≥ 0.95; ≥ 8 trials/condition/class; per-condition class coverage; **≥ 20 eligible paired
subjects**; `Z`-dim ≥ 6 & `rank(Z) ≥ 3`; else `NEED_MORE_LABELS`/`INVALID_PAIR_STRUCTURE`.

## 6. Semi-synthetic injection bank on REAL features (ground-truth core)

Each cohort takes real `Z` (real covariate structure) and injects a **known** truth, subject-grouped cross-fit,
no leakage:

| kind | construction on real `Z` | truth | role |
|---|---|---|---|
| **NULL_cov** (GATING) | real session-1/2 split (real covariate drift) + `Y* ~ p̂(Y\|Z)` from a SINGLE pooled boundary fit on both sessions | no concept, real covariate shift | **primary type-I** |
| NULL_exch | within-session stratified random split (mean/balance-matched) | no concept, no covariate | calibration check only (log \|meanZ_A−meanZ_B\|) |
| POS_concept | real session split + injected antisymmetric boundary rotation in one condition | concept present | **power** (reported) |
| POS_concept+cov | rotation on top of real covariate drift | concept + covariate | power under confound (reported) |
| LABEL | shift `P(Y)` prior across conditions, `P(Z\|Y)` fixed | label shift (must abstain) | trap control |
| PURE_COND | invisible relabel (boundary change, marginal fixed) | pure-conditional | secondary, weak-by-theory |

Synthetic bank additionally strengthened with **scale/heteroscedastic covariate nulls** before the real run
(the existing synthetic `paired_covariate` is a rigid mean-offset only and already shows elevation).

## 7. Genuine real contrast (DESCRIPTIVE, non-gating)

One certifier run on the real session-1-vs-2 cohort → verdict + `T` + studentized `Z_subj` + LCB + per-subject
`Δ_s`. Report the **fraction of subjects with `Δ_s>0`** and a **subject-bootstrap CI** on mean `Δ_s`.
**Descriptive only:** the constructed nulls do not fully reproduce the genuine between-session covariate regime,
so a real `CONCEPT_CONFIRMED` is a reported verdict, **not** validated concept drift; a non-confirmation is
**not** evidence of absence (unmeasurable power).

## 8. Endpoints / PASS criteria (validity, not truth-detection)

Gating (conjunction), per route, on the §6 bank:
- **R1 (primary type-I) — GATING:** on **NULL_cov**, the subject-clustered bootstrap (`B_subject=2000`) upper
  bound on the false-confirmation rate ≤ family `0.05` (resample SUBJECTS, not label-draws). Denominator =
  **valid** cohorts only; invalid/abstain fraction reported separately and capped at ≤ 20% (family
  non-estimable above the cap; abstains do NOT pad the type-I denominator). NULL_exch reported alongside.
- **R2 (power) — REPORTED, NOT GATING:** on POS_concept / POS_concept+cov, B3's confirmation rate with a
  subject-bootstrap lower bound, and the **A-vs-B3 gap** (the descriptive headline). **If B3 power is low, the
  result is not re-run or rescued; it is reported as limited transfer of the synthetic positive to this
  real-feature pipeline.** Power never sets PASS/FAIL.
- **R3 (guards) — GATING:** eligibility holds (§5).
- **R4 (no silent failure) — GATING:** every state in the valid 5-state set; sampler/bootstrap invalid fraction
  below the pre-registered cap (NOT "exactly 0").
- **R5 (stability of the type-I control) — GATING:** R1 (NULL_cov control) **replicates** on **Lee2019**
  under subject-bootstrap / leave-k-subjects-out / disjoint subject-half. A control that does not replicate is
  not a control. **`BNCI2014_004`/2b (tangent-space) is ROBUSTNESS-ONLY: reported if feasible, cannot change
  PASS/FAIL, NOT gating.**
- **R6 (red-team) — GATING:** independent re-aggregation reproduces the GATING criteria **R1, R3, R4, R5**
  without correction. R2/R7 reproduction is reported, never changes PASS/FAIL.

Non-gating / descriptive: **R7** genuine-contrast report (§7); R2 power + A-vs-B3 gap; LABEL/PURE_COND bank;
NULL_exch; within-session contiguous-drift probe; the tangent-space secondary.

**Interpretation.** A PASS = *on real EEG features with injected known truth, the frozen certifiers satisfy the
pre-registered false-confirmation and validity gates (incl. type-I under real covariate drift), and replicate*.
Power and the A-vs-B3 gap are reported to describe transfer, not to gate. A FAIL on R1 for either route is an
honest **negative** (type-I does not transfer under this feature pipeline) — reported, not retried (§9). It
never claims a genuine real verdict is correct (§1).

## 9. Freeze discipline, stopping rule, anti-gaming

- **Single frozen primary** feature + montage (§3); no feature-family selection after unblinding.
- **Stopping rule:** a FAIL on R1 for the frozen primary is the reported result; it is **not** grounds to swap
  to the secondary and re-judge. A later feature is a NEW pre-registration disclosing the original FAIL.
- No optional stopping / threshold search; feature, montage, α, bank, `B_certifier`, `B_subject`, invalid cap,
  criteria all frozen in the manifest before any run. Pin scipy in the run env so the exact Student-t LCB path
  executes; record which path ran; add a subject-bootstrap LCB robustness cross-check.

## 10. Freeze package plan (built only after this v3 is used to build; NO run without a separate go)

- **Isolated code** in `csc/mininfo/` (no `cmi` import): one-time `build_lee2019_b3_cache.py` runs in **eeg2025**
  → `LEE2019_B3.npz` (`Z, subject, session, y, classes, channel_list, exact counts, provenance`); certifier/
  runner read ONLY that cache (env `icml`, CPU/SLURM). Fail-closed if any frozen channel is absent (§3).
- **Two manifests** (B3 + A): data source + exact counts + frozen 17-ch montage + bank spec + injection seeds
  + criteria R1–R6 + pinned code hashes (both B3 files) + disjoint seed base (> all synthetic ranges, verified).
- **Runners** (`run_b3_realeeg.py`, `run_a_realeeg.py`): dry-run + guarded `--execute`, fail-closed provenance,
  conservative denominators, SLURM wrapper. **Tag:** `csc-realeeg-v1`. Tests + independent audit **before** any
  run.

## 11. FROZEN choices (reviewer sign-off, 2026-07-02)

1. **Primary feature:** log-bandpower (`normalize=None`), 17-ch sensorimotor montage in §3. Frozen. Fail-closed,
   no substitute. Tangent-space = robustness-only.
2. **Scope:** Route A **and** Route B3 on the same real-feature injected bank.
3. **R2 power:** reported only, **not** gating; A-vs-B3 gap is the descriptive headline.
4. **R5:** Lee2019 subject-half/bootstrap stability is **gating**; 2b/tangent-space is **robustness-only** (not
   gating).
5. **Bootstraps:** `B_certifier = 200` (method lock, fixed); `B_subject = 2000` (subject-clustered reported
   bounds). Invalid-fraction cap 20%. α = 0.025 per decision cohort / 0.05 family.
6. **Tag** `csc-realeeg-v1`; cache build env `eeg2025`; certifier run env `icml`.

Build-time reads (recorded in manifest, not frozen here): exact trials/class/session counts; confirmation that
all 17 montage channels are present; the exact returned channel list.

## 12. Out of scope / NOT authorized by this document

No clinical/PD claim (MI cross-session, not medication). No change to either method lock. No touching the
synthetic tags. See §13 for the exact authorization boundary.

## 13. Authorization boundary (reviewer, 2026-07-02)

**Authorized now:** this v3 cleanup; the freeze-package build PLAN; a **dry-run / manifest build only**; a
**cache feasibility check** that reads Lee2019 metadata and computes `Z`/rank (verify the 17 channels exist and
`Z` is non-degenerate with `rank ≥ 3`, `dim ≥ 6`).
**NOT authorized:** running the real-EEG validation bank; creating the `csc-realeeg-v1` tag; executing Route A
or B3 certifiers on real-feature injected cohorts; using 2b as gating; switching feature family after a
failure; treating any genuine-contrast verdict as validated.
**If the cache feasibility check reveals the primary feature is invalid (degenerate / rank < 3 / channel
absent): STOP and report. Do NOT switch to covariance tangent-space within this protocol.**
