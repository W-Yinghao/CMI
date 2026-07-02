# CSC on REAL EEG — validation PRE-REGISTRATION (DESIGN DRAFT v2, not frozen, not run)

Status: **DESIGN DRAFT v2 for reviewer approval.** Red-team-hardened successor to a v1 B3-only draft that a
3-lens adversarial review returned with **2 blockers** (degenerate primary feature; primary null did not test
the real covariate-shift failure mode). This document specifies what a real-EEG validation of the CSC
conclusions *would* lock and how it *would* be judged. It creates no tag, writes no manifest, builds no
adapter, downloads/touches no data, and runs nothing. The frozen synthetic tags `csc-confirmatory-v1`/`dee8958`
(Route A) and `csc-b3-confirmatory-v1`/`0595f64` (Route B3) are untouched; both certifier method locks are
**byte-unchanged** — only the data source and the (semi-synthetic) truth injection are new. A run happens only
after this design is approved, the open choices in §11 are frozen, AND a separate explicit go.

## 0. Motivation (reviewer's concern) and the honest ceiling

All CSC evidence so far is **synthetic**, and real EEG can differ sharply (1/f spectra, within/cross-session
non-stationarity, artifacts, heteroscedasticity, volume conduction, large subject heterogeneity). The goal is
to test whether the two headline conclusions survive on real EEG:
- **Route A (negative):** a strong $Z$-only certificate fails to certify concept shift under control.
- **Route B3 (positive):** a paired within-subject, minimal-label certificate controls false confirmation and
  has power.

**Ceiling:** genuine real EEG has **no ground-truth concept-shift label**, so on genuine contrasts we can never
measure power/correctness — only describe. **Bridge:** *semi-synthetic on real EEG features* — keep real
covariate structure/noise/subject effects, inject a **known** shift onto the real features, and thereby restore
ground truth. This lets us measure **both** false-confirmation **and** power for A and B3 on realistic data.
This bridge is also exactly the fix the red-team requires (its covariate-only null is a semi-synthetic
injection on the real session split).

## 1. What a PASS does and does not mean (quotable guardrail — reuse verbatim downstream)

> **A PASS means the frozen certifiers behave as claimed on real-EEG features with injected, known ground
> truth (false confirmation controlled; B3 powered where A is not). It does NOT mean any real
> `CONCEPT_CONFIRMED` verdict on a genuine session contrast is validated truth, does NOT report power on
> genuine real shifts, and does NOT establish a clinical or PD result. Genuine-contrast findings are
> descriptive only.**

## 2. Data and mapping (feasibility verified by red-team)

**`Lee2019_MI` (OpenBMI)** via `cmi/data/moabb_data.py`, offline datalake cache
(`configure_offline_moabb()`, no network): 54 subjects × **2 sessions** × Left/Right MI, 62 channels, 8–30 Hz,
128 Hz, window 0.5–3.5 s. Exact trials/class/session are **read at build time into the manifest** (OpenBMI has
an offline + online phase; do not hard-code "~100"). Mapping to the certifier tuple `(s,c,e,Y,Z)`:
subject `s`; condition `c=±½` = session 1/2; epoch `e` = MI trial; label `Y` = Left/Right; features `Z` = §3.
Stability second dataset (if used): see §8 R5.

## 3. Feature `Z` (FROZEN before run; the one genuinely new choice — §11 for sign-off)

The certifier is dimension/scale-agnostic (internal per-fold weighted standardization + weighted-SVD rank-3 PC
basis on TRAIN only), so `Z` needs **zero** certifier code change. **No external PCA** (the certifier has no
in-fold transform hook; an external fit would be leaky or would break the pinned hash, and it is redundant with
the internal PC basis — red-team major #7).

**Proposed FROZEN primary:** per-channel **log band-power from the UN-normalized signal** (`load(..., normalize=None)`;
`Z_ch = log(var_t(x_ch))`), computed per trial (leakage-free as a per-trial statistic), on a **pre-registered,
data-independent channel subset** (sensorimotor montage, exact list frozen in the manifest). This fixes the v1
degeneracy (blocker #4) and needs no fitting. Dimensionality guard: require `Z`-dim ≥ 6 and assert `rank(Z) ≥ 3`
(else fail closed to `NEED_MORE_LABELS`), so the locked rank-3 interaction is genuinely rank-3 (red-team #8).
**Robustness-only secondary (pre-declared, cannot change PASS/FAIL, reported regardless):** covariance
tangent-space features. Per-session robust standardization of `Z` **inside** the certifier folds is achieved by
the certifier's own per-fold standardization; additionally we log whether `c×Z_pc` directions align with the
session covariate-drift axis (a red flag).

## 4. Method locks (UNCHANGED) and pinned hashes

- **Route B3:** `pc_centered_calibrated`, `calibration_version = p24d_cross_budget_alpha_spending_studentized_fixed_margin`;
  `h0:[Z,c]` vs `h1:[Z,c,c×Z_pc(rank 3)]`, centered ±½, logistic `C=0.5`; condition-matched fixed-margin `h0`
  bootstrap (`n_boot=200`, invalid counted); studentized subject gate `LCB_{0.975}(Δ_s)>0`; family `α=0.05`,
  budgets `{20,30}`, `α_budget=0.025`; `n_folds=3`; subject-grouped cross-fit; subject-condition class-balanced
  weights. **Pin BOTH** `sha256(paired_calibrated.py)=26e505ed…` AND
  `sha256(paired_conditional_test.py)=1263f672…` (red-team #8), plus `paired_certifier.py`.
- **Route A:** the frozen `dee8958`-line certifier (source-anchored 3-state), method + code hashes pinned in
  the A manifest (exact hashes recorded at build from the frozen A sources). No parameter changes.
- **α reconciliation (red-team minor #3):** each null cohort's certifier decision uses the frozen
  `α_budget=0.025`; the AGGREGATE false-confirmation bound target is the family `0.05`. Both levels are
  intentional and frozen, not a post-hoc relaxation.

## 5. Eligibility guards (contract)

Pair integrity ≥ 0.95; ≥ 8 trials per condition per class; per-condition class coverage; **≥ 20 eligible paired
subjects**; `Z`-dim ≥ 6 & `rank(Z) ≥ 3`; else `NEED_MORE_LABELS`/`INVALID_PAIR_STRUCTURE`.

## 6. The semi-synthetic injection bank on REAL features (ground-truth core)

For each cohort we take real `Z` (real covariate structure) and inject a **known** truth by relabeling/
perturbing, always with subject-grouped cross-fit and no leakage. Truth is known by construction, so both
false-confirmation and power are measurable. Kinds:

| kind | construction on real `Z` | truth | role |
|---|---|---|---|
| **NULL_cov** (GATING) | real session-1/2 split (real covariate drift) + `Y* ~ p̂(Y\|Z)` from a SINGLE pooled boundary fit on both sessions | no concept, real covariate shift | **primary type-I** (fixes blocker #1) |
| NULL_exch | within-session stratified random split (mean/balance-matched) | no concept, no covariate | calibration check only (demoted N1; log per-cohort \|meanZ_A−meanZ_B\|) |
| POS_concept | real session split + inject an antisymmetric boundary rotation into one condition | concept present | **power** for B3 (and A) |
| POS_concept+cov | rotation on top of the real covariate drift | concept + covariate | power under confound |
| LABEL | shift `P(Y)` prior across conditions, `P(Z\|Y)` fixed | label shift (must abstain) | trap control |
| PURE_COND | invisible relabel (boundary change, marginal fixed) | pure-conditional (Z-only unidentifiable) | secondary, weak-by-theory |

The synthetic bank is additionally strengthened with **scale/heteroscedastic covariate nulls** (translate AND
rescale/rotate within the covariate subspace) before the real run, since the existing synthetic
`paired_covariate` is a rigid mean-offset only and already shows elevation (red-team major #3).

## 7. Genuine real contrast (DESCRIPTIVE, non-gating)

One certifier run on the real session-1-vs-2 cohort → verdict + statistic `T` + studentized `Z_subj` + LCB +
the per-subject `Δ_s` distribution. Report the **fraction of subjects with `Δ_s>0`** and a **subject-bootstrap
CI** on mean `Δ_s` (a within-cohort rate over the 54 exchangeable units; red-team major #2). **Descriptive
only:** because the constructed nulls of §6 do not fully reproduce the genuine between-session covariate regime,
a real `CONCEPT_CONFIRMED` is reported as a verdict, **not** interpreted as validated concept drift; a
non-confirmation is **not** evidence of absence (unmeasurable power). (Red-team majors #4, #5.)

## 8. Endpoints / PASS criteria (validity, not truth-detection)

Gating (conjunction), evaluated for **each route** on the §6 bank:
- **R1 (primary type-I):** on **NULL_cov**, subject-clustered/bootstrap upper bound on the false-confirmation
  rate ≤ family `0.05` (resample SUBJECTS, not label-draws; the 54 subjects are the unit — red-team blocker #3).
  Denominator = **valid** cohorts only; invalid/abstain fraction reported separately and capped at ≤ 20%
  (family non-estimable above the cap; abstains do NOT pad the type-I denominator — red-team major #1).
  Also report NULL_exch (calibration) alongside.
- **R2 (power, ground-truth-restored):** on **POS_concept** (and POS_concept+cov), report B3's confirmation
  rate with a subject-bootstrap lower bound; and the A/B3 CONTRAST — the study's core transfer test is
  **B3 powered where Route A is not**, both measured on the SAME real-feature injected bank. (Descriptive-plus:
  a pre-registered success target may be set, e.g. B3 CP-lower ≥ some bar on POS_concept, if the reviewer wants
  R2 gating; default: R2 reported, the A-vs-B3 gap is the headline.)
- **R3 (guards):** eligibility holds (§5).
- **R4 (no silent failure):** every state in the valid 5-state set; sampler/bootstrap invalid fraction below
  the pre-registered cap (NOT "exactly 0" — real EEG will occasionally degenerate; red-team major #1).
- **R5 (stability, gating for type-I):** R1 (NULL_cov type-I control) **replicates** under subject-bootstrap /
  leave-k-subjects-out and, **gating-if-feasible**, on a disjoint dataset (`BNCI2014_004`/2b using
  tangent-space `Z` so `rank≥3` holds, or a disjoint Lee2019 subject-half). A control that does not replicate
  is not a control.
- **R6 (red-team):** independent re-aggregation reproduces **R1–R5** without correction (gating criteria only;
  R2/R7 descriptive reproduction is reported, never changes PASS/FAIL — red-team major #6).

Non-gating / descriptive: **R7** genuine-contrast report (§7); LABEL/PURE_COND bank results; NULL_exch;
the within-session contiguous-drift probe (reported CP, red-team minor #1).

**Interpretation.** A PASS = *on real EEG features with injected known truth, the frozen certifiers control
false confirmation (incl. under real covariate drift) and B3 is powered where A is not.* That directly tests
whether the synthetic A-negative / B3-positive conclusions transfer to real EEG statistics. A FAIL on R1 for
either route is an honest **negative** (type-I does not transfer under this feature pipeline) — reported, not
retried (§9). It does NOT claim any genuine real verdict is correct (§1).

## 9. Freeze discipline, stopping rule, anti-gaming

- **Single frozen primary** feature/montage (no `d` search, no feature-family selection after unblinding); the
  secondary (tangent-space) is robustness-only and **cannot** change PASS/FAIL and is reported regardless
  (red-team minors #4).
- **Stopping rule:** a FAIL on R1 for the frozen PRIMARY is the reported result; it is **not** grounds to swap
  to the secondary and re-judge. Any later feature is a NEW pre-registration disclosing the original FAIL
  (red-team minor #5).
- No optional stopping, no threshold search; feature, montage, α, bank, bootstrap `B`, criteria all frozen in
  the manifest before the run. Pin scipy in the run env so the exact Student-t LCB path executes; record which
  path ran; add a subject-bootstrap LCB robustness cross-check (red-team minor #2).

## 10. Freeze package plan (built only AFTER design approval + §11 sign-off)

- **Isolated code** in `csc/mininfo/` (no `cmi` import): one-time `build_lee2019_b3_cache.py` runs in **eeg2025**
  (moabb) → `LEE2019_B3.npz` (`Z, subject, session, y, classes, channel_list, exact counts, provenance`); the
  certifier/runner then read only that cache (env `icml`, CPU/SLURM), preserving isolation + reproducibility.
- **Two manifests** (B3 + A): data source + exact counts + frozen feature/montage + bank spec + injection
  seeds + criteria R1–R6 + pinned code hashes (both certifier files each) + disjoint seed base (> all synthetic
  ranges, verified).
- **Runners** (`run_b3_realeeg.py`, `run_a_realeeg.py`): dry-run + guarded `--execute`, fail-closed provenance
  (manifest hash, code hashes, HEAD==tag, clean tree, seed disjointness), conservative denominators, SLURM
  wrapper. **Tag:** `csc-realeeg-v1` (distinct). Tests + independent audit of the package **before** any run.

## 11. OPEN decisions to FREEZE before build (reviewer sign-off)

1. **Primary feature (single):** log-band-power (`normalize=None`) on a frozen sensorimotor channel subset
   [proposed] vs covariance tangent-space. Confirm the exact channel list.
2. **Scope:** both Route A and Route B3 [proposed], or B3 first then A.
3. **R2 gating?** A-vs-B3 power gap reported [proposed] vs a pre-set B3 power bar as an additional gate.
4. **R5 second dataset:** 2b (tangent-space) gating-if-feasible [proposed] vs Lee2019 subject-half only.
5. **Bootstrap `B`, invalid-fraction cap (≤20% proposed), α levels (0.025 per-cohort / 0.05 family) — confirm.**
6. Tag name `csc-realeeg-v1`; cache-build env `eeg2025`.

## 12. Out of scope / NOT authorized

Building anything, downloading/touching data, creating a tag, or running — all require design approval + §11
sign-off + a separate run go. No clinical/PD claim (MI cross-session, not medication; the cached PD data is
resting-state with `Y=med-state` and does not fit a within-condition B3 test). No change to either method lock.
No touching the frozen synthetic tags.
