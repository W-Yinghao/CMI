# ACAR v3 — FREEZE SKELETON (NON-BINDING) — incorporates Amendment 1 (2026-06-22)

**Status:** `DRAFT / NON-BINDING / NOT ACAR_FROZEN_v3.md / NO LOCKBOX ENDPOINT ACCESSED`
**Date:** 2026-06-22 (Amendment 1 folded in; see `notes/ACAR_V3_AMENDMENT_1.md` for the changelog)
**Parent:** ACAR v2 `MEASUREMENT_ONLY` (`acar-v2-protocol` @ `9b2f0c1`; result `1528a94`; audit `6a0c3d0`/`ce5c330`)
**Companion:** `notes/ACAR_V3_DESIGN_DRAFT.md` (rationale; its V3.4–V3.6 are superseded by this file).

This skeleton pre-commits the protocol RULES. It is **not** a freeze. Two distinct locks gate execution (§S0). It
leaves `<<TBD-after-DEV>>` / `<<TBD-after-audit>>` only for values that must come from the DEV gate or the
metadata-only audit. Operating point stays **α=0.10, δ=0**, action set unchanged — v3 does **not** chase G2 by
loosening the operating point.

---

## S0. Two-phase lock + taxonomy (Amendment 1)

- **`DEV_DESIGN_LOCK`** — fixed (separate commit + light tag `acar-v3-dev-design-v1`) **before the first DEV numerical
  run**: every candidate formula (S1), all architecture/optimizer/training constants (S5), folds/seeds/early-stop
  (S5), all S2/S4 thresholds, the SELECT scalar + tie tolerance + deterministic tie order, the width definition (S6),
  best-fixed rule (S6), candidate failure/NaN handling, and the final-refit rule (S5). It does **not** authorize any
  lockbox access. After this lock the DEV bake-off is run; its thresholds/definitions are **not** retroactively
  editable.
- **`EXTERNAL_PROTOCOL_FREEZE`** — only **after** the DEV gate passes **and** the metadata audit completes: fill the
  selected model + weights hash, retained sites + split seed, and the external G2 numbers into the real
  `ACAR_FROZEN_v3.md`, commit + tag on a clean `acar` worktree, then run the single binding Arm-B evaluation once.
- **Taxonomy:** `PROCEED_SAFE_ROUTER` · `UTILITY_ONLY` (G2 pass, coverage diagnostic fail) · `MEASUREMENT_ONLY`
  (G1 pass, G2 fail) · `TERMINATE` (held-out **G1** fail) · `RUN_QUARANTINED/PROTOCOL_INVALID`. **DEV-stage**
  no-candidate-passes is **`DEV_STOP / NO_LOCKBOX_CONSUMED`** — NOT `TERMINATE` (which is reserved for held-out G1).

## S1. Candidate predictors + exact training losses (Amendment 1: C3 = additive one-sided CQR)

Target per batch `B`, action `a∈{matched_coral,spdim,t3a}` rel. `identity`: `ΔR_a(B)=R_B(f_a)−R_B(f_0)` (NLL).
All preprocessing/normalization/architecture/clipping fit on **FIT only**; no label enters feature
construction/normalization/selection/deployment. **C0 is comparator-only** (never selected; baseline for §S6).

| id | predictor | center `m_c` | training loss (frozen) | nonconformity score |
|----|-----------|--------------|------------------------|---------------------|
| C0 | v2 batch-summary HGB (comparator only) | μ̂ | full v2 recipe (HGB; Ridge/constant fallback) as in `9b2f0c1` | raw `ΔR−μ̂` |
| C1 | DeepSets mean-only | μ̂ | Huber, `δ=1.0` in **standardized-target units** | raw `ΔR−μ̂` |
| C2 | DeepSets mean+scale | μ̂ | β-NLL (β=0.5), exact form §S3 | standardized `(ΔR−μ̂)/max(σ̂,σ_min,d,a)` |
| C3 | DeepSets additive one-sided CQR | q̂₀.₅₀ | pinball `L=½ρ₀.₅+½ρ₀.₉` | additive `ΔR − q̂₀.₉₀` |

**C3 monotone parameterization (no crossing, no scale head, no `w_min`):**
`q̂₀.₉₀,a(B) = q̂₀.₅₀,a(B) + softplus(d_a(B)) + ε`. Only `q̂₀.₉₀ − q̂₀.₅₀` is required positive; `q̂₀.₅₀` may be
negative (median incremental risk can be < 0). `q̂₀.₅₀` is the C3 point predictor (G1) and width center (S6).
Conformal: `S_s = max_{B∈𝓑(s)} max_a [ΔR_a(B) − q̂₀.₉₀,a(B)]`, `U_a(B) = q̂₀.₉₀,a(B) + q`. **C3 has no `w_min`** —
the additive CQR keeps finite-sample coverage + heteroscedastic adaptivity without re-introducing C2's scale head.

## S2. DEV scale/quantile-calibration gate — CANDIDATE-SPECIFIC, subject-balanced, disease×action (Amendment 1)

A first-class **prerequisite** (a failing candidate is ineligible regardless of width/utility). Computed on
**out-of-fold held-out DEV subjects** (not the final external CAL), **per disease × action**, **subject-equal-weight**
(each subject's batches sum to weight 1). These are efficiency/admissibility diagnostics — NOT additional conformal
validity assumptions (validity holds from FIT-frozen scores + exchangeability regardless).

**C2 (mean/scale):** with `r_{sBa}=(ΔR−μ̂)/max(σ̂,σ_min,d,a)` —
- variance ∈ `[0.5, 2.0]`; `|mean| ≤ 0.25`; positive-tail 90th percentile ∈ `[0.8, 2.0]·z₀.₉₀` (z₀.₉₀≈1.28).

**C3 (additive CQR):** NO variance≈1 / Gaussian condition. Instead —
- OOF `q̂₀.₉₀` per-action **exceedance rate** `P(ΔR>q̂₀.₉₀)` ∈ `[0.05, 0.20]` (nominal 0.10);
- positive-excess `max(ΔR−q̂₀.₉₀,0)` tail finite (95th percentile < `<<TBD-DEV-DESIGN>>` SD of OOF ΔR);
- zero quantile crossing; all predicted gaps `q̂₀.₉₀−q̂₀.₅₀` finite and positive.

**`max_a` dominance (both candidates), subject-level + action-order invariant, fractional ties:**
`M_{s,a}=max_{B∈𝓑(s)} r_{sBa}` (C2) or `=max_{B} [ΔR−q̂₀.₉₀]_{Ba}` (C3); `T_s = argmax_a M_{s,a}` (set);
`share_a = (1/N) Σ_s 1[a∈T_s]/|T_s|`. Require `max_a share_a ≤ 0.60`. Fractional tie credit (not "first max"), so the
gate is invariant to action ordering. (C0/C1 use the raw residual scaled by the global FIT residual SD for the
diagnostic only.)

## S3. FIT σ_min / β-NLL — exact (Amendment 1)

- **σ_min is per disease × action** (a disease-pooled floor could let one action stay chronically low-scale and
  re-capture `max_a`). For the final refit it is **derived from OOF scale predictions**, not in-sample:
  `σ_min,d,a = Q₀.₀₅{ σ̂^OOF_{sBa} : disease d, action a }`. (During the DEV bake-off, a per-fold FIT-only
  `Q₀.₀₅(σ̂^FIT)` is used; the final frozen value uses the OOF rule above.)
- **β-NLL (Seitzer 2022), exact:** `v = softplus(h_v) + ε`, `σ = √v`,
  `L_β = ½[ (y−μ)²/v + log v ] · stopgrad(v^β)`, `β = 0.5`, weight = `v.detach()**0.5` (variance detached).
  Frozen: `ε = 1e-6`; reduction = **subject-balanced mean** (each subject's batches weight-normalized to 1, then mean
  over subjects); target `y` = **per-disease standardized ΔR** (FIT mean/SD); `μ,h_v` heads operate in standardized
  units; Huber `δ=1.0` is in those standardized units; gradient clipping max-norm `1.0`; init = PyTorch default
  (seeded). C3 pinball uses the same subject-balanced reduction and standardized target.

## S4. Single DEV model-selection scalar (Amendment 1: among C1/C2/C3 only)

Among candidates passing **all** DEV pre-lock criteria (S2 gate + list below), select the one maximizing:
> **SELECT = disease-macro OOF router NLL reduction** (mean of PD and SCZ OOF `red_router`) at α=0.10, δ=0,
> tie-break (|Δ| ≤ `1e-4`) by **smaller** disease-macro OOF width `W_c` (S6); residual ties broken by fixed order
> C2 ≺ C3 ≺ C1.

DEV pre-lock criteria (PASS/FAIL): S2 calibration gate · ≥1 PD action OOF harm AUROC ≥0.60 · SCZ continuous metric
(S6 G1, subject-clustered MAE of `m_c` vs ΔR) **not worse** than C0 · disease-macro OOF width `W_c` **≥30% below C0**
· OOF adaptation coverage (α=0.10,δ=0) **≥15%** · OOF `red_router>0` and **not below** C0 · all guards (S8) pass.
**No passer ⇒ `DEV_STOP / NO_LOCKBOX_CONSUMED`** (no lockbox label read).

## S5. Train-once / refit / serialize / hash / no-retrain-on-double-run (Amendment 1: unique procedure)

The "DEV FIT pool" is made unique:
1. C1/C2/C3 produce OOF predictions over **pre-declared outer subject-disjoint folds** (seeded).
2. Each fold's **inner** split is used **only** for early stopping + FIT-only normalization (never for thresholds).
3. Select the single candidate via S2 + S4.
4. **Retrain the selected candidate once on ALL S2-admissible DEV subjects** (CPU, fixed seed,
   `torch.use_deterministic_algorithms(True)`).
5. `σ_min,d,a` from the OOF scale predictions (S3), **not** from the final model's in-sample σ̂.
6. Serialize + **full 64-char SHA-256** of predictor weights, normalizer, action vocabulary, source state.
7. **v2 replay** is the **full v2 recipe** (HGB with Ridge/constant fallback exactly as in code — not "one HGB")
   trained once on the identical final DEV pool, then run under the identical Arm-B protocol (S6).
- The **double-run** re-runs only deployment + site-local CAL on the **loaded frozen** predictor (no retraining);
  neural training is never on the binding-hash path.

## S6. Binding endpoints — Arm B only, exchangeable same-site subjects (Amendment 1)

Binding G2 + the finite-sample coverage theorem are **Arm B** (site-local conformal); Arm A (zero-shot) is
**descriptive** only. Per retained site: deterministic **random subject-hash** split → CAL (labels used ONLY for the
site-local `q`) + EVAL (label-free). Coverage claim = exchangeable **same-site** subjects + fixed batching `𝓑(S)`
(not "future", not cross-site).

**Width (candidate-comparable):** center `m_c` = μ̂ (C0/C1/C2) or q̂₀.₅₀ (C3);
`W_c = subject-macro mean over OOF EVAL of (U_a − m_c)` (subject-balanced). (Replaces the C3-undefined "U−μ".)

**best-fixed:** the action maximizing **DEV OOF** `red` (NLL reduction) per disease, selected on DEV only, frozen.

**v2-router replay:** C0 full recipe refit on the final DEV pool → identical Arm-B site-local CAL `q` → identical
EVAL subjects.

**G1 (per disease):** ≥1 action-specific held-out harm AUROC ≥0.60 (evaluable, per-site) · continuous prediction
improves over C0 by the frozen **subject-clustered MAE of `m_c` vs ΔR**.

**G2 (per disease, ALL):** `red_router>0` · `>red_bestfixed` · `>red_v2_router` (identical EVAL) · oracle retention
≥0.50 · adaptation coverage ≥0.20 · **harmful adapted-batch test** passes (below) · **two-site rule** (below).

**Harmful adapted-batch test (frozen):** denominator = router-adapted (non-identity) batches; per-subject harmful
rate `= #(adapted batch with ΔR_chosen>0)/#(adapted batches)`; compare router vs best-fixed by a **one-sided Wilcoxon
signed-rank across subjects**, `α=0.05`, **Holm across sites**; pass = router significantly lower (or ≤ with no site
worse). Subjects with zero adapted batches are excluded from that subject's rate (recorded).

**Two-site rule:** with both admissible sites per disease, **BOTH** must individually satisfy {`red_router>0`,
`>red_v2_router`, harmful-rate test}, **and** disease-macro pooled-EVAL retention ≥0.50 + adaptation coverage ≥0.20.
**Contingency:** one admissible site ⇒ single-site verdict labeled *site-specific (no within-disease replication)*;
zero ⇒ G2 not evaluated for that disease.

**Coverage diagnostic (wording frozen):** conditional on the realized site-local `q`, a **one-sided exact binomial
lower-tail undercoverage diagnostic** (H0: site EVAL subject-event coverage ≥ 1−α), **Holm across sites**, retaining
the theorem's explicit exchangeability assumption. It is a **diagnostic**, NOT an exact test of the marginal
split-conformal theorem (whose probability also integrates the random CAL quantile). **"Not rejecting" ≠ "proving
coverage."** Failure is recorded, never silently ignored.

## S7. External deployment substrate — frozen from DEV (Amendment 1, §二)

For Arm B the supervised state is **frozen from DEV**; external diagnosis labels may compute **only** `q`:
- **encoder, base classifier `f0`, source moments/readout, class prototypes, action state** = all DEV-frozen; no
  external label rebuilds any source state; nothing supervised is refit per external site.
- **raw→feature pipeline frozen + fully hashed:** channel mapping, reference, filtering, resampling, window
  length/stride, artifact handling, missing-channel policy, resting-condition selector, batch construction, encoder
  checkpoint, feature-dump format. This makes `f_0` and `f_a` (hence `ΔR_a`) uniquely determined on any external site.
- The external site's `(z_ev,y_ev)`-equivalent is **not** used to fit a new SourceState; the DEV-frozen SourceState
  is applied. (This tightens v2's `data.py` substrate, which fit a per-cohort SourceState — acceptable for v2 DEV but
  **disallowed** for v3 external sites.)

## S8. Hard guards (v2 set + v3 additions)

All v2 guards PLUS: set-permutation invariance of `μ̂,σ̂/q̂,U,action`; action-order invariance; FIT-only normalization
**and** σ_min derivation; strict positivity+finiteness of σ̂ / predicted gaps; **value+availability mask** for
geometry-unavailable windows (NEVER NaN→0 collapse — structural-missing must be distinguishable from a true zero);
CAL-label changes affect EVAL **only** through the calibrated standardized/CQR quantile; EVAL-label permutation leaves
`μ̂,σ̂/q̂,q,U,actions` bit-identical; serialization round-trip of the set encoder + all preprocessing state;
record-level hash includes per-window paired inputs (or canonical digest) + masks, `μ̂,σ̂/q̂₅₀,q̂₉₀`, subject score,
`q`, `U`, chosen action, split assignments, `predictor_weights_sha256`.

## S9. Frozen manifest schema (full provenance)

Distinct units per disease/site/fold: `n_fit_subjects,n_fit_batches,n_cal_subjects,n_cal_batches,n_eval_subjects,
n_eval_batches`. **Full 64-char SHA-256** for every raw/derived dump (no truncation); `dataset_version_doi`;
`subject_list_sha256` (FIT/CAL/EVAL); `split_assignments`; `source_state_sha256`; `predictor_weights_sha256`;
`raw_pipeline_sha256`; `protocol_commit`; `immutable_tag`; `environment_lock`; `double_run_hash`; per-site/fold
`m=n_cal_subjects`, `k=⌈(m+1)(1−α)⌉`, `q`, `σ_min,d,a`, coverage, Holm p.

## S10. Hard metadata gates (lockbox admissibility)

Admissible iff ALL pass (metadata-only, pre-endpoint): (1) **usable HC-vs-Patient binary label** matching DEV
(HC=0/Patient=1) — a site without a confirmed usable contrast is excluded; (2) **≥30 CAL subjects** after the
subject-hash split, per disease; (3) license permits research use + derived-stat redistribution; raw signal
available; 10–20-compatible montage; resamplable Fs; pre-specified **resting** condition; subject IDs present; **no
subject/recording overlap** with the seven DEV cohorts; preprocessing compatible. **Site definition:** if a dataset
spans multiple physical acquisition units/devices, the **acquisition unit/device is the calibration stratum**; each
unit must independently meet CAL feasibility, or the coverage claim is for the pooled mixture (not any single site).
Selection among admissible sites uses **only** these metadata criteria — never any ACAR outcome.

## S11. `<<TBD>>` to fill only after DEV gate + audit (then → ACAR_FROZEN_v3.md)

`<<TBD-after-DEV>>` selected predictor (C1/C2/C3) + weights hash; realized SELECT; frozen `σ_min,d,a`; frozen
best-fixed per disease; the C3 positive-excess tail threshold. `<<TBD-after-audit>>` retained sites per disease (≤2,
with acquisition-unit strata) + DOIs/versions + split seed + per-site/unit CAL/EVAL counts. `<<TBD-before-freeze>>`
binding-run command + output path.

## S12. Module layout (design intent; implemented under DEV_DESIGN_LOCK)

New isolated `acar/v3/`: `set_features.py` (per-window paired tensor + **availability masks** + batch context; reuses
`score_actions()` which already returns `p` and `ztil`), `predictors.py` (C1/C2/C3 with a unified candidate-specific
**`upper_bound()`** interface replacing v2's hardcoded `reg.predict()+q`), `conformal.py` (subject joint score per
candidate), `develop.py` (DEV bake-off + S2/S4 gates). v2 router code is left untouched.
