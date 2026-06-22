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

**C2 deployment uses `q⁺ = max(q, 0)` (Amendment 2):** `U_a = μ̂_a + q⁺·max(σ̂_a,σ_min,d,a)`. A negative standardized
quantile would make larger uncertainty *lower* the bound (uncertainty inversion → preferring high-scale actions);
clamping `q` at 0 only raises-or-keeps each `U_a`, so it **cannot reduce coverage** while removing the inversion.
**C1 and C3 use the raw additive `q` with NO clamp (frozen, Amendment 3)** — their `q` enters additively (not as a
multiplier on a scale), so there is no uncertainty inversion to fix. Only C2 clamps. The manifest records `q_raw` and
`q_used`.

## S2. DEV scale/quantile-calibration gate — CANDIDATE-SPECIFIC, subject-balanced, disease×action (Amendment 1)

A first-class **prerequisite** (a failing candidate is ineligible regardless of width/utility). Computed on
**out-of-fold held-out DEV subjects** (not the final external CAL), **per disease × action**, **subject-equal-weight**
(each subject's batches sum to weight 1). These are efficiency/admissibility diagnostics — NOT additional conformal
validity assumptions (validity holds from FIT-frozen scores + exchangeability regardless).

**C2 (mean/scale):** with `r_{sBa}=(ΔR−μ̂)/max(σ̂,σ_min,d,a)` —
- variance ∈ `[0.5, 2.0]`; `|mean| ≤ 0.25`; positive-tail 90th percentile ∈ `[0.8, 2.0]·z₀.₉₀` (z₀.₉₀≈1.28).

**C3 (additive CQR):** NO variance≈1 / Gaussian condition. Instead —
- OOF `q̂₀.₉₀` per-action **exceedance rate** `P(ΔR>q̂₀.₉₀)` ∈ `[0.05, 0.20]` (nominal 0.10);
- positive-excess `max(ΔR−q̂₀.₉₀,0)` tail finite: 95th percentile **≤ 2.0 × (OOF ΔR SD per disease×action)**
  (threshold **pinned pre-DEV** in the DEV_DESIGN_LOCK — Amendment 2; no post-DEV fill);
- zero quantile crossing; all predicted gaps `q̂₀.₉₀−q̂₀.₅₀` finite and positive.

**`max_a` dominance — applies to EVERY selectable candidate (C1, C2, C3); Amendment 2:** with the candidate's own
nonconformity (`r_{sBa}=(ΔR−μ̂)/max(σ̂,σ_min,d,a)` for C2; `[ΔR−q̂₀.₉₀]_{Ba}` for C3; **raw `ΔR−μ̂` for C1**):
`M_{s,a}=max_{B∈𝓑(s)} (score)_{sBa}`; `T_s=argmax_a M_{s,a}` (set); `share_a=(1/N)Σ_s 1[a∈T_s]/|T_s|`; require
`max_a share_a ≤ 0.60` with **fractional tie credit** (action-order invariant). **C1 is selectable only if it passes
this raw-residual dominance gate** (else C1 is ablation-only and cannot be SELECTed). **C0 is comparator-only** (its
raw-residual dominance is reported as a diagnostic, never selectable).

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

## S5. DEV split-as-ONE-algorithm + train-once / refit / serialize / hash (Amendment 1 + 2)

**Models are disease-specific (Amendment 2):** train **separate** PD and SCZ predictors (a chosen candidate ID is
global, but PD and SCZ have their own weights, target normalizer, and `σ_min,d,a`). The pipeline below runs **within
each disease** on that disease's pooled DEV subjects.

**Pinned DeepSets/training hyperparameters (Amendment 3/4; `acar/v3/predictors.HP`, frozen pre-DEV):** shared ψ =
2-layer MLP (hidden 64, ReLU) per window over `concat(values, mask)` [2F]; pooling = **mean ⊕ std**
(permutation-invariant); context = `concat(context_values, context_mask)` [2C]; shared ρ = 2-layer MLP (hidden 64);
**per-action `ModuleDict` heads — NO action embedding** (Amendment 4: heads keyed by NON_IDENTITY; head = Linear→
{C1: μ; C2: μ, softplus v→σ; C3: q̂₅₀, softplus gap}); dropout 0; Adam lr 1e-3, weight_decay 1e-4, grad-clip max-norm
1.0; max_epochs 200, patience 20, min_delta 1e-4; **target SD floor 1e-3, input-feature SD floor 1e-6** (Amendment 4);
β-NLL β=0.5 with weight `v.detach()**β`; ε=1e-6; seeds (outer 0, fit/cal 1, early-stop 2); K=5 folds; FIT_FRAC 0.70;
TRAIN_FRAC 0.80; CPU + 1 thread + `use_deterministic_algorithms`. **Normalizers fit on TRAIN only (VAL uses TRAIN
stats).** Per-fold C2 `σ_min` = `Q₀.₀₅` of raw σ̂ over the **full fold FIT = TRAIN∪VAL** after best-state restoration;
the **final** artifact uses the **OOF** `σ_min`. **Final all-DEV refit epoch = `round_half_up(median_k(best_epoch_k +
1))`** (best_epoch is 0-based ⇒ +1), fixed in advance — no new validation at refit.

**The DEV OOF estimator is one unique algorithm (Amendment 2):**
1. Partition the disease's subjects into **K pre-declared outer subject-disjoint folds** (seed `S_outer`). Each fold
   in turn is **EVAL**.
2. The non-EVAL subjects are hash-split (seed `S_fitcal`, ratio `FIT_FRAC=0.70`) into **FIT** and **CAL** (subject-
   disjoint).
3. FIT is further hash-split (seed `S_es`, ratio 0.80) into **TRAIN/VAL**, used **only** for early stopping +
   FIT-only normalization (never thresholds).
4. The **predictor sees FIT only**; the conformal **`q` sees CAL only**; **S2/S4 diagnostics aggregate on outer
   EVAL** (out-of-fold). Fallback `<MIN_BATCH` batches are **retained and routed to identity** (included in EVAL loss
   accounting, excluded from FIT/CAL fitting). Three seeds `(S_outer,S_fitcal,S_es)` + `K` + the ratios are frozen.
5. Select the single candidate via S2 + S4.
6. **Refit the selected candidate once on all PRE-SPECIFIED ELIGIBLE DEV subjects of that disease — eligibility is
   by the frozen split/inclusion rule ONLY; subjects are NEVER excluded based on residuals, scale diagnostics, or
   candidate performance** (CPU, fixed seed, `torch.use_deterministic_algorithms(True)`).
7. `σ_min,d,a` from the **OOF** scale predictions (S3), not the final model's in-sample σ̂.
8. Serialize + **full 64-char SHA-256** of each disease's predictor weights, normalizer, action vocabulary, source
   state (two weights hashes: `predictor_weights_sha256.PD`, `.SCZ`).
9. **v2 replay** = the **full v2 recipe** (HGB + Ridge/constant fallback exactly as in code) refit once per disease on
   the identical eligible DEV pool, run under the identical Arm-B protocol (S6).
- The **double-run** re-runs only deployment + site-local CAL on the **loaded frozen** predictors (no retraining);
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

**Harmful adapted-batch test (single executable statistic, Amendment 2):** for each subject, restrict to the **batches
the ROUTER actually adapted (chose non-identity)**; on exactly those batches compute the harmful rate of (a) the
router's chosen action and (b) the frozen **best-fixed** action: `rate = #(ΔR>0)/#(those batches)`. Subjects with
**zero** router-adapted batches are **excluded and counted/reported**. Paired statistic (H1: router rate < best-fixed)
on the per-subject differences `d` — **tie-aware (Amendment 3, implemented in `conformal.harmful_rate_test`):** drop
zero differences (recorded); require **≥10 nonzero** pairs else `NOT_EVALUABLE` (→ G2 fail for that site); **exact
Wilcoxon** only when the remaining `|d|` are **all distinct and n<25** (where SciPy's exact null is valid), **else a
deterministic sign-flip permutation test** (fixed `seed=0`, `n_perm=20000`); all-zero ⇒ `NOT_EVALUABLE`. Pin SciPy
version + continuity convention in the env lock. **Holm across sites**; **PASS = Holm-adjusted p < 0.05** — the single
condition; there is **no** secondary "≤ with no site worse" alternative.

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

For Arm B the supervised state is **frozen from DEV**. Label access at an external site is split (Amendment 2):
external **CAL labels** are used **only** to compute the site-local conformal `q`; external **EVAL labels** are
**invisible to the entire deployment path** (predictor, features, `q`, `U`, routing) and are read **once**, after all
of the above are frozen, solely for the one-shot endpoint scoring (G1/G2/coverage). (Earlier wording "external labels
may compute only q" is corrected — it must not forbid computing G1/G2 from EVAL labels at scoring time.)
Additionally:
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

`<<TBD-after-DEV>>` selected predictor (C1/C2/C3) + per-disease weights hashes; realized SELECT; frozen `σ_min,d,a`;
frozen best-fixed per disease. (The C3 positive-excess tail threshold is **no longer TBD** — pinned pre-DEV at
2.0×OOF ΔR SD, S2.) `<<TBD-after-audit>>` retained sites per disease (≤2, with acquisition-unit strata) +
DOIs/versions + split seed + per-site/unit CAL/EVAL counts. `<<TBD-before-freeze>>` binding-run command + output path.

## S12. Module layout (design intent; implemented under DEV_DESIGN_LOCK)

New isolated `acar/v3/`: `set_features.py` (per-window paired tensor + **availability masks** + batch context),
`predictors.py` (C1/C2/C3 with a unified candidate-specific **`upper_bound()`** interface replacing v2's hardcoded
`reg.predict()+q`), `conformal.py` (subject joint score per candidate), `develop.py` (DEV bake-off + S2/S4 gates).
v2 router code is left untouched.

## S13. Set-contract canon (Amendment 2 — IMPLEMENTED + tested in `acar/v3/set_features.py`, 685a526)

Frozen DEV-design rules now enforced in code (15 synthetic guards pass):
- **`WindowActionSet`** = `values[n,F] + availability_mask[n,F]{0,1} + context_values[C] + context_mask[C] +
  action_name + action_index + window_keys`. **Validated + immutable** (`__post_init__`: shapes, binary masks,
  masked-slots-exactly-0, finiteness, `action_name∈NON_IDENTITY`, `action_index==ACTION_VOCAB.index(action_name)`,
  unique non-empty keys; arrays read-only). Missing-zero (mask 0) is **distinct** from genuine-zero (mask 1).
- **Canonical row order BEFORE adapters** — `(z, keys)` sorted by `canon_key` first ⇒ permutation invariance is
  byte-identical at the path level (tested via `np.array_equal`), not a hash tolerance.
- **`canonical_digest` = full 64-char SHA-256** over schema header (incl. `SCHEMA_VERSION`, action, ACTION_VOCAB,
  feature lists, shapes) + raw float64-LE values/context + uint8 masks + canonical keys. No rounding (single-ULP
  sensitive).
- **Canonical action execution order** (ACTION_VOCAB, never caller order); selection validated (unknown/dup/identity/
  empty rejected).
- **Action capability map** `{matched_coral:geom, spdim:geom, t3a:no-geom}` asserted vs adapter output (drift guard);
  T3A geometry features masked unavailable.
- **Probability/shape validation** of `p0,pa,z0,za`; NaN/Inf rejected; `<MIN_BATCH` short-circuits to identity with
  **no** adapter call (guard monkeypatches `apply_action`).
- **Structured `WindowKey`** `(dataset_id, subject_id, recording_id, window_index)` with **disambiguated** canonical
  serialization (WK structured-JSON / S string; non-key → TypeError); the real v3 loader must emit `WindowKey` (the
  v2 `Batch` lacks per-row window identity — to be added in the v3 data layer). Does **not** call v2 `feature_vector()`.
- **Object-level immutability (Amendment 3):** `WindowActionSet` is `@dataclass(frozen, slots)` → field rebind raises
  `FrozenInstanceError` (not just read-only buffers). `<MIN_BATCH` returns an immutable **`FallbackBatchRecord`**
  (forced_identity, reason, window_keys, `canonical_input_digest` full-64, n_windows) — no adapter called. The
  **identity reference is computed exactly once per batch** (call-count guard).
- **Predictors/training/conformal/data implemented (Amendments 3+4):** `predictors.py` (per-action `ModuleDict`
  heads; validated `CandidatePrediction`; **immutable `FittedCandidateArtifact` storing canonical parameter BYTES**,
  rebuilt via `build_net`, with `verify_integrity()` (no-rounding floor hash; candidate/arch cross-check) and
  `assert_disease`); `training.py` (`fit_candidate_earlystop` + `refit_candidate_fixed_epochs` + `final_epochs`;
  exact Huber/β-NLL(stop-grad)/pinball; subject-balanced; TRAIN/VAL subject-disjoint; deterministic fail-closed);
  `normalizers.py` (FIT-only mask-aware input + target, floors 1e-6/1e-3); `conformal.py` (fail-closed everywhere;
  1-D CAL-score shape; route full-set/canonical/q∈finite∪{+∞}/δ≥0; ONE Wilcoxon harmful estimand);
  `data.py` (`SubjectKey`/`RecordingKey`/`WindowKey`; `DeploymentBatch`(no y, fallback⇔n<MIN_BATCH, 1≤n≤B, 64-hex
  source) vs `LabeledRiskRecord`). **41→ all v3 synthetic guards pass** (set-feature/data/training/predictor-conformal).
  **Still NOT tagged** `acar-v3-dev-design-v1` — pending the real v3 loader (building `DeploymentBatch` from dumps),
  `develop.py` (S5 split orchestration + S2/S4 + C0/v2 replay), env lock, and a full green re-run; no DEV cohort read.
