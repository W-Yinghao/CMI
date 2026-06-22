# ACAR v3 — FREEZE SKELETON (NON-BINDING)

**Status:** `DRAFT / NON-BINDING / NOT ACAR_FROZEN_v3.md / NO LOCKBOX ENDPOINT ACCESSED`
**Date:** 2026-06-22
**Parent:** ACAR v2 `MEASUREMENT_ONLY` (`acar-v2-protocol` @ `9b2f0c1`; result `1528a94`; audit `6a0c3d0`/`ce5c330`)
**Companion:** `notes/ACAR_V3_DESIGN_DRAFT.md` (scientific rationale).

This skeleton **pre-commits the protocol RULES** (the review resolutions) **before** any lockbox is chosen and
**before** the DEV gate is run. It is **not** a freeze: it deliberately leaves `<<TBD-after-DEV>>` /
`<<TBD-after-audit>>` placeholders for the items that may only be filled from (a) the DEV development gate and
(b) the metadata-only lockbox audit. Only when those are complete are the selected model/sites/splits/thresholds
copied verbatim into `ACAR_FROZEN_v3.md`, committed and tagged on the `acar` lineage on a clean worktree, and the
single binding external evaluation run once. Operating point stays **α=0.10, δ=0**, action set unchanged — v3 does
not chase G2 by loosening the operating point.

---

## S1. Candidate predictors + pre-specified training losses (resolution 1)

All candidates predict, per batch `B` and action `a∈{matched_coral,spdim,t3a}` (relative to `identity`), a one-sided
upper-bound input for `ΔR_a(B)=R_B(f_a)−R_B(f_0)`. All preprocessing, normalization, architecture, and clipping
constants are fit on **FIT only**; no label touches feature construction/normalization/selection/deployment.

| id | predictor | training loss (frozen) | conformal nonconformity |
|----|-----------|------------------------|--------------------------|
| C0 | v2 batch-summary HGB (baseline) | v2 HGB (squared error), as in `9b2f0c1` | raw joint residual `ΔR−μ̂` |
| C1 | DeepSets mean-only | **Huber** on `ΔR` (δ_Huber = 1.0, frozen) | raw joint residual `ΔR−μ̂` |
| C2 | DeepSets mean+scale (heteroscedastic) | **β-NLL** Gaussian (Seitzer 2022; β=0.5, frozen) | standardized `(ΔR−μ̂)/max(σ̂,σ_min)` |
| C3 | DeepSets CQR (upper-quantile) | **pinball** at quantiles {0.5, 1−α=0.90} (frozen) | one-sided CQR `ΔR − q̂_{0.90}` |

Notes: C3 (conformalized quantile regression) is added because Gaussian μ/σ heteroscedastic NLL is the least stable
fit at this data size; CQR predicts the upper conditional quantile directly and is the robustness alternative to C2.
σ̂/quantile heads use a softplus (C2) or monotone head (C3) to guarantee strict positivity (S8).

## S2. Per-action OOF scale-calibration DEV gate (resolution 2) — FIRST-CLASS, PREREQUISITE

A candidate is **calibration-admissible** only if, on **out-of-fold** DEV (nested subject-disjoint CV), for **every**
action `a` independently:

1. **Scale calibration:** the per-action standardized residual `r_{a}=(ΔR_a−μ̂_a)/max(σ̂_a,σ_min)` (C2) /
   CQR-pivoted residual (C3) has out-of-fold **variance ∈ [0.5, 2.0]** and **|mean| ≤ 0.25** (no systematic
   mis-scaling). C0/C1 use the raw residual scaled by the global FIT residual SD for this diagnostic.
2. **Tail diagnostic:** the per-action standardized residual **90th-percentile ∈ [0.8, 2.0]·z_0.90** (z_0.90≈1.28),
   i.e. tails neither collapsed nor exploded.
3. **No chronic `max_a` domination:** across CAL subjects, the fraction of subject scores `S_s` whose `argmax` over
   actions is a **single** action must be **≤ 0.60** (no one chronically under-scaled action driving the joint max).
   Report the per-action `argmax`-share vector.

A candidate failing the calibration gate is **ineligible** regardless of width/utility — this is the crux: it
prevents v3 from reproducing v2's "one high-variance action dominates the joint residual" failure in standardized
space. If **no** candidate is calibration-admissible, v3 closes at DEV (`TERMINATE` at development, no lockbox read).

## S3. FIT-only σ_min / width-floor rule (resolution 3) — fixed in advance

- **C2:** `σ_min = 5th percentile of the FIT-set σ̂ predictions` (computed by cross-fitting within FIT; never CAL/
  EVAL). Frozen scalar per disease, recorded in the manifest.
- **C3:** width-floor `w_min = 5th percentile of FIT predicted half-width (q̂_0.90 − q̂_0.50)`; the conformal padding
  uses `max(q̂_0.90 − q̂_0.50, w_min)` analogously.
- **C0/C1:** no scale floor (raw residual).
σ_min/w_min are derived once on FIT and **frozen before CAL**; they are not tunable against coverage/width.

## S4. Single DEV model-selection scalar (resolution 4)

Among candidates that pass **all** DEV pre-lock criteria (S2 calibration gate + the V3.4 list below), select the
**one** model maximizing a single scalar:

> **SELECT = disease-macro out-of-fold router NLL reduction** at the frozen `α=0.10, δ=0` (mean of PD and SCZ OOF
> `red_router`), tie-broken by smaller disease-macro OOF bound padding `mean_{a,B}[U_a−μ̂_a]`.

DEV pre-lock criteria (PASS/FAIL filters; from V3.4, with S2 added):
- S2 per-action calibration gate passes.
- ≥1 PD action risk-regressor harm AUROC ≥ 0.60 (OOF).
- SCZ risk regression **not worse** than v2 C0 on the frozen continuous metric (S6 G1 metric), OOF.
- disease-macro OOF bound padding falls **≥ 30%** vs C0 (measured out-of-fold).
- OOF adaptation coverage at α=0.10, δ=0 is **≥ 15%**.
- OOF `red_router > 0` and **not below** the C0 (frozen-v2) OOF router.
- all guards (S8) pass on synthetic + DEV.

If no candidate passes, v3 closes at DEV without reading any lockbox label.

## S5. Train-once / serialize / hash / no-retrain-on-double-run (resolution 5)

- The selected predictor is **trained exactly once** on the DEV FIT pool: **CPU**, fixed seed,
  `torch.use_deterministic_algorithms(True)`, then **serialized** and **SHA-256 hashed** (full 64 chars) into the
  manifest as `predictor_weights_sha256`.
- The **double-run determinism check re-runs only deployment + site-local CAL** on the **loaded frozen predictor**
  (no retraining). It must reproduce the record-level canonical hash bit-identically. Neural training is never on the
  double-run path, so training-stage nondeterminism cannot affect the binding hash.

## S6. Binding endpoints — bound to **Arm B**, exchangeable same-site subjects (resolutions 6, 7)

The binding G2 and the finite-sample coverage theorem are **Arm B (site-local conformal safety)** only. Arm A
(zero-shot external) is reported **descriptively** (no finite-sample theorem). For each retained held-out site,
deterministic subject-hash split → CAL subset (labels used ONLY for the site-local conformal quantile) and EVAL
subset (label-free deployment). The split is **random over subjects** (subject hash); the coverage claim is for
**exchangeable same-site subjects** + the fixed finite batching protocol `𝓑(S)` — **not** "future" subjects and not
across sites/clinics. (The draft's "historical/future" wording is dropped; exchangeability is the random-split claim.)

**v2-router replay (resolution 6):** the v2 baseline used in `red_router > red_v2_router` is the **v2 C0 HGB recipe
refit once on the DEV FIT pool** (frozen-from-DEV, parallel to v3), then run under the **identical Arm-B protocol**:
same site-local CAL subjects → its own joint-residual `q` → same EVAL subjects. Apples-to-apples.

**G1 (external measurement/regression), per disease:**
- ≥1 action-specific held-out harm AUROC ≥ 0.60 (evaluable; report per-site).
- continuous risk prediction improves over v2 C0 by a **frozen metric = subject-clustered mean absolute error (MAE)
  of μ̂ vs ΔR** (lower better), per disease.

**G2 (useful calibrated control), per disease — ALL must hold:**
- `red_router > 0`; `red_router > red_bestfixed` (best-fixed selected on **DEV only**);
  `red_router > red_v2_router` on the identical held-out EVAL subjects.
- oracle-benefit retention ≥ 0.50; adaptation coverage ≥ 0.20.
- harmful adapted-batch rate < best-fixed under a frozen subject-clustered test.
- **Two-site success rule (resolution 8):** with the two admissible sites per disease, **BOTH** sites must
  individually satisfy {`red_router>0`, `red_router>red_v2_router`, harmful-rate < best-fixed}, **and** the
  disease-macro (pooled-EVAL) retention ≥ 0.50 and adaptation coverage ≥ 0.20. **Contingency:** if a disease retains
  only **one** admissible site after the audit, G2 is evaluated single-site and the verdict is labeled
  **site-specific (no within-disease external replication)**; if a disease retains **zero**, G2 is not evaluated for
  that disease.

**Coverage diagnostic:** per site & disease, report exact subject-event coverage; run a **one-sided exact binomial
lower-tail undercoverage test** (H0: coverage ≥ 1−α) with **Holm correction across sites**, retaining the theorem's
explicit exchangeability assumption. The brittle "observed ≥ 0.90" rule is **not** used. Diagnostic failure is
recorded, never silently ignored.

## S7. Decision taxonomy (from V3.7)

`PROCEED_SAFE_ROUTER` (G1∧G2∧coverage) · `UTILITY_ONLY` (G2 pass, coverage fail) · `MEASUREMENT_ONLY` (G1 pass, G2
fail) · `TERMINATE` (held-out G1 fail) · `RUN_QUARANTINED/PROTOCOL_INVALID` (guards/isolation/provenance/prereg fail).

## S8. Hard guards (v2 set + v3 additions)

All v2 guards (no-label API, route_batch label-invariance, whole-batch, ΔR label-sensitivity, serialize roundtrip,
fallback retained, record-level hash, double-run determinism, split isolation incl. `U'_a−U_a=q'−q` for finite q and
`q=+∞→identity`) PLUS:
- set-permutation invariance of `μ̂, σ̂, U, action`; action-order invariance.
- FIT-only normalization **and** σ_min/w_min derivation (S3).
- strict positivity + finiteness of `σ̂` / predicted widths.
- CAL-label changes affect EVAL **only** through the calibrated standardized/CQR quantile (S2-style isolation,
  exact shift form generalized to the standardized score).
- EVAL-label permutation leaves `μ̂, σ̂, q, U, actions` bit-identical.
- serialization round-trip for the set encoder + all preprocessing state.
- record-level hash includes per-window paired inputs (or canonical digest), `μ̂, σ̂`, subject score, `q`, `U`,
  chosen action, split assignments, and `predictor_weights_sha256`.

## S9. Frozen manifest schema (resolution 10)

`run_manifest.json` MUST contain, with **distinct units**:
- `n_fit_subjects, n_fit_batches, n_cal_subjects, n_cal_batches, n_eval_subjects, n_eval_batches` (per disease/site/fold).
- **full 64-char SHA-256** for every raw and derived dump (no truncation); `dataset_version_doi` per site;
  `subject_list_sha256` (FIT/CAL/EVAL); `split_assignments`; `source_state_sha256`; `predictor_weights_sha256`;
  `protocol_commit`; `immutable_tag`; `environment_lock` (conda/pip freeze hash); `double_run_hash`.
- per-site/per-fold `n_cal_subjects (=m)`, conformal rank `k=⌈(m+1)(1−α)⌉`, `q`, `σ_min/w_min`, coverage, Holm p.

## S10. Hard metadata gates for lockbox admissibility (resolution 9)

A candidate site is **admissible** only if ALL pass (metadata-only, pre-endpoint):
1. **HC-vs-Patient binary target compatibility:** the site contains **both** healthy controls and patients with usable
   labels matching the DEV target (HC=0 / Patient=1). **A site without a usable HC-vs-patient contrast is excluded.**
   (ds007020 is treated as **unqualified** unless its metadata confirms this structure.)
2. **CAL-size feasibility:** after the deterministic subject-hash split, the site yields **≥ 30 CAL subjects**
   (substantially above the ~9 mathematical minimum where `k=⌈(m+1)·0.9⌉ ≤ m`), per disease.
3. License permits research use + redistribution of derived stats; raw signal available; 10–20-compatible montage;
   sampling rate resamplable to the DEV pipeline; a pre-specified **resting** condition exists; subject IDs present;
   **no subject/recording overlap** with the seven DEV cohorts; preprocessing compatibility.

Filled values for the chosen sites come from `ACAR_V3_LOCKBOX_AUDIT.md` (metadata only). Selection among admissible
sites uses **only** these metadata criteria — never any ACAR outcome.

## S11. `<<TBD>>` placeholders to fill only after DEV gate + audit (then copy into ACAR_FROZEN_v3.md)

- `<<TBD-after-DEV>>` selected predictor (C1/C2/C3) + its frozen weights hash; the realized DEV SELECT value; the
  frozen `σ_min/w_min` per disease; the frozen best-fixed action per disease.
- `<<TBD-after-audit>>` retained sites per disease (≤2) + their dataset DOIs/versions + subject-hash split seed +
  per-site CAL/EVAL subject counts.
- `<<TBD-before-freeze>>` exact numeric thresholds confirmed (the proposals above become binding only in
  ACAR_FROZEN_v3.md), plus the single binding-run command and output path.

## S12. Immediate plan (design-only)

1. (this file) pre-commit rules. 2. **metadata-only** audit of the 4 candidates → `ACAR_V3_LOCKBOX_AUDIT.md`.
3. implement per-window paired-set extraction + synthetic invariance tests. 4. implement C1/C2/C3 + standardized/CQR
conformal. 5. run **DEV gate only** (S2 + S4). 6. iff gate passes: copy selections into `ACAR_FROZEN_v3.md`,
commit/tag clean worktree on `acar`, run the single binding Arm-B evaluation once.
