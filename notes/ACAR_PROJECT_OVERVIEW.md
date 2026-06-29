# ACAR — Project Overview (read-me-first)

*A single document to understand the whole project: where it came from, what ACAR is, how the v3 code is built, what
is frozen, what was run, and what is still gated. Written 2026-06-24. Authoritative pointers: `notes/EVIDENCE_LEDGER.md`
(claim status), `notes/ACAR_V3_DEV_DESIGN_SPEC.md` (the normative DEV-design lock), `notes/ACAR_FROZEN_v2.md` (v2
protocol), and `notes/ACAR_V3_AMENDMENT_{1..14}.md` (the design changelog).*

---

## 0. One-paragraph summary

The paper's question is **EEG test-time adaptation (TTA) safety**: when you adapt a frozen EEG classifier to a new
batch, does the adaptation *help or harm*? ACAR ("Action-Conditional counterfactual Adaptation-Risk router") **predicts
negative transfer** — the paired incremental risk `ΔR_a(B) = R_B(f_a) − R_B(f_0)` of taking adaptation action `a` on
batch `B` versus doing nothing — rather than predicting distribution shift or absolute accuracy. **ACAR v2** showed the
*signal exists* (label-free features predict harm out-of-fold on both PD and SCZ) but the router is *not deployable* (the
"measurement→control gap") → binding verdict **MEASUREMENT_ONLY**. **ACAR v3** is a redesign (a heteroscedastic
set-conformal router with a strict pre-registration + admissibility/selection gate) whose code is now **frozen at a
DEV-design lock** (`acar-v3-dev-design-v1 @ 817b04f`); the first real DEV gate run was launched and **operationally
aborted (killed) before producing a verdict**.

---

## 1. Lineage (how we got here)

This is **Direction 2** of a larger AAAI effort. Earlier directions are closed and recorded:

- **Tri-CMI / LPC-CMI** — the original idea: measure/control EEG domain leakage as conditional mutual information
  `I(Z;D|Y)`. Outcome: the *measurement* was rock-solid but no *control* method survived audit — the three LPC
  "pillars" collapsed (leakage reduction was via representation collapse; the calibration win was a temperature
  side-effect; the accuracy match was just matched-CORAL). Net contribution = the **measurement→control gap**. (See
  `notes/EVIDENCE_LEDGER.md`, memory `cmi-survivor-audit`.)
- **A0 gate-falsification line** — tried a source-free "harm controller" to abstain from harmful adaptation. Closed as
  **DIAGNOSTIC_ONLY**: no source-free score reduced deployed loss; density/CMI signals were wrong-signed; a promising
  batch-rollback turned out to be target-label leakage. (memory `cmi-gate-falsification`.)
- **ACAR (Direction 2)** = the **leak-proof successor** to the A0 line. It keeps the honest question (predict harm) but
  fixes A0's five leakage modes and changes the estimand to a *paired, action-conditional* incremental risk.

Two other parallel directions exist as isolated packages and are **not** part of this v3 work: `oaci/` (strict-DG
overlap-aware), `csc/` (concept-shift certificates), `tos_cmi/` (task-orthogonal selective CMI), `h2cmi/`
(hierarchical-D). They are tracked in separate memories.

---

## 2. The ACAR method

**Estimand (the pivot).** For a natural deployment batch `B` (recording-ordered, size `B=32`), an adaptation action
`a ∈ {matched_coral, spdim, t3a}` versus `identity` (= `f_0`, do nothing):
`ΔR_a(B) = R_B(f_a) − R_B(f_0)` with `R` = NLL. `ΔR_a > 0` means the action *harmed* this batch. This is **not** shift
magnitude and **not** absolute accuracy — it is the counterfactual value of adapting.

**Why it's leak-proof.** No label enters feature construction, normalization, model selection, or deployment. Features
`φ_a(B)` are label-free paired pre→post observables (d_entropy, d_margin, flip_rate, JS, Bures, post_sep, n_eff +
context). Labels (`y_te`) are read **only** to compute `ΔR` (the target / endpoint), and the v3 loader makes that a
*structural* firewall (below). A0's failure modes (y in scoring, per-class batch deletion, non-serializable state, etc.)
are all closed.

**Substrate.** The archived `erm_0` (CITA-no-LPC) 16-dim tangent-space feature dumps from the LPC-CMI closeout
(`archive/lpc-cmi-failed/results/feat_dump_v4/audit_{disease}_{cohort}_erm_0.npz`), 7 cohorts:
PD `ds002778/ds003490/ds004584` (230 subjects), SCZ `ds003944/ds003947/ds004000/ds004367` (225 subjects). GPU-free.
Conda env **eeg2025**. The dumps carry `z_ev/y_ev` (source readout training), `z_te/y_te` (deployment + labels),
`subject_id_te/recording_id_te/window_index_te`, and provenance (`feat_hash_te`, `freeze_a1_hash`).

---

## 3. ACAR v2 — the binding result (CLOSED = MEASUREMENT_ONLY)

Run under `notes/ACAR_FROZEN_v2.md`; pre-run code tagged **`acar-v2-protocol` @ `9b2f0c1`**; result commit `1528a94`.

- **G1 (signal exists) HOLDS** on BOTH diseases: label-free paired pre→post features predict negative transfer
  out-of-fold (e.g. matched_coral d_margin AUROC PD 0.650 / d_entropy SCZ 0.790; spdim flip_rate ~0.68 both).
- **G2 (deployable router) FAILS**: the conformal router does not reduce deployed NLL vs best-fixed/random-abstain with
  meaningful coverage; the gap is **measurement → risk-regression → calibrated-control** (3 links), not conformal
  alone.
- **Coverage**: PD 0.900 (207/230) meets nominal; **SCZ 0.8933 (201/225)** is a literal diagnostic miss (24 subjects
  uncovered; 0.67 pp below 0.90). Reported, not enforced; never rounded/rerun.
- **Verdict: MEASUREMENT_ONLY.** Defensible claim = a *label-free action-conditional paired-harm predictor of negative
  transfer*; NOT a deployable router. Working title: **"Predicting Negative Transfer Is Not Enough: The
  Measurement–Control Gap in EEG TTA."**

v2 is closed and immutable; it is the **parent** of v3 and must never be modified in place.

---

## 4. ACAR v3 — the redesign (HSCR)

**Hypothesis:** a **Heteroscedastic Set-Conformal Router** can close part of v2's measurement→control gap with a
stronger risk-regressor + properly-calibrated joint conformal — *without* loosening the operating point (still
**α=0.10, δ=0**, same actions). v3 is a **strict pre-registration**: the protocol RULES were committed *before* any DEV
numerical run, and the first DEV run can only emit `SELECT` (a candidate passes) or `DEV_STOP / NO_LOCKBOX_CONSUMED`.

### 4.1 Candidates (S1)
DeepSets set-encoders over the per-window paired features (shared ψ + pooling mean⊕std + shared ρ + per-action heads):
- **C1** mean-only (Huber δ=1 in standardized units), nonconformity raw `ΔR−μ̂`.
- **C2** mean+scale (Seitzer β-NLL β=0.5, `v.detach()**β`), nonconformity `(ΔR−μ̂)/max(σ̂,σ_min)`, deploy clamps
  `q⁺=max(q,0)`.
- **C3** additive one-sided CQR (`q̂₉₀=q̂₅₀+softplus(d)+ε`, pinball ½ρ₀.₅+½ρ₀.₉), nonconformity `ΔR−q̂₉₀`, no clamp.
- **C0** = the **v2 recipe verbatim** (`acar.regressor.ActionRegressor`: HGB≥40 / Ridge≥8 / constant, seed 0) on the
  bit-for-bit v2 11-D feature vector — **comparator only, never selectable**.

### 4.2 The DEV estimator (S5 — one algorithm, deterministic, permutation-independent)
Per disease, pooled over its cohorts. Outer **K=5** subject-disjoint folds over ALL subjects (each EVAL once, incl.
fallback-only subjects; canonical-`SubjectKey` hash, `seed_outer=0`); non-EVAL **eligible** → FIT/CAL (`seed_fitcal=1`,
0.70); FIT → TRAIN/VAL (`seed_es=2`, 0.80). **Predictor sees FIT only; conformal `q` sees CAL only** (exactly one
subject-clustered joint score per eligible CAL subject); **S2/S4 aggregate on OOF EVAL**. Each batch is executed
**exactly once** (the source adapters), and its features + ΔR come from that single execution. Fallback (`<MIN_BATCH=8`)
batches are forced to identity (ΔR 0, not adapted) but **retained** in EVAL accounting. Final refit: once on the frozen
eligible set; C2 `σ_min,a = Q₀.₀₅` of the OOF `scale_raw`.

### 4.3 Gates
- **S2 admissibility** (per disease×action, subject-equal-weighted, fail-closed): C2 residual mean/variance/positive-
  tail bands; C3 exceedance/positive-excess/no-crossing; `max_a`-dominance ≤ 0.60 (fractional tie) for C1/C2/C3.
- **S4 selection** (`develop.s4_eligible`, pure): a candidate is eligible only if ALL hold — S2 pass · dominance · PD
  ≥1-action center-AUROC ≥0.60 · SCZ subject-clustered MAE ≤ C0 · disease-macro width ≥30% below C0 · OOF adaptation
  coverage ≥0.15 · `red_router` >0 AND ≥C0 · all fold `q` finite. **SELECT** = max disease-macro OOF `red_router`; tie
  set `{c: max_red−red_c ≤ 1e-4}` → smaller width → fixed `C2 ≺ C3 ≺ C1`. No eligible candidate ⇒ `DEV_STOP`.
- **S6 width/MAE/best-fixed** are computed on DEV only as S4 inputs. **Binding G2, site-local coverage, harmful-rate,
  two-site rule are LATER external Arm B** — the DEV gate never emits them.

---

## 5. Code architecture (`acar/v3/`)

All synthetic-fixture-tested; runs on the eeg2025 env. v2 code (`acar/*.py`) is untouched and has **zero** `acar.v3`
imports.

| module | role |
|--------|------|
| `_util.py` | `frozen_array` — bytes-backed immutable ndarrays (writeable cannot be re-enabled). |
| `set_features.py` | `WindowActionSet` (per-window paired features + availability masks + context), `build_action_sets`, `_build_was` (feature computation from precomputed adapter outputs), canonical digests. |
| `data.py` | `SubjectKey/RecordingKey/WindowKey`, `DeploymentBatch` (no `y`; canonical row order; fallback⇔n<MIN_BATCH), `LabeledRiskRecord` (binds digest + action_outputs hash), `deployment_batch_digest`/`canonical_row_digest`, `build_deployment_batches`. |
| `normalizers.py` | FIT-only mask-aware input + target normalizers (floors 1e-6 / 1e-3). |
| `predictors.py` | `CandidatePrediction`, `score`/`upper_bound`, `DeepSetsNet`, immutable `FittedCandidateArtifact` (canonical `<f4` bytes, injective hash incl. its own HP snapshot, `verify_integrity`), `make_artifact`, frozen `HP`, `env_versions`. |
| `conformal.py` | subject joint score, `conformal_rank/conformal_q` (strict +∞ when k>m), `route` (fail-closed), `harmful_rate_test` (one tie-aware Wilcoxon estimand). |
| `training.py` | exact losses, **subject-balanced epoch optimization** (one step/epoch, gradient accumulation), `fit_candidate_earlystop`, `refit_candidate_fixed_epochs`, `final_epochs`, `TrainExample`/`DeploymentFeatureRecord`. |
| `loader.py` | the **structural real loader**: field-separated provenance hashes, immutable **bytes** `SourceStateArtifact` + `SourceStateRegistry`, single-execution `BatchActionExecutionRecord`, `ActionOutputsRecord`, `LoadedDumpManifest`, `CohortInput` (binds dataset↔manifest↔source↔batches↔labels), strict dtype readers, label firewall. |
| `splits.py` | the S5 split-as-one-algorithm (`cv_assignment`), permutation-independent canonical-SubjectKey hash splits. |
| `develop.py` | the DEV bake-off + S2/S4 gates, `run_dev` (non-binding), `BindingContext`, `run_binding_dev`/`freeze_dev_run` (require a context), C0 replay, S4 select, the frozen runner. |
| `envlock.py` | `build_env_lock`/`apply_runtime`/`verify_env_lock` — pins library versions + single-thread runtime (torch deterministic/intra-op/inter-op=1, threadpool limit 1). |
| `run_dev_binding.py` | the **single binding CLI** (stdlib-first preflight bootstrap). |

---

## 6. The binding/provenance machinery (why the code is so defensive)

The project went through **14 adversarial code-review rounds** (each: the user finds reproducible fail-closed/identity
gaps → an "Amendment N" closes them, synthetic-only). The result is a deployment path that is hard to fool:

- **Label firewall (structural):** the deployment path reads only `z_te`/ids/window-index — never `y_te`. Proven by a
  **label-poisoned proxy**: flip `y_te` → deployment digests, executions, and predictions are byte-identical; only ΔR
  changes. Field-separated hashes (`full_dump`/`source_fit`/`deployment_input`/`label`/`subject_list`) mean `y_te`
  cannot touch any deployment identity.
- **Immutable artifacts:** `SourceStateArtifact` (covers `classes_` + env; ephemeral reconstruction; no mutable sklearn
  exposed) and `FittedCandidateArtifact` (canonical bytes, injective length-prefixed hash, `verify_integrity`).
  Tampering any byte fails the stored hash.
- **One execution per batch:** features and ΔR share a single `BatchActionExecutionRecord` (no second adapter pass);
  cross-pairing a record with the wrong batch/source fails.
- **`CohortInput`** binds dataset_id ↔ manifest ↔ source ↔ batches ↔ immutable labels, validated at construction — two
  cohorts' source states cannot be swapped undetected.
- **Binding CLI (`run_dev_binding`)** = the ONLY way to produce a binding DEV run. Stdlib-first bootstrap, in order:
  output-absent → manifest schema → `git HEAD == protocol_commit` + tag→HEAD → **clean worktree** → per-file
  `full_dump_sha256` → set single-thread runtime → import heavy → apply+verify **env lock** → build `BindingContext` →
  open cohorts + re-check all 5 dump-derived hashes → `freeze_dev_run`. No commit/tag/repo bypass. `freeze_dev_run`
  atomically claims `<out>.tmp` **before any DEV compute**, refits predictor + C0 exactly once, serializes with reload
  `verify_integrity` + file SHA-256, and `os.rename`s into place only on full success (DEV_STOP → marker only).
- Calling `run_binding_dev`/`freeze_dev_run`/`_verified_context_for_tests` directly is **non-binding / quarantined**;
  `BindingContext` is process evidence, not a tamper-proof token.

---

## 7. Current frozen state (what is on origin/acar)

- **DEV_DESIGN_LOCK:** `notes/ACAR_V3_DEV_DESIGN_SPEC.md` committed at **`817b04f`**, tagged **`acar-v3-dev-design-v1`**
  (annotated tag object `c3239de` → commit `817b04f`), both pushed to `origin/acar`. This is the protocol commit; it
  must never be amended/rebased/force-pushed and the tag must never move.
- **Environment lock:** `notes/ACAR_V3_ENV_LOCK.json`,
  `env_lock_sha256 = 2cb61360a01af61001ac4a97e6269c16ee4d89c998122d22d557c7d7c84cab17`.
- **All guard suites pass single-process** at the lock: 6 v3 suites (`test_set_features`, `test_data`, `test_training`,
  `test_predictors_conformal`, `test_loader`, `test_develop`) + the v2 leakage-guard suite. On synthetic data
  `run_dev` returns `DEV_STOP` (the correct refusal — random data can't pass the full S4 gate).
- **v2 unchanged:** `acar-v2-protocol @ 9b2f0c1`, result `1528a94`.

---

## 8. The first real DEV run — OPERATIONALLY ABORTED (no verdict)

Launched the binding CLI at the tagged commit on the real 7 cohorts (PD 230 + SCZ 225 subjects, d=16; input manifest
built outside the repo with the dumps' `feat_hash_te` as `raw_pipeline_sha256`). **Preflight passed** (HEAD==protocol
commit, tag→HEAD, clean worktree, env lock, per-file hashes). The S2/S4 gate then computed silently and the process was
**killed before producing a verdict** (external session/timeout, exit ≠ 0). Per the acceptance rules this is
**no scientific verdict / operationally aborted — NOT `DEV_STOP`**. Evidence: `binding_run.log` empty; `dev_out` never
formed (no atomic rename); only an empty `dev_out.tmp` remains (the fail-closed stale-temp marker). No auto-rerun was
performed. The DEV gate has simply **not yet been evaluated** on real data.

**To re-run** (when authorized): the gate must run at the tagged commit on a clean worktree, in a process allowed to run
to completion (the bake-off over ~550 eligible batches × source adapters + the C1/C2/C3 + C0 fits is long — tens of
minutes to hours). Steps: `git -C <acar worktree> checkout acar-v3-dev-design-v1` (detached HEAD == protocol commit) →
ensure a fresh output dir name (or remove the stale `dev_out.tmp` after recording it) → rebuild the out-of-repo input
manifest → `python -m acar.v3.run_dev_binding --input-manifest <abs json> --output <abs new dir>`. Result goes in a
**separate result commit** (the protocol commit and tag stay put).

---

## 9. Boundaries that remain sealed

- **External Arm B is UNAUTHORIZED.** Binding G2, the site-local coverage diagnostic, the harmful-rate endpoint, and the
  two-site rule require a separate `EXTERNAL_PROTOCOL_FREEZE` after the DEV gate passes + a metadata-only lockbox audit.
- The **seven cohorts are DEV-only** (model selection). Held-out / lockbox cohorts (candidates incl. SCZ Zenodo
  14808296, ASZED, PD ds007526; ds007020 excluded) are the route to confirmatory evidence and are **not** consumed at
  the DEV stage. The lockbox stays sealed.

---

## 10. Quick reference

- **Worktree:** `/home/infres/yinwang/CMI_AAAI_acar` (branch `acar`). Main checkout: `/home/infres/yinwang/CMI_AAAI`.
- **Env:** `conda env eeg2025`; `export ACAR_FEAT_DUMP=/home/infres/yinwang/CMI_AAAI/archive/lpc-cmi-failed/results/feat_dump_v4`.
- **Run guards:** `python -m acar.v3.tests.<suite>` (set_features/data/training/predictors_conformal/loader/develop);
  v2: `python -m acar.tests.test_leakage_guard`. Run suites one at a time (the env's background execution is flaky; kill
  strays with the `[a]`-class trick to avoid self-matching the shell).
- **Tag:** `acar-v3-dev-design-v1 → 817b04f`. **Env lock:** `2cb61360…`. **v2:** `acar-v2-protocol → 9b2f0c1`.
- **Design changelog:** `notes/ACAR_V3_FREEZE_SKELETON.md` (S0–S13) + `notes/ACAR_V3_AMENDMENT_{1..14}.md`.
- **Normative spec:** `notes/ACAR_V3_DEV_DESIGN_SPEC.md`. **Claim status:** `notes/EVIDENCE_LEDGER.md`.
