# ACAR v3 — DEV DESIGN SPEC (tag candidate for `acar-v3-dev-design-v1`)

**This is the single normative DEV-design specification.** It supersedes `ACAR_V3_FREEZE_SKELETON.md` as the
authoritative description of the implemented DEV-lock code; the skeleton + `ACAR_V3_AMENDMENT_{1..14}.md` are retained
as the design changelog. Committing this file and tagging its commit `acar-v3-dev-design-v1` constitutes the
**DEV_DESIGN_LOCK**. It does **not** authorize any external Arm-B endpoint or lockbox access.

- **Status:** DEV-design lock candidate. NON-BINDING until tagged; after tagging, the FIRST real DEV read is the unique
  binding CLI run (below), whose only outcomes are `SELECT` (+ frozen DEV artifacts) or `DEV_STOP / NO_LOCKBOX_CONSUMED`.
- **Parent:** ACAR v2 `MEASUREMENT_ONLY` (`acar-v2-protocol` @ `9b2f0c1`; result `1528a94`).
- **Protocol commit:** defined as the **target of the tag `acar-v3-dev-design-v1`** (i.e. `git rev-list -n 1
  acar-v3-dev-design-v1`). This spec does NOT embed its own commit SHA (self-reference); the exact 40-char SHA is
  written into the external input manifest and the result manifest AFTER the tag is created.
- **Environment lock:** `notes/ACAR_V3_ENV_LOCK.json`,
  `env_lock_sha256 = 2cb61360a01af61001ac4a97e6269c16ee4d89c998122d22d557c7d7c84cab17` (python 3.13.7 / torch
  2.6.0+cu124 / numpy 2.4.4 / scipy 1.17.0 / scikit-learn + joblib + threadpoolctl; torch deterministic + intra-op =
  inter-op = 1; threadpool limit = 1; numpy quantile `linear`; scipy Wilcoxon PermutationMethod seed 0). `env eeg2025`.

## 1. Estimand, candidates, losses (S1)
Per batch `B`, action `a ∈ {matched_coral, spdim, t3a}` vs `identity`: `ΔR_a(B) = R_B(f_a) − R_B(f_0)` (NLL). No label
enters feature construction / normalization / selection / deployment. Predictors: **C1** DeepSets mean (Huber δ=1 in
standardized units); **C2** DeepSets mean+scale (Seitzer β-NLL β=0.5, `v.detach()**β`); **C3** DeepSets additive
one-sided CQR (`q̂₉₀=q̂₅₀+softplus(d)+ε`, pinball ½ρ₀.₅+½ρ₀.₉). **C0** = the v2 recipe (`acar.regressor.ActionRegressor`:
HGB≥40 / Ridge≥8 / constant, seed 0) on the **bit-for-bit v2 11-D feature vector** (`acar.features.feature_vector`);
comparator only, never selectable. Nonconformity: raw `ΔR−μ̂` (C0/C1), standardized `(ΔR−μ̂)/max(σ̂,σ_min)` (C2),
additive `ΔR−q̂₉₀` (C3). C2 deploy clamps `q⁺=max(q,0)`; C1/C3 use raw `q`.

## 2. DEV split (S5) — one algorithm, deterministic, permutation-independent
Per disease, pooled over its cohorts. Outer **K=5** subject-disjoint folds over ALL subjects (each EVAL once, incl.
fallback-only subjects, by canonical-`SubjectKey` hash, `seed_outer=0`). Non-EVAL **eligible** subjects → FIT/CAL
(`seed_fitcal=1`, `fit_frac=0.70`); FIT → TRAIN/VAL (`seed_es=2`, `train_frac=0.80`). The predictor sees **FIT only**;
the conformal `q` sees **CAL only** (exactly one subject-clustered joint score per eligible CAL subject); S2/S4
diagnostics aggregate on **OOF EVAL**. Each batch is executed **exactly once** (the source adapters), and its predictor
features and ΔR derive from that one execution. Fallback (`<MIN_BATCH=8`) batches are forced to identity (ΔR 0, not
adapted) but **retained** in EVAL accounting (`n_eval_batches_total == eligible + fallback`). Final refit: once on the
frozen eligible set at `final_epochs = round_half_up(median_k(best_epoch_k+1))`; C2 `σ_min,a = Q₀.₀₅` of the OOF
`scale_raw`.

## 3. S2 admissibility gate (per disease × action, subject-equal-weighted, fail-closed)
- **C2:** standardized-residual subject-balanced `|mean| ≤ 0.25`, `variance ∈ [0.5, 2.0]`, positive-tail-90 (subject-
  weighted) `∈ [0.8, 2.0]·z₀.₉₀`.
- **C3:** subject-weighted exceedance `P(ΔR>q̂₉₀) ∈ [0.05, 0.20]`, positive-excess-95 `≤ 2·(OOF ΔR SD)`, no quantile
  crossing.
- **`max_a` dominance (C1/C2/C3):** fractional-tie `max_a share_a ≤ 0.60`. C1 is selectable only if it passes this.

## 4. S4 selection gate (full admissibility + max-first SELECT)
A candidate is **eligible** only if ALL hold (`develop.s4_eligible`, pure): S2 pass · `max_a` dominance · **PD ≥1-action
center-AUROC ≥ 0.60** · **SCZ subject-clustered MAE ≤ C0** · **disease-macro width ≥ 30 % below C0** · **OOF adaptation
coverage ≥ 0.15** · **`red_router` > 0 AND ≥ C0** · **all fold `q` finite** (any `q=+∞` blocks). SELECT = **max** disease-
macro OOF `red_router`; tie set `{c : max_red − red_c ≤ 1e-4}` (relative to the true max) → smaller disease-macro width
→ fixed order `C2 ≺ C3 ≺ C1`. **No eligible candidate ⇒ `DEV_STOP / NO_LOCKBOX_CONSUMED`.** S6 width / MAE / best-fixed
are computed on DEV as S4 inputs only; binding G2 / site-local coverage / harmful-rate / two-site are **later external
Arm B** and are NOT emitted by the DEV gate.

## 5. Substrate, immutability, provenance
Disease-specific predictors. Source state per cohort is an immutable **bytes** `SourceStateArtifact` (covers
coef/intercept/`classes_`/moments/priors/`n_cls,d,rho,eps`/schema/vocab/prob-schema/source-fit/env; ephemeral
reconstruction for execution). A per-disease `SourceStateRegistry` resolves each batch's `source_state_ref` (unregistered
→ fail before any adapter). Field-separated provenance: `full_dump` (audit), `source_fit` (z_ev,y_ev), `deployment_input`
(z_te + keys), `label` (WindowKey-aligned y_te), `subject_list`. `source_state_ref` and every deployment identity depend
ONLY on `source_fit` + `deployment_input` — never on `full_dump` (which contains y_te). Each `loader.CohortInput` binds
dataset_id ↔ `LoadedDumpManifest` ↔ source artifact ↔ batches ↔ labels (immutable labels), validated at construction.

## 6. The binding DEV run — UNIQUE entry point
**Only the result of the single CLI is a binding DEV run.** Calling `develop.run_binding_dev` / `develop.freeze_dev_run`
or `develop._verified_context_for_tests` directly is **non-binding / quarantined**. `develop.BindingContext` is process
**evidence** that the preflight passed — it is NOT a tamper-proof security token.

```
python -m acar.v3.run_dev_binding --input-manifest <abs seven-cohort.json> --output <abs new dir>
```

The CLI is a stdlib-only bootstrap; before any heavy import or DEV file read it requires, in order: output dir absent →
manifest schema → `git HEAD == spec["protocol_commit"]` AND tag `acar-v3-dev-design-v1^{}` → HEAD → CLEAN worktree
(`git status --porcelain=v1 --untracked-files=all` empty) → each file's declared `full_dump_sha256` (stdlib) → set
single-thread runtime → import heavy → apply + verify the env lock → build `BindingContext` → open cohorts and re-check
all FIVE dump-derived hashes → `freeze_dev_run`. There is NO commit/repo/tag bypass. `freeze_dev_run` atomically claims
`<output>.tmp` (after checking output-absent + writable parent) BEFORE any DEV compute, refits the selected predictor +
C0 **exactly once**, serializes them (reload `verify_integrity` + file SHA-256), and `os.rename`s into place on full
success only; on `DEV_STOP` it writes a marker and no artifacts.

### Input manifest schema (created OUTSIDE the repo, after the tag)
```json
{ "protocol_commit": "<40-hex tag-target SHA>",
  "cohorts": [ { "dataset_id": "ds002778", "disease": "PD", "path": "<abs .npz>",
                 "full_dump_sha256": "<64-hex>", "source_fit_sha256": "<64-hex>",
                 "deployment_input_sha256": "<64-hex>", "label_sha256": "<64-hex>",
                 "subject_list_sha256": "<64-hex>", "raw_pipeline_sha256": "<64-hex>",
                 "dataset_version": "<non-empty str>" }, ... exactly 7 ] }
```
Exactly the seven `config.DISEASE` cohorts (PD ds002778/ds003490/ds004584; SCZ ds003944/ds003947/ds004000/ds004367),
unique dataset_id + path, all six SHA-256 lowercase 64-hex, non-empty `dataset_version`. The manifest lives OUTSIDE the
repo so the clean-worktree check is not broken by an untracked file. (For the archived erm_0 substrate the dumps' own
`feat_hash_te` is used as `raw_pipeline_sha256` — the frozen feature artifact hash that pins z.)

## 7. Result taxonomy (DEV stage)
- **`SELECT`** — one candidate passes §4; save the disease-specific predictors + C0 artifacts + full S5/S6/S8/S9
  manifest (env-lock hash, per-cohort manifests, per-fold FIT/CAL/EVAL hashes+counts+`m/k/q`, EVAL total/eligible/
  fallback, OOF digests, C2 σ_min, best-fixed, source_state_sha256, predictor+C0 file SHA-256, protocol commit/tag/
  command, `manifest_sha256`).
- **`DEV_STOP / NO_LOCKBOX_CONSUMED`** — no candidate passes; stop v3; no external endpoint touched.

External Arm B (binding G2, site-local coverage diagnostic, harmful-rate endpoint, two-site rule) remains **unauthorized**
and requires a separate `EXTERNAL_PROTOCOL_FREEZE` after this DEV gate + a metadata audit. The lockbox stays sealed.

## Changelog
`ACAR_V3_FREEZE_SKELETON.md` (S0–S13) + `ACAR_V3_AMENDMENT_{1..14}.md`. Amendments 1–9 build the design/loader/bake-off;
10 = full S4 gate + bit-for-bit C0; 11 = env lock + runner/provenance; 12 = tag-prep provenance + binding CLI; 13 =
binding preflight (stdlib bootstrap, clean worktree, CohortInput); 14 = binding-entry closure (no bypass, atomic claim,
full provenance, inter-op lock). All NON-BINDING, synthetic-fixture-tested; v2 endpoint untouched throughout.
