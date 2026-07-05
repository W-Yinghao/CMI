# ACAR V5 — Stage-2B0 Selection-Engine Implementation

```
CODE + SYNTHETIC/FIXTURE TESTS ONLY
NO REAL DEV LABEL USE
NO REAL CANDIDATE SCORING
NO REAL THRESHOLD FITTING
NO SELECTED CANDIDATE
NO S1/S2/S3
NO EXTERNAL
NO LOCKBOX

Stage-2B0 implements the binding selection engine but does not run it on the admitted package.
Stage-2B real DEV selection remains a separate authorization.
```

Stage-2 is the first label-consuming step of ACAR V5. Stage-2B0 builds the full selection engine and proves — on SYNTHETIC
fixtures — that it consumes the admitted Stage-1B package correctly, keeps labels out of routing, and cannot run on real data
without a Stage-2B authorization bound to the admitted package. No real DEV label is read, no candidate is scored on real data,
no threshold is fit on real data, and no candidate is selected.

## Pinned interpretations (user, 2026-07-05 — recorded so the real run inherits them)

**CAL/EVAL split — "CAL certifies · EVAL final":**
- FIT (train∪val): unlabeled standardization / FIT-only quantile thresholds only.
- CAL (cal): the H1–H3 LTT certification split. H1=G3 (UCB[L_harm_all]≤0.10), H2=G4 (UCB[harm_among_adapted]≤0.30), H3=G1
  (LCB[coverage]≥0.15) are computed on CAL for the family (candidate × disease × {H1,H2,H3}) with Holm at FWER α=0.05.
- EVAL (eval): the final OOF reporting split. G2 red / (red − v2_replay) margin (ε=0.02) and G5 red_upper / P3 comparator are on
  EVAL; the selected candidate's EVAL G1–G5 table is the reported Stage-2 DEV result.
- **A CAL-certified selected candidate that FAILS the final EVAL G1/G3/G4 report ⇒ Stage-2 DEV STOP condition — never a silent
  reselection from EVAL safety results.**

**f_0 = LDA readout of source_state** (EEGNet head is diagnostic-only): `f0(z)=softmax_k(w_k·z+b_k)`, `w_k=Σ⁻¹μ_k`,
`b_k=logπ_k−0.5μ_kᵀΣ⁻¹μ_k`, over the frozen `{means μ_k, shared pooled cov Σ, priors π_k}`; same readout for `p0=f0(z_pre)`,
`p_a=f0(z_after_action_a)`. Fail-closed on missing μ/Σ/π, non-invertible Σ, class order ≠ {control=0,case=1}, NaN/Inf, or shape ≠ [batch,2].

## Modules (new, `acar/v5/`)

- **stage2b_authorization.py** — structured, fail-closed Stage-2B gate. Pins stage/tag/full-target-sha, the admitted Stage-1B
  package (run_id + registry_sha256), EXACTLY the 10 canonical selection refs, EXACTLY the 22 frozen candidate ids, and the
  forbid flags (S1 refs / external / lockbox). No global boolean. `require_stage2b_ready` binds the auth to the actual package.
- **stage2_feature_loader.py** — loads a feat_dump's per-window embeddings grouped by subject + split_role (re-validates the
  label-free V5 schema). One batch = one subject's windows. Real embedding loading is a real-run concern (synthetic in Stage-2B0).
- **stage2_label_loader.py** — closure-backed EVALUATION-ONLY label view over the frozen `cohort_label_spec`; resolves one
  authorized subject at a time, no bulk dump, no path attribute. Labels are structurally unreachable from routing/scalarization.
- **stage2_action_records.py** — `SourceLDA` (the pinned f_0) + record assembly (reuses `acar.features.paired_features` for the 7
  features). PLUGGABLE action provider: `production_action_provider` reuses the FROZEN `acar.actions` (matched_coral/spdim/t3a)
  via a v5→old source-state adapter, LAZY-imported (torch/cmi.eval) so it never loads in the label-free suite; identity is served
  by the LDA. A torch-free `synthetic_action_provider` drives the tests.
- **stage2_thresholds.py** — FIT-only thresholds (thin, fail-closed wrapper over the frozen quantile universe; zero FIT records ⇒
  NON-EVALUABLE).
- **stage2_policy_eval.py** — the ONLY label-consuming module: applies a candidate to a split's per-subject batches → subject-
  clustered {adapted, harmful} records + red / red_upper (ΔR = NLL(f_a) − NLL(f_0)). Labels enter only here (evaluation view).
- **stage2_gates.py** — CAL: EB p-value inversion for H1–H3 + Holm (reimplemented pure-numpy). EVAL: G2 (effect size, not in Holm)
  + G5 (P3 fallback when red_upper≤0). Pluggable, fail-closed v2-replay seam.
- **stage2_selection_engine.py** — orchestrates OOF over folds → Holm certification (CAL) + utility (EVAL) → joint
  min_disease(red−v2) objective + deterministic tie-break (lower harm → higher coverage → P3≺P1/P2/P4≺P5) → SELECT or DEV_STOP.
  Fails closed without a valid, bound authorization; DEV_STOP if v2-replay is not evaluable or the winner fails final EVAL G1/G3/G4.
- **stage2_selection_report.py** — the Stage-2 DEV result schema (SELECTED-with-EVAL-table or DEV_STOP); fail-closed against any
  S1/S2/S3/external/lockbox key.

## Structured Stage-2B authorization (implemented + tested; NO real authorization issued)

```
stage = Stage-2B ; protocol_tag = acar-v5-protocol ; protocol_tag_target_sha = 4278435975a72b1127803dd2cffab420c083e430
stage1b_run_id = <admitted> ; stage1b_registry_sha256 = <admitted, 64-hex>
allowed_selection_refs = exactly the 10 seed-20260711 refs ; allowed_candidate_ids = exactly the 22 frozen candidates
forbid_s1_refs_for_selection = true ; forbid_external_read = true ; forbid_lockbox = true
```

## Guards (14 synthetic, in `run_all.py`; suite now 177 modules, green py3.9 + py3.13)

```
test_stage2b_authorization_contract_required          auth required + fail-closed + package binding
test_stage2b_rejects_nonselection_seed_refs           S1-seed refs barred from the selection authorization
test_stage2b_exact_22_candidate_universe              candidate universe = exactly the 22
test_stage2b_joint_pd_scz_single_candidate_identity   joint scope; single selected id across diseases
test_stage2b_fit_quantiles_use_fit_only               thresholds fit on FIT only; zero FIT ⇒ non-evaluable
test_stage2b_labels_do_not_enter_scalarization        routing/scalarization/thresholds label-free; eval-only label view
test_stage2b_g4_no_adapted_subjects_fails             G4 conditional-on-adapted; no-adapt ⇒ non-evaluable ⇒ fail
test_stage2b_holm_family_candidate_disease_h1h2h3_only Holm over {H1,H2,H3}; step-down correct
test_stage2b_g2_effect_size_not_in_holm               G2 point-estimate effect-size gate outside Holm
test_stage2b_g5_p3_fallback_when_upper_noninformative red_upper≤0 ⇒ P3 comparator fallback
test_stage2b_tiebreak_deterministic                   max-margin then lower-harm → higher-cov → conservative family
test_stage2b_report_does_not_include_external_or_s1_results  report excludes S1/S2/S3/external/lockbox
test_stage2b_binding_real_run_fails_without_stage2b_auth     engine fails closed without a bound auth; v2-replay fail-closed
test_stage2b_f0_lda_and_action_records                f_0 LDA matches the pinned formula + fail-closed; label-free records
```

Every `acar.v5` Stage-2B0 module imports with NO heavy dependency (numpy lazy; torch/sklearn only inside the real seams, never
loaded in the label-free suite).

## Real-run prerequisites (flagged, NOT silently implemented)

- The three real action transforms (matched_coral/spdim/t3a) are wired to the FROZEN `acar.actions` via the LDA/state adapter but
  are torch/cmi.eval-backed and are **validated at real-run time** (never exercised in the label-free suite).
- The v2-replay comparator (`acar.features.feature_vector` + `acar.regressor.ActionRegressor` seed 0; FIT-train / CAL-q /
  EVAL-route) is a fail-closed seam: if `v2_replay_red` is missing for either disease, real selection cannot run (DEV_STOP).
- The recording-ordered batch size (STAGE2_BATCH_SIZE=32 / MIN_BATCH=8) is the ACAR default; re-confirmed for the real run.

## Forbidden in Stage-2B0

```
no real DEV label use · no real candidate scoring · no real threshold fitting · no selected candidate
no S1/S2/S3 robustness · no external / held-out read · no ASZED · no lockbox · no substrate rebuild
no repair/montage/label policy change
```

## Next gate (SEPARATE authorization)

**Stage-2B — real DEV candidate selection**: the first real label-consuming V5 selection run, limited to the admitted Stage-1B
package only, the 10 canonical selection refs only, the exact 22 candidates only, joint PD/SCZ, G1–G5 only; no S1/S2/S3, no
external, no lockbox. It requires a new authorization pinned to this reviewed implementation SHA and issuance of a real Stage-2B
authorization, plus real-run wiring/validation of the action transforms and the v2-replay comparator.
