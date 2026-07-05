# ACAR V5 — Stage-2B1 Real-Seam Readiness

```
CODE + SYNTHETIC/FIXTURE TESTS ONLY
NO REAL DEV LABEL USE
NO REAL CANDIDATE SCORING
NO REAL THRESHOLD FITTING
NO SELECTED CANDIDATE
NO S1/S2/S3
NO EXTERNAL
NO LOCKBOX

Stage-2B1 wires and validates the real action-provider seam and the v2-replay comparator seam.
Real Stage-2B selection remains a separate authorization.
```

Stage-2B0 left two real-run seams intentionally fail-closed. Stage-2B1 wires + validates them on SYNTHETIC embeddings /
synthetic source_state / synthetic labels (v2-replay only). No real DEV label is read, no candidate is scored on real data, no
threshold is fit on real data, and no candidate is selected. The engine still fails closed without a Stage-2B authorization.

## Environment fact (test gating)

`acar.actions`, `acar.regressor` (sklearn), `cmi.eval.*` all IMPORT on py3.9 (home, no torch) — torch is lazy inside
`act_spdim`. When CALLED: **identity / matched_coral / t3a run torch-free on both Pythons; only spdim needs torch** (validated on
py3.13, its functional check skipped on py3.9). v2-replay uses `acar.regressor.ActionRegressor` (sklearn) — runs on both.

## 1. Production action transforms (`stage2_real_action_provider.py` + `stage2_action_provider_validation.py`)

`real_action_provider(name, source_lda, Z)` / `validated_real_action(...)`: identity is the source-state LDA f_0;
matched_coral / spdim / t3a route through the FROZEN `acar.actions.apply_action` via the v5→old source-state adapter (torch/cmi.eval
lazy). Every output is validated: p_a is [n,2], finite, in [0,1], rows sum to 1, class order [0,1]; z_post is None ONLY for t3a
(probability-only) and a finite [n,D] geometry array otherwise. The seam reads NO labels (consumes only source_state + z).

## 2. Non-finite feature semantics (pinned)

```
Required finite for EVERY action:  d_entropy, d_margin, flip_rate, JS, n_eff
Allowed NaN:                       Bures and post_sep ONLY for t3a (probability-only, z_post=None)
Forbidden:                         NaN/Inf in d_margin/JS/flip_rate/d_entropy/n_eff; NaN Bures/post_sep for matched_coral/spdim
```

`validate_feature_finiteness(action, features)` enforces this. A guard proves P4's post_sep selector can never pick t3a because
of its NaN post_sep (the argmax uses strict improvement `v > best + eps`, so NaN never wins).

## 3. v5→old source-state adapter (validated)

`validate_source_state_adapter(source_lda)` fail-closes unless `SourceLDA.old_state` has every field
`{clf, n_cls, mu_y, mu_pool, Sig_pool0, Sig_y0, pi_S, d, rho, eps}`, shapes means[2,D] / cov[D,D] / priors[2], class order [0,1],
a duck-typed LDA `clf` (predict_proba/predict/coef_/intercept_), finite entries, and invertible covariance.

## 4. v2-replay comparator (`stage2_v2_replay.py`)

`v2_replay_red_by_disease(disease, folds, ...)` recomputes the v2 recipe with `acar.features.feature_vector` (11-D = 7 paired + 4
context) and per-action `acar.regressor.ActionRegressor(seed=0)`: train ĝ_a on FIT, one-sided threshold on CAL, route on EVAL,
subject-macro `red`. `make_engine_v2_replay_provider(...)` adapts it to the engine's `v2_replay_provider(disease, ctx)`. Fail-closed
(`V2ReplayNotEvaluable`) if: a disease is missing, a FIT/CAL/EVAL split is missing, labels are unresolvable, a feature_vector is
non-finite, a regressor cannot fit, or `v2_replay_red` cannot be computed. Synthetic fixtures only; exact recipe fidelity validated
at real-run time.

## 5. v2-replay stays out of candidate routing

The comparator READS LABELS only inside itself (to form ΔR); it references no scalarization / routing / threshold code (guarded by
source inspection) and exposes no route/decide/fit-threshold/select API. Its only output is `v2_replay_red`, consumed solely by G2
(`red − v2_replay_red ≥ 0.02`).

## 6. Real Stage-2B stays fail-closed

No global enable flag was flipped. The selection engine still requires a valid structured Stage-2B authorization bound to the
admitted package; wiring the real v2-replay provider does NOT bypass the gate (guarded). Stage-2B0b's fixed 132-cell Holm family is
unchanged.

## Guards (13 synthetic, in `run_all.py`; suite now 195 modules, green py3.9 + py3.13)

```
real_actions_call_frozen_acar_actions          matched_coral/spdim/t3a route through acar.actions; identity=LDA
real_actions_are_label_free                    seam signatures + apply_action call carry no label
real_actions_probability_shape_and_finite      p_a [n,2] finite in [0,1] row-sum 1 (spdim torch-gated)
t3a_allows_only_geometry_nan                   t3a Bures/post_sep NaN allowed; routing features finite; others rejected
matched_coral_spdim_geometry_features_finite   geometric actions → finite geometry (spdim torch-gated)
p4_post_sep_nan_does_not_select_t3a            NaN post_sep never wins the P4 post_sep vote
source_state_adapter_required_fields           adapter field/shape/inverse fail-closed
source_state_adapter_class_order_fail_closed   class order != [0,1] rejected
v2_replay_uses_actionregressor_seed0           ActionRegressor(seed=0); finite v2_replay_red
v2_replay_missing_disease_fails                missing disease/fold/split → V2ReplayNotEvaluable
v2_replay_nonfinite_feature_vector_fails        non-finite feature_vector → V2ReplayNotEvaluable
v2_replay_does_not_enter_routing_or_thresholds  no routing/threshold/select references
real_runner_still_fails_without_stage2b_auth   real v2-replay wired ≠ gate bypass
```

## Still forbidden / next gate

```
real DEV label use · real candidate scoring · real threshold fitting · real v2 replay on the admitted package
selected candidate · Stage-2B result report · S1/S2/S3 · external / held-out · ASZED · lockbox
substrate rebuild · repair/montage/label policy changes
```

Next likely gate (SEPARATE authorization): **Stage-2B1P** — a narrow, label-free real-package action-provider smoke on the admitted
Stage-1B package (header-free embedding read for identity/matched_coral/spdim/t3a output contracts only; NO labels, NO scoring), and
only THEN the full real Stage-2B selection authorization. The v2 comparator is label-consuming and must not run on real DEV labels
until the actual Stage-2B authorization.
