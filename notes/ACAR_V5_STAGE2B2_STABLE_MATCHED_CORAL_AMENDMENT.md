# ACAR V5 — Stage-2B2 Stable matched_coral Amendment (`stable_matched_coral_v1`)

```
CODE + SYNTHETIC/FIXTURE TESTS ONLY

Stage-2B real selection pinned to a5c44c3 was superseded before full run.
Reason: matched_coral produced non-finite p_a on a real 32-window CAL batch.
The blocker is numerical conditioning of CORAL on the V5 256-D / 32-window substrate.

Authorized amendment:
  stable_matched_coral_v1 with eigenvalue condition cap and transport operator gain cap.

Not authorized:
  dropping matched_coral
  changing candidate space
  changing batch size
  PCA-reducing substrate
  real candidate selection
  S1/S2/S3
  external
  lockbox
```

The `a5c44c3` Stage-2B real-run authorization is `SUPERSEDED_BY_MATCHED_CORAL_NUMERICAL_BLOCKER` (`0ab40ec` also remains
superseded). See the blocker note `ACAR_V5_STAGE2B_REAL_SELECTION_BLOCKED_MATCHED_CORAL_NONFINITE.md`. Stage-2B2 is code +
synthetic/fixture tests only: no real DEV labels, no real v2 replay, no real scoring, no thresholds, no candidate selection.

## What changed (minimal — candidate universe preserved)

The action NAME `matched_coral` is kept; only its V5 Stage-2 *implementation* is replaced. `spdim` and `t3a` still route through
the frozen `acar.actions.apply_action`. The historical `acar.actions` / `cmi.eval.pmct_predict_serialized` behavior is NOT modified
globally.

- **`acar/v5/stage2_stable_coral.py`** — `stable_matched_coral_v1(source_lda, Z)`: numpy-only (no torch, no
  `pmct_predict_serialized`), deterministic, bounded, rank-aware. Fail-closed (no silent identity fallback).
- **`acar/v5/stage2_real_action_provider.py`** — the Stage-2 provider now routes `matched_coral` → `stable_matched_coral_v1`;
  `identity` = LDA f_0; `spdim`/`t3a` = frozen provider. `validated_real_action` keeps the FROZEN path (spdim/t3a + old-vs-new
  comparison).

## Pinned policy — `stable_matched_coral_v1`

```
action name:              matched_coral
implementation:           stable_matched_coral_v1
rho:                      0.1     (unchanged from the V5 source-state adapter)
eps:                      1e-3    (unchanged)
condition_number_cap:     1e6     (eigenvalue floor caps cond(Σ_shrunk))
transport_operator_smax:  10.0    (SVD cap on the whiten-color operator gain)
gate:                     reliability-style; gate uncertainty from IDENTITY f_0 probs (no extra unsafe global-CORAL pass)
class readout:            source_state LDA f_0 only
fallback:                 NONE — non-finite output ⇒ fail closed (Stage2StableCoralError)
```

Algorithm (Z ∈ ℝ[n=32, D=256]): shrink both target `cov(Z)` and source pooled covariance with `(1-rho)C + rho·tr(C)/D·I + eps·I`;
**eigenvalue-floor** each at `max(eps, λ_max/cond_cap)` (caps the condition number before (inv)sqrt); form
`M_raw = sqrt(C_R) · invsqrt(C_T)` and **SVD-cap** its singular values at `smax`; transport `Tz = μ_R + (Z−μ_T)·Mᵀ`;
reliability-interpolate `z_post = (1−α)·Z + α·Tz` with `α = clip(n/2D,0,1)·exp(−8·se)` and `se = tr(cov(f_0(Z)))/n` from the
identity readout; `p_a = f_0(z_post)`; then validate `z_post`/`p_a` finiteness + probability contract and raise on violation.

Boundedness comes from three composed bounds: the **shrink** is the PRIMARY conditioner — it alone caps `cond(Σ_shrunk) ≤
(1−ρ)·D/ρ + 1 ≈ 2305` for any PSD covariance at ρ=0.1, D=256 (so `invsqrt` eigenvalues ≤ `1/√min` are bounded); the **eigenvalue
floor** (`cond_cap = 1e6`) is a redundant safety net that is inert at ρ=0.1/D=256 but engages if the shrink were ever weaker /
D different (tested directly by feeding `_conditioned_eig` a `cond ≫ cap` spectrum); and the **SVD cap** bounds the operator gain
at `smax = 10`. A rank-deficient (rank ≤ 31 in 256-D) target covariance therefore can no longer produce unbounded transport →
no overflow → deterministic finite output. (This is why the frozen `pmct` path — which lacks the eigenvalue floor + SVD cap and
uses a weaker target-covariance conditioning — overflowed intermittently.)

## Guards (10 synthetic, in `run_all.py`; suite now 205 modules, green py3.9 + py3.13)

```
stable_coral_rank_deficient_32x256_finite            finite on rank-{1,2,5,16,31} + fully-duplicated 32x256 batches
stable_coral_operator_norm_capped                    M_smax ≤ smax; cap active when M_raw_smax > smax
stable_coral_condition_number_capped                 raw target cov near-singular (cond>cap) → floored cond ≤ cap; output finite
stable_coral_no_pmct_unsafe_call                     matched_coral bypasses pmct/acar.actions; spdim/t3a still frozen
stable_coral_no_silent_identity_fallback             non-finite readout ⇒ Stage2StableCoralError (no identity fallback)
stable_coral_label_free                              signature (source_lda, Z); no label read
stable_coral_preserves_shape_and_probability_contract p_a [32,2] finite [0,1] row-sum 1; finite [32,256] z_post
stable_coral_feature_finiteness_contract             paired features finite incl Bures/post_sep
stable_coral_deterministic_repeated_calls            byte-identical repeats; no randomness in source
real_runner_still_requires_stage2b_auth              gate intact; amended provider routes matched_coral to stable
```

The condition-number guard doubles as the requested regression fixture: a rank-deficient 32×256 batch whose RAW target covariance
is near-singular (`cond = inf` / > cap) — the old unbounded path's failure mode — while `stable_matched_coral_v1` stays finite.

## Next gate (SEPARATE authorization)

**Stage-2B2P** — all-batch label-free stable-action stress on the admitted Stage-1B package: admit the package; 10 selection refs
only; open ALL full 32-window batches across train/val/cal/eval; run identity / stable matched_coral / spdim / t3a; assert every
p_a / z_post / paired feature finite; spdim + stable matched_coral checked on every full batch; NO labels, NO v2 replay, NO
thresholds, NO scoring, NO selection; run twice / on two nodes if possible. Only after Stage-2B2P passes would a new real Stage-2B
authorization be pinned to the reviewed Stage-2B2 commit.
