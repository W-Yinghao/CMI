# ACAR V5 — Stage-2B3: forced-tail action-eligibility correction

```
CODE + SYNTHETIC/FIXTURE TESTS ONLY

Stage-2B real selection pinned to f079aca FAILED before producing a result
(run_id acar-v5-stage2b-f079aca-r1; note ACAR_V5_STAGE2B_REAL_SELECTION_RESULT_acar-v5-stage2b-f079aca-r1.md).
Reason: a forced n=1 tail batch was sent to matched_coral BEFORE the forced-identity routing check
(stage2_policy_eval called subject_action_outputs on every batch), and stable_matched_coral_v1 had no
n<2 guard, so np.cov(Z) was undefined -> non-finite covariance -> SVD non-convergence.

Stage-2B3 pins forced-tail semantics:
  sub-MIN_BATCH tails are identity-only, counted in all-batch denominators, and EXCLUDED from all
  non-identity action evaluation and from the red_upper oracle.

No candidate-space, batch-size, MIN_BATCH, stable-CORAL hyperparameter (rho/eps/cap), CAL/EVAL interpretation,
G2 margin, or Stage-2 endpoint change is authorized.
```

The `f079aca` Stage-2B real-run authorization is `SUPERSEDED_BY_FORCED_TAIL_ACTION_ELIGIBILITY_BUG`. Stage-2B3 is code +
synthetic/fixture tests only: no real DEV labels, no real v2 replay, no real scoring, no thresholds on real data, no candidate
selection, no S1/S2/S3, external, ASZED, or lockbox.

## Pinned forced-tail contract (`n < STAGE2_MIN_BATCH = 8`)

```
route                          = identity
adapted                        = false
harmful                        = false
chosen ΔR                      = 0
red_upper contribution         = 0            (min_a ΔR_a is NOT computed — non-identity actions are inadmissible)
coverage numerator contribution= 0
denominator contribution       = PRESENT      (the tail is NOT dropped; it is an all-batch identity/fallback batch)
non-identity actions           = NOT evaluated (matched_coral / spdim / t3a not called; action provider not invoked)
```

The eligibility boundary is `n < STAGE2_MIN_BATCH` (=8), NOT `n < STAGE2_BATCH_SIZE` (=32): a partial but eligible batch
(`8 ≤ n < 32`) still runs all actions. Batch size and MIN_BATCH are unchanged.

## Changes (minimal)

- **`acar/v5/stage2_policy_eval.py`** (`evaluate_candidate_disease`) — the forced-tail check now runs BEFORE
  `subject_action_outputs`. A forced tail appends `{"adapted": False, "harmful": False, "forced_identity": True}`,
  `chosen_drs += 0.0`, `upper_drs += 0.0`, and `continue`s — the action provider is never invoked for it. Eligible batches are
  unchanged (all actions, `SCAL.decide`, `chosen ΔR`, `min_a ΔR_a` oracle) and now carry `"forced_identity": False`.
- **`acar/v5/stage2_stable_coral.py`** — `stable_matched_coral_v1` and `transport_operator` both raise
  `Stage2StableCoralError` for `n < 2` (target covariance undefined) BEFORE any `np.cov`/eig/SVD. Fail closed — NOT an
  identity-equivalent return (returning identity would hide a forced-tail misuse of the provider). The guard is exactly `n < 2`,
  so eligible batches (n ≥ 8) are unaffected.
- **`acar/v5/stage2_v2_replay.py`** — already implements the same contract (its `_batches` yields `(None, forced)` and every
  FIT/CAL/EVAL loop skips or routes forced tails to identity without calling the provider). No code change; a regression guard is
  added.
- **`acar/v5/stage2_selection_engine.py`** (`_fit_batches`) — already skips forced tails (`if forced_id: continue`). No change; a
  regression guard is added.
- `acar/v5/metrics.collect` (unchanged) uses `total = len(batches)`, so a forced tail stays in the coverage / L_harm_all
  denominator while contributing 0 to every numerator — exactly the pinned semantics.

## Guards (10 synthetic, torch-free; suite now 215 modules, green py3.9 + py3.13)

```
forced_tail_does_not_call_action_provider           forced tail -> ZERO action-provider calls; identity-only record
forced_tail_red_upper_zero                          forced-only subject red_upper term = 0.0
forced_tail_counts_in_denominator                   forced tail present in batch list (denominator); numerators exclude it
forced_tail_adapted_false_harmful_false             forced record = {adapted:False, harmful:False, forced_identity:True}
thresholds_exclude_forced_tails                     _fit_batches uses only eligible batches; no forced-tail provider call
v2_replay_excludes_forced_tails                     v2 comparator makes no provider call on a forced tail (same contract)
stable_coral_n1_direct_call_fails_closed            n<2 -> Stage2StableCoralError (no identity fallback); n>=2 valid
full_batch_still_calls_all_actions                  full 32-window batch still calls all 4 actions
partial_but_eligible_batch_still_calls_all_actions  8<=n<32 still calls all actions (boundary is MIN_BATCH, not 32)
run_selection_handles_n1_tail_without_crash_synth   end-to-end: run_selection completes on 1-window-tail subjects; a strict
                                                    provider (fails on any non-identity forced-tail routing) proves the regression
```

## Next gate (SEPARATE authorization)

**Stage-2B3P** — all-batch label-free engine-path stress on the admitted Stage-1B package INCLUDING forced tails (differs from
Stage-2B2P, which tested full 32-window batches only): for every `window_batches` batch, if `n < STAGE2_MIN_BATCH` assert the
non-identity action provider is called 0 times and `forced_identity=True`; else run identity/matched_coral/spdim/t3a and assert
finite contracts. No labels / v2 / thresholds / scoring / selection. Only after Stage-2B3P passes would a new real Stage-2B
authorization be pinned to the reviewed Stage-2B3 commit.
