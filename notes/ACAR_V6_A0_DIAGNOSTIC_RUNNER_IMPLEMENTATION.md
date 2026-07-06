# ACAR V6-A0a — Action-Viability Diagnostic Runner (implementation)

```
CODE + SYNTHETIC/FIXTURE TESTS ONLY
NO REAL DEV LABEL USE
NO REAL ΔR COMPUTATION
NO V6_CONTINUE / V6_STOP RESULT YET
NO POLICY FITTING
NO EXTERNAL / NO LOCKBOX
```

Implements (but does NOT run) the diagnostic runner for the pre-registered V6-A0 action-viability audit
(`notes/ACAR_V6_A0_ACTION_VIABILITY_AUDIT_PLAN.md`). V6-A0 asks the only question that matters after V5's DEV_STOP: *is any
adaptation action worth routing?* Executing the runner on real DEV labels is a **separate authorization (V6-A0b)**.

## Modules (all `acar.v5.*`; import torch-free; sklearn LAZY)

- **`acar/v5/v6_a0_action_viability.py`** (Q1 — oracle envelope + beneficial coverage). `batch_action_delta_r(outputs, y)` is the
  **single LABEL SEAM** (ΔR_a = NLL(f_a)−NLL(f_0)); `batch_label_free_features(outputs, n)` is label-free (paired φ_a + source
  confidence / batch entropy / batch size); `collect_eval_records(disease_folds, action_provider, provenance_by_subject=)` collects
  ELIGIBLE EVAL batch records (forced tails excluded, counted in `accounting`); `oracle_envelope(records)` →
  `{oracle_red_upper, beneficial_coverage, oracle_conditional_harm, no_safe_action_rate}`.
- **`acar/v5/v6_a0_sign_predictability.py`** (Q3 — the BINDING gate). `sign_cv_fold` (deterministic subject-grouped fold, salt
  `ACAR_V6A0_SIGN_CV_V1`), `build_sign_records`, `design_matrix` (paired features + source-conf/entropy/size + action one-hot +
  provenance one-hot; t3a's structurally-NaN Bures/post_sep → 0 per the v2 convention), `primary_sign_auroc` (pooled action-record,
  subject-grouped 5-fold OOF, L2 `LogisticRegression(C=1.0, class_weight='balanced', seed 0)`, train-standardization on TRAIN
  subjects only), `subject_block_permute` (permutes INTACT subject label-blocks ONLY within equal-record-count strata — never
  fragments a subject's block, handles the real unequal-size regime; singleton-size subjects stay fixed), `permutation_pvalue`
  (subject-block null, 1000 perms, `p=(1+#{null≥obs})/(1+n_valid)`, + `n_permutable_subjects` null-power diagnostic).
- **`acar/v5/v6_a0_report.py`** (schema + gate). `continuation_gate(per_disease_eval)` (EVAL-primary, BOTH diseases, all four
  sub-gates), `build_v6a0_report`, `validate_v6a0_report` (fail-closed; forbids any selection/routing/gate-pass/later-stage key).

## The five pinned details (as authorized)

1. **Primary action set** = V5 final admitted Stage-2B: `identity` + `{matched_coral, spdim, t3a}`, where the `matched_coral`
   *implementation* is `stable_matched_coral_v1` (via `stage2_real_action_provider.real_action_provider`). The old unsafe
   `pmct_predict_serialized` CORAL is NEVER used. `action_id = matched_coral`, `implementation = stable_matched_coral_v1` — no dual
   column. (guard: `test_v6a0_primary_action_set_uses_stable_matched_coral`.)
2. **Primary split = OOF EVAL only.** The continuation gate uses EVAL across the 5 OOF folds; each subject contributes through its
   EVAL fold only. FIT/CAL are descriptive (`descriptive_fit_cal_and_secondary`) and cannot set the decision. `primary_split` MUST
   be `"EVAL"`. (guard: `test_v6a0_eval_only_primary_gate`.)
3. **Forced-tail semantics (Stage-2B3).** `n < STAGE2_MIN_BATCH` = adaptation-ineligible: no non-identity action, oracle
   contribution 0, excluded from the beneficial-coverage denominator; counted separately in `accounting`. Q1's primary denominator
   = eligible EVAL batches. (guard: `test_v6a0_forced_tails_excluded_from_action_envelope`.)
4. **Sign-predictability primary model** (exactly one primary test): unit = eligible EVAL (subject, batch, action) record; target
   `beneficial(a,B)=1[ΔR_a<0]`; features = the 7 paired φ + source confidence + batch entropy + batch size + action one-hot +
   repair/completion provenance one-hots; L2 logistic regression C=1.0, class_weight balanced, seed 0, train-standardization on
   training subjects only; subject-grouped 5-fold OOF by canonical SubjectKey hash; per-disease subject-clustered OOF AUROC;
   subject-block permutation null (1000 perms, seed 0). Secondary/descriptive only (never overrides the primary gate): per-action
   AUROC/AUPRC, harmful target, calibration, coefficients. (guards: `test_v6a0_subject_grouped_cv_no_leakage`,
   `test_v6a0_permutation_null_subject_blocked`.)
5. **Continuation gate** (per disease, EVAL-primary; V6_CONTINUE iff BOTH pass all four): `oracle_red_upper > 0.02` AND
   `beneficial_coverage ≥ 0.15` AND `primary sign-AUROC ≥ 0.60` AND `subject-block permutation p ≤ 0.05`; else `V6_STOP`.
   `V6_CONTINUE` authorizes ONLY drafting a V6 protocol — NOT policy fitting, candidate selection, external read, or lockbox.
   (guard: `test_v6a0_continue_gate_requires_both_diseases`.)

## What V6-A0a proves (7 synthetic guards; full v5 suite now 222 modules, green py3.9 + py3.13)

```
2. primary gate is EVAL-only; FIT/CAL descriptive              test_v6a0_eval_only_primary_gate
4. forced tails excluded from action envelope + coverage       test_v6a0_forced_tails_excluded_from_action_envelope
5. matched_coral impl = stable_matched_coral_v1; no pmct       test_v6a0_primary_action_set_uses_stable_matched_coral
6. labels enter ONLY the ΔR evaluator (flip -> only ΔR moves)  test_v6a0_labels_only_in_deltaR_evaluator
7. sign-CV subject-grouped; train/test subject-disjoint        test_v6a0_subject_grouped_cv_no_leakage
8. permutation null permutes SUBJECT BLOCKS, not batches       test_v6a0_permutation_null_subject_blocked
9. report forbids candidate/threshold/route/G1-G6/Stage-4/     test_v6a0_continue_gate_requires_both_diseases
   external/lockbox; gate requires BOTH diseases
```
Property 1 ("uses only the admitted Stage-1B package + the 10 seed-20260711 refs") is a **V6-A0b run-guard** property (there is no
real package in synthetic tests) — V6-A0b will front the runner with a no-label admit/registry-sha/10-ref/S1-not-opened guard
mirroring the Stage-2B `--guard`, plus the label-firewall + forbidden-stage confirmation.

## Exploratory firewall (unchanged from the plan)

Everything V6-A0 produces is diagnostic. The DEV labels have been used to *observe* that the V5 action class is harmful, so any V6
policy designed from these observations is **exploratory**; efficacy is not established until a NEW dated protocol evaluates it on a
still-sealed held-out/external substrate. The runner must NEVER be turned into a selector: `validate_v6a0_report` fail-closes on any
selection/routing/gate-pass/later-stage key, and `V6_CONTINUE` authorizes only *drafting* a V6 protocol.

## Next

`V6-A0b` (separate authorization): the real, label-consuming diagnostic execution — no-label guard, then per-batch ΔR_a / oracle
envelope / beneficial coverage / sign predictability / f_0 calibration on the admitted V5 substrate → a single `V6_CONTINUE` /
`V6_STOP` diagnostic report. Still forbids policy fitting, candidate selection, rerouting, threshold tuning, Stage-4, external,
lockbox.
