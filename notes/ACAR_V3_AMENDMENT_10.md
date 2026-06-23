# ACAR v3 — Amendment 10 (protocol-to-code conformance: full S4 gate + bit-for-bit C0; DEV-engineering, NON-BINDING)

**Date:** 2026-06-23 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED`
Makes `develop.py` actually execute the FULL S4 gate the skeleton already froze (the `a086b6c` version SELECTed on only
S2+dominance). Mechanical conformance only — no new scientific design. v2 endpoint `1528a94` / tag `acar-v2-protocol`
@ `9b2f0c1` untouched. SYNTHETIC ONLY. This is the last bounded patch before the env-lock / spec / tag sequence.

## Resolutions (1:1 with the review)

1. **Full S4 admissibility now gates SELECT** (`s4_eligible`, pure): S2 pass · `max_a` dominance · **PD ≥1-action
   center-AUROC ≥0.60** · **SCZ subject-clustered MAE ≤ C0** · **disease-macro width ≥30% below C0** · **adaptation
   coverage ≥0.15** · **red_router > 0 AND ≥ C0** · **q finite** (no all-+inf candidate). Each criterion + its raw stat
   is stored in `CandidateReport` / `DevelopResult.s4_inputs`. A negative-red / low-coverage / wide / worse-MAE / weak-
   AUROC candidate can no longer SELECT.

2. **C0 is now a bit-for-bit v2 replay.** `develop._c0_vector` reuses `acar.features.paired_features /
   context_features / feature_vector` on the SAME captured execution → the exact v2 **11-D** vector (7 paired + 4
   context, NaN→0, v2 ordering). `ActionRegressor` is fit with **seed 0** (the v2 binding-runner seed, not `seed_es`).
   C0 now reports red / coverage / MAE / center-AUROC / width.

3. **Fallback enters the deployment denominators.** Candidate and C0 red/coverage now count fallback batches as
   identity (ΔR 0, not adapted): `coverage = n_adapt / (eligible + fallback)`, `red = −mean(chosen ΔR over ALL EVAL
   batches)`. MAE/width/AUROC stay on eligible batches (predictor output required) — the denominator split is explicit.

4. **S2 tail statistics are subject-equal-weighted.** C2 positive-tail-90 and C3 exceedance / positive-excess-95 / ΔR
   SD use a deterministic subject-weighted empirical distribution (each subject totals weight 1). Guard: a rare 1-batch
   subject is no longer buried by a 100-batch subject.

5. **`s4_select` is max-first.** `max_red = max(red_macro)`; tie set `{c : max_red − red_macro[c] ≤ 1e-4}` (relative to
   the TRUE max, so a non-transitive chain can't drift a far candidate in); within the tie set min width, then fixed
   `C2 ≺ C3 ≺ C1`. Guard includes the 1.00000 / 0.99991 / 0.99982 chain.

6. **Binding entrypoint + frozen runner.** `run_binding_dev` fails closed unless diseases=={PD,SCZ}, candidates==
   (C1,C2,C3), α==0.10, δ==0.0. `freeze_dev_run` runs it and writes to a **non-overwritable** dir: the selected
   per-disease predictor artifacts + final C0 regressors (pickled, **reload-hash verified**), the manifest (verdict,
   selected, α/δ, pool/eligible/source/label field-separated hashes, final_epochs, full `s4_inputs`); on DEV_STOP it
   writes a `DEV_STOP / NO_LOCKBOX_CONSUMED` marker and no artifacts.

## Guards (11 develop guards, all green on synthetic; `run_dev` verdict = DEV_STOP as expected for random data)
S4 each-criterion-gates (negative/below-C0 red, coverage<.15, <30% width, worse SCZ MAE, PD AUROC<.60, q=+inf, S2,
dominance); S4 max-first tie incl. non-transitive chain + DEV_STOP; **C0 vector == v2 `feature_vector` exactly**;
fallback changes red/coverage denominators (candidate + C0); S2 tails subject-equal-weighted (rare subject not buried);
binding entrypoint rejects wrong diseases/candidates/α; frozen runner non-overwrite + predictor reload-hash identity;
plus the prior multi-cohort registry / leak-isolation / fallback-eval / C2-floor-from-raw guards. The other five v3
suites' sources are unchanged this turn (Amendment 10 edits only `develop.py`); v2 has zero `acar.v3` imports.

## Next (gated): env lock → single `ACAR_V3_DEV_DESIGN_SPEC.md` consolidation (Amendments 1–10 as changelog) →
clean-PROCESS re-run of ALL v3 + v2 guards → clean-worktree (tracked+untracked) verify → tag `acar-v3-dev-design-v1` →
first real DEV read → **S2/S4 DEV gate only** (SELECT + frozen artifacts, or DEV_STOP/NO_LOCKBOX_CONSUMED). Binding
G2 / coverage / harmful-rate / two-site remain LATER external Arm B. No real DEV value before the tag; lockbox sealed.
