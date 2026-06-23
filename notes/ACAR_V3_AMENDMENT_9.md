# ACAR v3 — Amendment 9 (DEV-lock completion: bake-off + S2/S4 gates; DEV-engineering, NON-BINDING)

**Date:** 2026-06-23 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED`
Turns the `3b00330` single-disease/single-candidate OOF skeleton into the registered `DEV bake-off + S2/S4 gates`, and
hardens the source state into a full bytes artifact. v2 endpoint `1528a94` / tag `acar-v2-protocol` @ `9b2f0c1`
untouched. SYNTHETIC ONLY. Per directive: this is the final DEV-lock completion before the env-lock / spec / tag
sequence — no further open-ended review.

## Resolutions (1:1 with the review)

1. **`SourceStateArtifact` is now a full immutable BYTES artifact** (peer of `FittedCandidateArtifact`). No mutable
   sklearn object is stored or exposed; `predict`/`execute` go through a private EPHEMERAL reconstruction.
   `source_state_sha256` covers coef/intercept/**`classes_`**/all moments/priors/`n_cls,d,rho,eps`/schema/vocab/prob-
   schema/source_fit + the canonical env. env is sorted/no-dup/length-prefixed (no silent dict-collapse). The frozen
   blob CARRIES its own hash + ref and `load_frozen_source_state_artifact` VERIFIES them (no new ref minted from the
   current env). Guard: tampering `classes_`, env, or any coef byte fails the stored hash (closes repro #5).

2. **`SourceStateRegistry`** (per disease, multi-cohort). A pooled-disease DEV run holds several cohort source states
   (3 PD / 4 SCZ); each batch resolves to the UNIQUE artifact matching its `source_state_ref`. Unique refs; one
   disease; an **unregistered ref fails before any adapter**. (The single-`source_artifact` API could not run a pooled
   disease — fixed.)

3. **Outer split covers ALL subjects; FIT/CAL only from non-EVAL eligible.** `cv_assignment(..., eligible=…)`: outer
   folds over every subject (each EVAL exactly once, incl. fallback-only), FIT/CAL drawn only from non-EVAL ELIGIBLE
   subjects. Fallback-only subjects stay in EVAL accounting but never enter FIT/CAL/predictor/refit.

4. **First OOF pass emits complete immutable `OOFRecord`s** (candidate, disease, subject, batch_digest, fold, action,
   ΔR, point, upper_center, **scale_raw**, scale_used, score, q, U, chosen). All S2/floor/S4 inputs come from these
   records — no second training pass.

5. **C2 final σ_min,a = Q05 of OOF `scale_raw`** (NOT `scale_used`/the fold floor), over the SAME records, with a
   pinned `numpy.quantile(method="linear")`. Guard: with `scale_raw≠scale_used` the floor responds to raw only.

6. **Real C0/v2 replay.** `run_c0` actually TRAINS the v2 recipe (`acar.regressor.ActionRegressor`: HGB≥40 / Ridge≥8 /
   constant) per action on FIT, computes a one-sided conformal `q` on CAL subject scores, and ROUTES on EVAL — over
   the identical splits/pool (cached executions). No more pool-hash-equality stand-in.

7+8. **S2 candidate-specific gates + S4 selection.** Pure, fail-closed: `s2_c2_gate` (subject-balanced standardized
   residual mean/var/positive-tail), `s2_c3_gate` (exceedance / positive-excess / crossing), `maxa_dominance`
   (fractional-tie `max_a share ≤ 0.60` for C1/C2/C3; C1 selectable only if it passes). `s4_select` = disease-macro OOF
   `red_router`, `1e-4` tie → smaller width, residual tie → fixed `C2 ≺ C3 ≺ C1`; **no passer → `DEV_STOP /
   NO_LOCKBOX_CONSUMED`**. `run_dev` runs the bake-off on each disease + C0 + the disease-macro select + (if SELECT)
   the per-disease refit.

9. **Phase boundary corrected** in the skeleton: the first real DEV run computes ONLY the **S2/S4 DEV gate**; binding
   G2 / coverage / harmful-rate / two-site are LATER external Arm B (S6). DEV emits only a SELECTed candidate (+ frozen
   artifacts) or `DEV_STOP / NO_LOCKBOX_CONSUMED`.

**Performance note (no silent cap):** the adapters (`execute`) are the expensive, candidate-independent step, so each
disease executes every eligible batch EXACTLY ONCE into a cache reused across all folds/candidates/C0 — consistent with
the single-execution principle. Nothing is sampled or truncated.

## Guards (all green, synthetic)
multi-cohort registry (3 refs) + leak isolation + unregistered-ref-fails-pre-adapter; fallback-only retained-in-EVAL /
never-FIT-CAL; S2 C2 var/mean/tail boundaries + floor-from-`scale_raw`; S4 SELECT max-red / 1e-4→width / full-tie
order / DEV_STOP; `run_dev` smoke + C0 actually-trained; plus the Amendment-8 loader guards now incl. classes_/env/coef
tamper → integrity failure. Six v3 suites + the v2 guard suite pass.

## Next (gated): env lock → single `ACAR_V3_DEV_DESIGN_SPEC.md` → clean-process re-run of all guards → clean-worktree
verify → `acar-v3-dev-design-v1` tag → first real DEV read → **S2/S4 DEV gate only**. No real DEV value before the tag;
lockbox fully sealed.
