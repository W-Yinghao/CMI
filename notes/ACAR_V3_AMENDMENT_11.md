# ACAR v3 — Amendment 11 (environment lock + runner/provenance closure; DEV-engineering, NON-BINDING)

**Date:** 2026-06-23 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED /
NO FINAL SPEC YET`. Generates the **environment lock** (authorized) and closes the last runner/provenance deltas
between code and the frozen protocol. The final `ACAR_V3_DEV_DESIGN_SPEC.md`, the tag, and the first real DEV read
remain GATED. v2 endpoint `1528a94` / tag `acar-v2-protocol` @ `9b2f0c1` untouched. SYNTHETIC ONLY.

## Environment lock (authorized this turn)
`acar/v3/envlock.py` + `notes/ACAR_V3_ENV_LOCK.json` (`env_lock_sha256 = 5633f4d3aaf82a5c9a5f351d1c862394dd2a8d0a83b6502744aac739421e995e`).
`build_env_lock()` is the single source; `verify_env_lock()` rebuilds the lock from the running process and asserts a
byte match (env_versions python 3.13.7 / torch 2.6.0+cu124 / numpy 2.4.4 / scipy 1.17.0, determinism flags, scipy
Wilcoxon convention, numpy quantile `linear`, schemas, frozen constants, the seven cohorts, full HP). `run_binding_dev`
verifies it before running (fail-closed on drift).

## Resolutions (1:1 with the review)

1. **C0 width is now subject-macro** (`_subject_macro_mean`, shared by candidate AND C0 for width and MAE) — a
   100-batch subject no longer outweighs a 1-batch subject, so "≥30% below C0" is apples-to-apples. Unit guard included.
2. **`q_finite` is literal.** `CandidateReport.any_q_inf = any(¬finite(q) for fold q)`; S4 blocks SELECT if ANY fold's
   `q=+∞` (was "all folds +∞").
3. **Refit EXACTLY once.** `run_dev` now produces and stores the final per-disease predictor artifacts AND final C0
   regressors (each refit once, in the run). `freeze_dev_run` ONLY serializes those stored objects — it never
   re-executes adapters or retrains. Guards assert refit-count == #diseases and execute-count == #eligible batches.
4. **Manifest expanded** to S5/S6/S8/S9: env-lock hash, field-separated `deployment_input`/`label`/`subject_list`/
   `source_state_refs`/`pool` hashes, per-fold `q`, per-fold CAL-score counts, **OOF record digest**, **C2 σ_min**,
   per-candidate S2 raw diagnostics + `max_a` fractional shares + per-action AUROC + per-criterion S4 eligibility,
   **best-fixed action per disease** (argmax DEV OOF red, frozen), predictor + C0 **file SHA-256**.
5. **Binding binds the seven cohorts.** `run_binding_dev` requires the exact dataset IDs
   (PD ds002778/ds003490/ds004584; SCZ ds003944/ds003947/ds004000/ds004367), one source-state ref per cohort, the
   right cohort count, AND a verified env lock — in addition to {PD,SCZ}/C1C2C3/α0.10/δ0.
6. **Frozen runner robust + deterministically tested.** ATOMIC write (build in `<outdir>.tmp`, `os.rename` only on full
   success; temp removed on any failure so no half-written non-overwritable dir). A forced-SELECT guard
   (monkeypatched `s4_eligible`) exercises the save path: refit/execute exactly once, saved artifact == the run's
   artifact (`predictor_sha256 == refit_sha256`), `verify_integrity()` on reload, predictor + C0 file SHA-256, and a
   truncation-tamper that fails closed.

## Guards (14 develop guards, all green; `run_dev` verdict = DEV_STOP on synthetic, the correct refusal)
All Amendment-10 guards plus: subject-macro mean (unequal batch counts) + q_finite-on-any-+inf; binding binds the exact
7 cohorts (wrong ids / wrong count rejected); forced-SELECT frozen runner (atomic, refit/execute-once, saved==run,
verify_integrity, file SHA, tamper-fails). Amendment 11 edits only `develop.py` + adds `envlock.py`; the other five v3
suites' sources are unchanged; v2 has zero `acar.v3` imports.

## Next (GATED — separate authorization): final spec → clean re-run → clean worktree → tag → first DEV read
1. Write the single `ACAR_V3_DEV_DESIGN_SPEC.md` (so it matches the FINAL code; Amendments 1–11 retained as changelog).
2. Single-process, suite-by-suite clean re-run of ALL v3 + the v2 guard suite.
3. `git status --porcelain` clean (tracked + untracked).
4. Tag `acar-v3-dev-design-v1` and verify it points at the clean protocol commit.
5. ONLY then the first real DEV read → **S2/S4 DEV gate only** (SELECT + frozen artifacts, or DEV_STOP /
   NO_LOCKBOX_CONSUMED). Binding G2 / coverage / harmful-rate / two-site remain later external Arm B. Lockbox sealed.
