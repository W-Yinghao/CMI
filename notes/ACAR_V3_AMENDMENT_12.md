# ACAR v3 — Amendment 12 (tag-prep provenance closure + binding CLI; DEV-engineering, NON-BINDING)

**Date:** 2026-06-23 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED /
NO FINAL SPEC YET`. The last bounded closure between code and the frozen protocol. After this: write the spec, clean
re-run, clean worktree, tag, first DEV S2/S4 gate (all GATED). v2 endpoint `1528a94` / tag `acar-v2-protocol` @
`9b2f0c1` untouched. SYNTHETIC ONLY.

## Resolutions (1:1 with the review)

1. **Predictor reload fail-OPEN fixed.** The chained `reloaded != art != refit_sha` became two separate `or`-guarded
   checks (reload≠artifact, artifact≠run-result), each with its own error + guard.

2. **Environment lock is a real runtime lock.** `env_versions()` now also records **scikit-learn / joblib /
   threadpoolctl** (C0 HGB/Ridge/scaler + source-state LogisticRegression). `build_env_lock` captures a `runtime`
   block from the CURRENT process (torch determinism + num_threads + threadpool backends + OMP_NUM_THREADS).
   `apply_runtime()` sets the required deterministic runtime; `run_binding_dev` calls it then `verify_env_lock()` (no
   constant claim). The `verify_env=False` bypass is REMOVED — synthetic tests use the non-binding `run_dev`. The
   verified hash flows from `run_binding_dev` into the manifest (not recomputed at freeze time). Lock regenerated:
   `env_lock_sha256 8044d5f6a372f73c260f2357efca9b51ed85aad124e2b029a1cf54f11740cb12`.

3. **Binding binds to real per-cohort objects.** New immutable `loader.CohortInput` binds dataset_id ↔
   `LoadedDumpManifest` ↔ `SourceStateArtifact` ↔ batches ↔ labels and validates them mutually at construction
   (manifest dataset/disease; every batch disease/dataset_id/source_state_ref; recomputed field hashes == manifest;
   counts == manifest; unique batch digests; labels cover all WindowKeys exactly). `run_binding_dev` now takes the seven
   `CohortInput`s, requires exactly the `config.DISEASE` dataset IDs (one source-state ref each), and builds the
   registries — so two cohorts' source states cannot be swapped undetected (rejected at `CohortInput` construction).

4. **Manifest → full S9 + JSON safety.** Adds per-cohort `LoadedDumpManifest`s, per-fold FIT/CAL/EVAL subject-list
   hashes + counts (`n_fit/n_cal/n_eval_subjects/batches`) + `m/k/q_raw/q_used`, C0 per-fold `m/k/q` + C0 OOF digest,
   per-candidate OOF digest + S2 raw diagnostics + dominance shares + per-action AUROC + per-criterion S4 eligibility,
   `source_state_sha256` (not just ref), best-fixed per disease, predictor + C0 **file** SHA-256, and the protocol
   commit/tag/binding-command/output-path. JSON: `_json_safe` (numpy scalars coerced; non-finite → `NOT_EVALUABLE`
   sentinel; unknown objects raise — no silent `default=str`), `allow_nan=False`, and a **`manifest_sha256`** over the
   canonical manifest.

5. **Single binding CLI** `acar/v3/run_dev_binding.py`: `--input-manifest <seven-cohort.json> --output <new-dir>
   [--protocol-commit]`. BEFORE any DEV metric it fails closed unless the output dir is absent, git HEAD == the spec's
   protocol commit AND `acar-v3-dev-design-v1^{}` → HEAD, the env lock verifies, and the seven files exist with field
   hashes matching the manifest. Then it builds the `CohortInput`s and calls `freeze_dev_run`. First DEV run yields only
   SELECT (+frozen artifacts) or `DEV_STOP / NO_LOCKBOX_CONSUMED`.

## Guards (16 develop guards, all green; `run_dev` verdict = DEV_STOP on synthetic, the correct refusal)
new: CohortInput cohort-swap + missing-label rejected + no-verify_env-bypass; binding binds the exact 7 cohorts (wrong
id/count rejected, env-lock verified); JSON safety (non-finite→NOT_EVALUABLE, numpy coerced, unknown→TypeError); frozen
runner atomic + refit/execute-once + saved==run + verify_integrity + file & manifest SHA + truncation tamper; binding
CLI fail-closed (existing-output / missing-file / absent-tag / wrong-commit refused before DEV). Amendment 12 edits only
`develop.py` + `predictors.env_versions` + adds `envlock.py` runtime/`run_dev_binding.py`/`loader.CohortInput`; the
other v3 suites' sources are otherwise unchanged; v2 has zero `acar.v3` imports. (Full clean re-run of all suites is the
gated tag-time step.)

## Next (GATED — separate authorization)
1. Write the single `ACAR_V3_DEV_DESIGN_SPEC.md` (matching the FINAL code; Amendments 1–12 retained as changelog;
   includes the binding CLI command and the env_lock_sha256).
2. Single-process, suite-by-suite clean re-run of ALL v3 + the v2 guard suite.
3. `git status --porcelain` clean (tracked + untracked).
4. Commit the final spec/code/env lock; tag `acar-v3-dev-design-v1`; verify the tag → the clean protocol commit.
5. Run the single binding CLI for the first real DEV read → **S2/S4 DEV gate only** (SELECT + frozen artifacts, or
   DEV_STOP / NO_LOCKBOX_CONSUMED). Binding G2 / coverage / harmful-rate / two-site remain later external Arm B; the
   lockbox stays sealed.
