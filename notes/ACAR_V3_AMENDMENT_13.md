# ACAR v3 — Amendment 13 (binding preflight + provenance closure; DEV-engineering, NON-BINDING)

**Date:** 2026-06-23 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED /
NO FINAL SPEC YET`. The last bounded closure between code and the frozen protocol. After this: write the spec, clean
re-run, clean worktree, tag, first DEV S2/S4 gate (all GATED). v2 endpoint `1528a94` / tag `acar-v2-protocol` @
`9b2f0c1` untouched. SYNTHETIC ONLY.

## Resolutions (1:1 with the review)

1. **Preflight precedes any DEV file op; no bypass.** `run_dev_binding.run()` is a STDLIB-ONLY bootstrap that, before
   importing numpy/torch/sklearn or opening any cohort file, in order: parses the manifest, requires the output dir
   absent, validates the manifest schema, verifies `HEAD == protocol_commit` + tag → HEAD, verifies a CLEAN worktree,
   and checks each input file's declared `full_dump_sha256` (stdlib hashlib). The `verify_git=False` / `require_tag=
   False` bypasses are REMOVED. A new immutable `develop.BindingContext` (protocol commit / tag / clean status / env-
   lock hash / input_manifest_sha256 / binding command / repo root) is built ONLY after the preflight passes, and BOTH
   `run_binding_dev` and `freeze_dev_run` REQUIRE one. Synthetic guards use the private `_verified_context_for_tests`.

2. **Clean-worktree verification.** `verify_clean_worktree(root)` requires `git status --porcelain=v1
   --untracked-files=all` to be empty (modified tracked OR untracked files both fail), using the repo root resolved
   from the code location — not the caller's cwd.

3. **Environment lock now locks the thread runtime + import order.** `apply_runtime()` forces torch deterministic +
   intra-op threads = 1, `OMP_NUM_THREADS=1`, and a global `threadpoolctl.threadpool_limits(1)` (OpenBLAS/OpenMP/MKL
   behind numpy/scipy/sklearn run single-threaded; inter-op best-effort). The hashed lock records each threadpool
   backend's `internal_api / num_threads / version` (== 1 under the limit) + torch determinism/intra-op + OMP env;
   `verify_env_lock` applies the runtime first, so it verifies identically in a COLD (CLI) and a WARM (test) process.
   The CLI sets the runtime BEFORE importing the heavy libs. Lock regenerated:
   `env_lock_sha256 981e343ce301d1c245eb3063c2b00f8bcc929103eac9aad1a984668ec3726554`.

4. **Input manifest + CohortInput fail-open closed.** `validate_input_manifest` requires the exact schema: a 40-hex
   `protocol_commit`, exactly the seven `config.DISEASE` cohorts, each with a unique dataset_id + path + disease and ALL
   five field-separated SHA-256 as lowercase 64-hex (no optional hash). `loader.CohortInput.labels` is now stored as an
   IMMUTABLE mapping (post-construction mutation that would desync the label hash raises); added `raw_pipeline_sha256`
   (64-hex or None) and `dataset_version` provenance fields.

5. **S9 EVAL counts include fallback.** Per candidate the manifest now records `n_eval_eligible_batches`,
   `n_eval_fallback_batches`, and `n_eval_batches_total` with the strict `total == eligible + fallback` relation;
   per-cohort entries carry `raw_pipeline_sha256` / `dataset_version` alongside the field-separated dump hashes.

## Guards (17 develop guards + loader, all green; `run_dev` verdict = DEV_STOP on synthetic)
new/updated: CohortInput labels-immutable + raw_pipeline hex + binding-APIs-require-context; binding seven-cohorts with a
required context (no bypass); env-lock verify fails on torch/threadpool runtime drift; frozen runner records context
evidence (input_manifest_sha256, binding command, clean_status) + EVAL `total==eligible+fallback` + per-cohort
raw_pipeline/dataset_version; CLI stdlib-first fail-closed (clean/dirty/untracked worktree, tag/commit, required-schema
incl. uppercase-hash reject, existing-output, and **no `np.load` before the preflight**). Amendment 13 edits only
`develop.py`, `envlock.py`, `loader.CohortInput`, `run_dev_binding.py` (+ tests); the other v3 suites' sources are
otherwise unchanged; v2 has zero `acar.v3` imports. (Full clean re-run of all suites is the gated tag-time step.)

## Next (GATED — separate authorization)
1. Write the single `ACAR_V3_DEV_DESIGN_SPEC.md` (matching the FINAL code; Amendments 1–13 as changelog; includes the
   binding CLI command, the input-manifest schema, and `env_lock_sha256 981e343c…`).
2. Single-process, suite-by-suite clean re-run of ALL v3 + the v2 guard suite.
3. `git status --porcelain` clean (tracked + untracked).
4. Commit the final spec/code/env lock; tag `acar-v3-dev-design-v1`; verify the tag → the clean protocol commit.
5. Run the single binding CLI for the first real DEV read → **S2/S4 DEV gate only** (SELECT + frozen artifacts, or
   DEV_STOP / NO_LOCKBOX_CONSUMED). Binding G2 / coverage / harmful-rate / two-site remain later external Arm B; the
   lockbox stays sealed.
