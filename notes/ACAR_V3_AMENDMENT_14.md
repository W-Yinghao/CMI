# ACAR v3 — Amendment 14 (binding-entry closure; DEV-engineering, NON-BINDING)

**Date:** 2026-06-23 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED /
NO FINAL SPEC YET`. Pure mechanical binding-entry closure (no model / S2 / S4 / split / endpoint change). After this:
write the spec, clean re-run, clean worktree, tag, first DEV S2/S4 gate (all GATED). v2 endpoint `1528a94` / tag
`acar-v2-protocol` @ `9b2f0c1` untouched. SYNTHETIC ONLY.

## Resolutions (1:1 with the review)

1. **All commit/tag bypasses removed.** `run()` now takes ONLY `(input_manifest_path, output)` — no `protocol_commit`
   / `repo_root` overrides; the CLI drops `--protocol-commit`. The verified commit IS `spec["protocol_commit"]`; the
   repo root is resolved from the code location; `verify_protocol` verifies the tag UNCONDITIONALLY (the `require_tag`
   parameter is gone).

2. **Atomic output claim BEFORE any DEV compute.** `freeze_dev_run` pre-checks output-absent + parent-exists +
   parent-writable, then `os.mkdir(<outdir>.tmp)` (atomic) — all BEFORE `run_binding_dev`. A stale temp, a concurrent
   second runner, or an unwritable parent fails with ZERO adapter/training/metric calls (guard asserts the run count is
   0). On success the temp is `os.rename`d into place; on any failure it is removed.

3. **Full provenance is mandatory.** `validate_input_manifest` now requires `raw_pipeline_sha256` (lowercase 64-hex)
   and a non-empty `dataset_version` per cohort (no longer optional). After loading each cohort, `run()` re-checks ALL
   FIVE derived field hashes INCLUDING `full_dump_sha256` against the manifest — closing the file-substitution window
   between the stdlib preflight hash and the actual load.

4. **Replayable command + inter-op lock.** The recorded binding command uses `shlex.join([sys.executable, "-m",
   "acar.v3.run_dev_binding", "--input-manifest", <abs>, "--output", <abs>])` (shell-round-trippable even with spaces).
   The environment lock now records and verifies `torch.get_num_interop_threads() == 1` (`apply_runtime` sets it; the
   fresh CLI process sets it before any parallel work; the warm test process sets it at module import). Lock
   regenerated: `env_lock_sha256 2cb61360a01af61001ac4a97e6269c16ee4d89c998122d22d557c7d7c84cab17`.

## Guards (18 develop guards + loader, all green; `run_dev` verdict = DEV_STOP on synthetic)
new/updated: `run()` signature has no commit/repo bypass + `verify_protocol` has no `require_tag`; freeze atomic-claim
(stale temp / concurrent / unwritable-parent → zero DEV compute); required `raw_pipeline_sha256` + `dataset_version`
(each missing → fail; empty version → fail); full_dump load-time recheck catches substitution; shlex-round-trip command
with spaces; env-lock verify fails on torch intra-op / **inter-op** / threadpool drift. Amendment 14 edits only
`develop.freeze_dev_run`, `envlock`, `run_dev_binding`, and `loader.CohortInput` provenance (+ tests); v2 has zero
`acar.v3` imports. (Full clean re-run of all suites is the gated tag-time step.)

## Next (GATED — separate authorization)
1. Write the single `ACAR_V3_DEV_DESIGN_SPEC.md` (matching the FINAL code; Amendments 1–14 as changelog; includes the
   binding CLI command, the input-manifest schema, and `env_lock_sha256 2cb61360…`).
2. Single-process, suite-by-suite clean re-run of ALL v3 + the v2 guard suite.
3. `git status --porcelain` clean (tracked + untracked).
4. Commit the final spec/code/env lock; tag `acar-v3-dev-design-v1`; verify the tag → the clean protocol commit.
5. Run the single binding CLI for the first real DEV read → **S2/S4 DEV gate only** (SELECT + frozen artifacts, or
   DEV_STOP / NO_LOCKBOX_CONSUMED). Binding G2 / coverage / harmful-rate / two-site remain later external Arm B; the
   lockbox stays sealed.
