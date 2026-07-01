# ACAR V5 — Stage-1B1 full-build readiness hardening (SYNTHETIC-ONLY, NO DATA)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B0 proved the real-data door defaults shut; Stage-1B1 proves that when
it opens it can ONLY build the complete, correctly-specified set of fold-contained substrates. Wiring + guards only: validates
contracts and path STRINGS, opens nothing, trains nothing. Code: `acar/v5/substrate/stage1b_full_build_manifest.py` +
`stage1_runtime_lock.py` patch; guards under `acar/v5/tests/test_stage1b_*`.

## The five hardening points
1. **Both gates validate the build-manifest SCHEMA internally.** `require_stage1b_ready` now calls
   `build_manifest_schema.validate_build_manifest(plan)` first (a malformed plan can no longer bypass Stage-1A preflight into a
   build gate). The full-build gate validates it via `validate_full_build_manifest`.
2. **`require_stage1b_full_build_ready(plan, auth, lock)`** — the gate a REAL full build must pass. Requires ALL 30 canonical fold
   refs present, each authorized, each carrying complete real DEV inputs; `built == 30`; final-external refs schema-only. It
   REJECTS the default plan-only spec (0 real refs) and any partial plan. `require_stage1b_ready` remains the dry/partial gate.
3. **`source_paths_by_cohort` replaces the scalar `source_path` for real inputs.** Each fold ref's real inputs must be a mapping
   whose keys equal the disease's frozen DEV cohorts (PD {ds002778,ds003490,ds004584} / SCZ {ds003944,ds003947,ds004000,ds004367}),
   each path cohort-EXACT (references that cohort and no other DEV cohort / external site / prior artifact / cache). A scalar
   `source_path` on a fold ref is rejected in a full build.
4. **Runtime lock cross-binds `implementation_base_sha`.** `implementation_base_sha` is now a required lock field and must equal
   the authorization's; a real full-build authorization is expected to pin the reviewed Stage-1B implementation commit SHA.
5. **Full 40-hex target sha for real run contracts.** `require_stage1b_full_build_ready` requires
   `protocol_tag_target_sha == 4278435975a72b1127803dd2cffab420c083e430` (no prefix). The structural contract validator still
   accepts a hex prefix (for review convenience); only the REAL full-build gate demands the full commit.

## Guards (synthetic; part of `acar/v5/tests/run_all.py`)
`test_stage1b_full_build_requires_30_refs` (30-ref full build ok; prefix-sha / default / partial / scalar-source_path rejected) ·
`test_stage1b_source_paths_by_cohort` (keys == disease cohorts; cohort-exact; site/artifact rejected) ·
`test_stage1b_runtime_lock_binds_implementation_sha` (impl-sha required + cross-bound; full-build gate enforces it) ·
`test_stage1b_build_gate_validates_schema` (both gates reject a malformed plan via schema validation).

Full v5 suite = 26 guard modules, green on py3.9 and py3.13.

## Still forbidden in Stage-1B1 (unchanged)
real DEV read · OpenNeuro/Zenodo access · EEGNet/spectral-z training · source-state fitting · embedding dump · candidate
selection · S1/S2/S3 robustness execution · held-out/external read · lockbox consumption.

## Next gate (separate authorization; NOT started)
**Stage-1B real DEV read + full 30-ref fold-contained substrate build**, run ONLY behind a concrete run contract (full 40-hex tag
commit + reviewed implementation_base_sha) + a CAPTURED runtime lock (SLURM, acar-v5 env), passing
`require_stage1b_full_build_ready` before the first read/train. The real build code (DEV reader + encoder/source-state trainer +
registry populate) is a separate, later-authorized patch.
