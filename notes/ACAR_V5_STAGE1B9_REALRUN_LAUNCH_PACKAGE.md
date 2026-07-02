# ACAR V5 — Stage-1B9 final launch / registry-persistence / run-root hardening + REAL-RUN PACKAGE (CODE + SYNTHETIC/TEMP-FILE TESTS ONLY; NO REAL RUN)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B9 adds the last launch/persistence semantics so the Stage-1B real run
produces an auditable, hash-bound, monotone artifact package. All exercised on synthetic/temp-file inputs. Still NO real DEV read, NO
real training, NO real registry population, NO SLURM, NO Stage-2 selection, NO external/held-out.

## The five points (per review of 89bc5c1)
1. **Canonical registry persistence** (`stage1b_registry_io`): `export_registry` is deterministic (sorted refs + sorted keys),
   exactly 30 refs, each ref's hashes+meta; `write_registry` writes it ATOMICALLY (tmp→`os.replace`, temp cleaned on failure) and
   returns the sha256 of the exact bytes; `load_registry` round-trips to an identical export and re-validates each entry via
   `SubstrateRegistry.register`. Guard: registry-persisted-canonical.
2. **Registry hash binding + downstream admission** (`stage1b_finalize` + `stage1b_registry_io.admit_run`): finalize persists
   `registry.json` then writes `FINALIZED.json` whose `registry_sha256 == sha256(registry.json bytes)` with `n_refs==30`;
   `admit_run(output_root, run_id)` admits a run ONLY if both files exist, `status==FINALIZED`, `n_refs==30`, and the marker's
   `registry_sha256` matches `sha256(registry.json)` — Stage-2 consumes the FILE package, never an in-memory object. A missing file,
   a tampered registry.json, or a bad n_refs is inadmissible. Guards: finalized-marker-binds-registry-hash,
   downstream-admission-requires-registry-and-marker.
3. **Fresh run root** (`stage1b_launch_guard.assert_fresh_run_root`, wired into `stage1b_build` BEFORE factory instantiation):
   `output_root/run_id` must be ABSENT or EMPTY; any existing content (FINALIZED.json / registry.json / per-ref dir / stray file) or
   a non-dir aborts fail-closed. No resume, no overwrite. Guards: run-root-must-be-fresh, no-resume-or-overwrite.
4. **Finalize atomicity (with registry)**: populate → write registry.json (atomic) → write marker (atomic, LAST). A marker-write
   failure rolls the registry back AND removes registry.json; a registry.json write failure rolls the registry back — so the marker
   exists IFF the registry is fully populated AND persisted (covered by the existing finalize-marker-atomicity guard).
5. **Parser-level feature-dump hardening**: `feature_dump_schema.validate_loaded` now rejects `embedding_dim <= 0` (dumper-agnostic),
   in addition to forbidden-label / 2-D / finite / known-role / completeness checks. Guard: feature-dump-rejects-zero-dim-embedding.

Full v5 suite = **85 guard modules, green py3.9 + py3.13**; every `acar.v5.substrate` module imports with NO heavy dependency.

## Stage-1B REAL-RUN PACKAGE (the exact launch contract — for the SEPARATE real-run authorization)
The real run is the FIRST real-data step and needs its own explicit authorization. When authorized, it runs EXACTLY:

**Command (production entry):**
`run_stage1b_real_build(plan, authorization, runtime_lock, *, output_root, dev_reader_factory=make_real_dev_reader,`
`trainer_factory=make_real_trainer, dumper_factory=make_real_embedding_dumper)` — factories ONLY; the file-backed artifact writer +
per-ref containment is the default. (A CLI wrapper may load the three JSONs + call this; the CLI must set the runtime BEFORE
importing torch, per the runtime lock.)

**Authorization JSON** (`stage1b_authorization.STAGE1B_AUTH_FIELDS`, exact set, no extras):
`stage="Stage-1B"`; `protocol_tag="acar-v5-protocol"`; `protocol_tag_target_sha` = the FULL
`4278435975a72b1127803dd2cffab420c083e430` (the full-build gate requires the full sha, not a prefix);
`implementation_base_sha` = the FULL 40-hex commit of the reviewed real-run code (for the current reviewed code its short prefix is
`71e15c6` — or a reviewed successor commit); `allowed_ref_type="fold_contained_only"`; `allowed_refs` = EXACTLY the 30 canonical fold refs;
`allowed_seeds=[20260711,20260712,20260713]`; `selection_seed=20260711`; `forbid_final_external_refs=forbid_external_sites=`
`forbid_candidate_selection=forbid_external_read=True`; `run_id` = a fresh non-empty token; `statement` = the exact
`REQUIRED_STAGE1B_STATEMENT`.

**Runtime lock JSON** (`stage1_runtime_lock`): `stage="Stage-1B"`; `protocol_tag="acar-v5-protocol"`; `protocol_tag_target_sha` +
`implementation_base_sha` MATCHING the authorization; `run_id` matching; `device_kind`; `status="CAPTURED_AND_VERIFIED"` — captured on
the real SLURM/acar-v5 (torch) runtime.

**Full-build manifest (plan)**: `fold_contained_refs` = the 30 entries, each `{ref, disease, fold, seed, source_paths_by_cohort}`;
per disease the `source_paths_by_cohort` must be identical across that disease's refs and point at the frozen DEV cohort roots
(PD ds002778/ds003490/ds004584; SCZ ds003944/ds003947/ds004000/ds004367).

**What the real run then does**: fresh-run-root guard → gate → build ctx → for each of the 30 folds: eligibility (before split) →
FIT-only EEGNet+source-state train (deterministic) → freeze → ALL-fold label-free feature dump (schema .npz) → file-backed hashed
artifacts under `output_root/run_id/<ref-slug>/` → finalize barrier (count + global path uniqueness + canonical config sidecars +
feature-dump completeness) → persist `registry.json` + atomic `FINALIZED.json` bound to `registry_sha256`.

## Adversarial review hardening (post-implementation, before commit)
A 6-lens adversarial review, each finding independently refuted, surfaced 5 real findings (0 dismissed) → fixes:
- **admit_run TOCTOU (critical):** it hash-checked registry.json bytes then RE-READ the file via load_registry — a mid-call swap
  could return a different registry than the one hashed. Fixed: parse the SAME validated bytes (`load_registry_from_bytes`); no
  second read.
- **admit_run not fully fail-closed (high ×2):** malformed marker/registry JSON raised `JSONDecodeError` and a null `n_refs` raised
  `TypeError` instead of `RegistryIoError`. Fixed: all parsing wrapped → `RegistryIoError`.
- **Symlinked run root (critical):** `assert_fresh_run_root` used `os.path.isdir` (follows symlinks), so a symlinked run root →
  external dir passed and artifacts escaped (containment collapses under realpath). Fixed: reject `os.path.islink(run_root)`.
- **Silent registry-cleanup suppression (high):** on marker-write failure the registry.json removal was `except OSError: pass`.
  Fixed: the failure is surfaced in the raised `Stage1bFinalizeError` (registry.json may remain) instead of swallowed.

## Still forbidden in Stage-1B9 (unchanged)
real DEV read · OpenNeuro/Zenodo access · real training/registry population · SLURM submission · candidate selection · S1/S2/S3
robustness · held-out/external read · lockbox consumption.

## Next gate (the FIRST real-data step; separate authorization)
Stage-1B REAL run per the package above, on the pinned DEV cohorts, with the run bounds you specified (all 30 fold-contained builds;
FIT-only training; all-fold label-free dump; file-backed hashes; canonical registry.json + FINALIZED.json; no code changes / no
post-failure tuning without review; no Stage-2 selection / G1–G6 / S1–S3 / held-out / lockbox).
