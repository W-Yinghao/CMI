# ACAR V5 — Stage-1B7 backend-state + feat-dump schema + raw-BIDS/finalize hardening (CODE + SYNTHETIC/FIXTURE/TEMP-FILE TESTS ONLY; NO REAL RUN)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B7 makes the Stage-1B6 two-phase build path safe to actually run and
safe for Stage-2 to consume: the embedding dump is provably produced by the SAME frozen substrate, the dump has a pinned parseable
label-free schema, raw discovery is raw-BIDS-only and boundary-safe, finalize is atomic, and FIT records are validated before the
numeric backend. All exercised on synthetic/fixture/temp-file inputs; the two numeric backends (`TorchEegnetBackend.fit` /
`embed_from_artifacts`) remain NotImplemented seams. Still NO real DEV read, NO OpenNeuro/Zenodo, NO real training/registry
population, NO SLURM, NO Stage-2 selection, NO S1/S2/S3, NO external/held-out.

## The five points (per review of 495e109)
1. **Frozen-artifact binding.** `real_eegnet_trainer.FrozenSubstrateHandle.from_train_result(...)` builds a handle from the trainer's
   output files (encoder checkpoint / source-state / preprocessing_config / training_config), requiring each to be an existing file,
   and `assert_matches(disease, fold, seed)` binds it to the ref. `dump_fold_embeddings(...)` builds+matches the handle and calls
   `backend.embed_from_artifacts(windows_by_subject, frozen, training_config)` — the embedding is loaded FROM the frozen artifacts of
   the same ref, not from an incidentally shared backend object. The 4 substrate/config hashes in the dump are computed FROM the
   handle's files. A mismatched ref or a missing artifact fails closed. Guard: backend-uses-frozen-artifacts-for-embedding.
2. **Pinned, parseable, label-free feature-dump schema.** `feature_dump_schema` (SCHEMA_VERSION `ACAR_V5_STAGE1B_FEAT_DUMP_V1`;
   header provenance ref/disease/fold/seed + preprocessing/training/encoder-checkpoint/source-state sha256; parallel record arrays
   subject_key/split_role/window_id/embedding; FORBIDDEN label-like fields) + `stage1b_feature_dump_writer` (writes a single `.npz`
   = the hashed `feat_dump_path`, `np.savez` numeric/str only, `np.load(allow_pickle=False)`, validate-before-write + round-trip
   parse). Fail-closed on forbidden field / empty / non-finite embedding / unknown split role / wrong schema version.
   `dump_fold_embeddings` writes the dump via this schema; every record carries its split role (train/val/cal/eval). Guards:
   feature-dump-schema-parseable-label-free, feature-dump-includes-all-fold-split-roles.
3. **Raw-BIDS discovery + boundary safety.** `raw_recording_manifest.discover_raw_recordings` searches ONLY `<sub>/eeg` and
   `<sub>/ses-*/eeg`, ignores subject-root / derivatives / sourcedata files, rejects symlinked recordings and excluded-component
   paths, and `build_manifest` records a deterministic hashed file manifest. `real_mne_reader.preprocess_subject` windows EACH
   recording INDEPENDENTLY (`_windows_from_raw`) then `np.concatenate`s the WINDOW arrays — it never concatenates raws, so no window
   spans a recording boundary. Guards: raw-bids-discovery-excludes-derivatives, multi-recording-no-cross-boundary-windows.
4. **Atomic finalize.** `stage1b_finalize.write_finalized_marker` writes the marker to `FINALIZED.json.tmp` then `os.replace` →
   `FINALIZED.json` (atomic, fsync, temp cleanup). `finalize_and_populate` populates then writes the marker; a marker-write failure
   AFTER population rolls the registry back to empty — so a FINALIZED marker exists IFF the registry is fully populated.
   `populate_registry` is all-or-none (Stage-1B6 rollback). Guard: finalize-marker-atomicity.
5. **FIT-record validation.** `real_eegnet_trainer.train_encoder_and_source_state` validates every FIT record BEFORE `backend.fit`:
   each windows is a validated `SubjectWindows`, each label ∈ {0,1} (bool rejected), train/val subject-disjoint, no duplicate keys,
   val non-empty, keys canonical (`<disease>/<cohort>/<raw>`); it still emits no feat_dump. Guard: fit-record-validation.

Label firewall unchanged and re-checked: the embedding view still requires a windows-only reader (no `read_label` reachable, incl.
via a bound method's `__self__`); the dumper reads windows only and the schema forbids label fields. Every acar.v5.substrate module
still imports with NO heavy dependency (numpy/mne/torch lazy inside functions — verified).

## Guards (synthetic/fixture/temp-file; part of `acar/v5/tests/run_all.py`)
backend-uses-frozen-artifacts-for-embedding · feature-dump-schema-parseable-label-free · feature-dump-includes-all-fold-split-roles ·
raw-bids-discovery-excludes-derivatives · multi-recording-no-cross-boundary-windows · finalize-marker-atomicity · fit-record-validation.
Existing guards updated for the new dumper signature (role_by_subject), the schema `.npz` dump, and raw-BIDS-only discovery. Full v5
suite = **71 guard modules, green py3.9 + py3.13**.

## Adversarial review hardening (post-implementation, before commit)
A 6-lens adversarial review (frozen-artifact / feat-dump-schema / raw-BIDS-boundary / finalize-atomicity / fit-validation-labelfw /
gate-imports-completeness), each finding independently refuted, surfaced 5 real findings (1 false positive dismissed) → 3 distinct
fixes:
- **feat_dump not schema-validated before registration (high):** the file writer only hashed the dump. Fixed: the finalize BARRIER
  now parses every ref's `feat_dump.npz` with the pinned schema (dumper-agnostic — even a non-standard dumper's output is validated)
  and checks its provenance ref before any register; a malformed dump → registry empty + no marker. Guard added.
- **symlinked-directory discovery bypass (high, ×2):** a symlinked `ses-01 → ../derivatives` (or symlinked `eeg/`) let files from
  excluded locations be admitted, because `os.path.normpath` doesn't resolve symlinks. Fixed: `_eeg_dirs` rejects symlinked `eeg/`
  and session directories, and the excluded-component check now uses `os.path.realpath`. Guard added.
- **UnwiredEmbeddingDumper signature mismatch (high):** it lacked the new `role_by_subject` parameter → TypeError on the unwired CLI
  path. Fixed to match the dumper contract.

## Still forbidden in Stage-1B7 (unchanged)
real DEV read · OpenNeuro/Zenodo access · BIDS indexing on real paths · EEGNet/spectral-z training on real data · source-state
fitting on real data · embedding dump from real data · registry population from real artifacts · SLURM submission · candidate
selection · S1/S2/S3 robustness · held-out/external read · lockbox consumption.

## Next gate (separate authorization; NOT started)
**Stage-1B real run** = implement the two numeric backends — `TorchEegnetBackend.fit` (EEGNet + source-state under training_config,
emitting the 4 model files) and `TorchEegnetBackend.embed_from_artifacts` (load the frozen encoder/source-state from the handle's
files, embed each SubjectWindows) — in a reviewed patch, then a run authorization pinning `implementation_base_sha` to that reviewed
commit + a captured runtime lock (SLURM, acar-v5 env), invoking `run_stage1b_real_build(...)` with the three real factories on the
pinned DEV cohorts.
