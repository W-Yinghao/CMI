# ACAR V5 — Stage-1B5 numeric-seam + output/embedding semantics (CODE + SYNTHETIC/TEMP-FILE TESTS ONLY; NO REAL RUN)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B5 pins the numeric contract that the two remaining seams (mne DSP
read; EEGNet/source-state train + file emission) MUST satisfy, and the isolation semantics of the label-free embedding dump —
WITHOUT executing on real DEV data. The seams still `raise` ("wired at the Stage-1B run"); this stage makes the code prove, on
synthetic/temp-file inputs, that when those seams ARE wired they cannot violate the protocol (wrong montage, label leakage,
uncontained/duplicated output files, or a partial substrate table). Still NO real DEV read, NO OpenNeuro/Zenodo, NO BIDS index on
real paths, NO training/source-state fit on real data, NO embedding dump from real data, NO registry population from real
artifacts, NO SLURM, NO candidate selection, NO S1/S2/S3, NO held-out/external, NO lockbox.

## The seven points (per review of 6a68cc1)
1. **Gate-issued execution context / permit** — `stage1b_execution_context.Stage1BExecutionContext` is a frozen permit built ONLY
   by `build_execution_context(auth, lock, plan, *, output_root)`, i.e. AFTER the full-build gate. It pins run_id / protocol &
   impl SHAs / output_root / the exact 30 approved fold refs / the per-disease approved source paths. `RealBidsDevReader(context)`
   and `RealSubstrateTrainer(context)` now REQUIRE a context (None → error) and are built only by their factories from it; the
   reader refuses any source path `!=` the context-approved one. `run_stage1b_real_build` builds the context after the gate and
   hands it to `dev_reader_factory(ctx)` / `trainer_factory(ctx)`.
2. **Typed reader payload + schema guards** — `subject_windows.SubjectWindows` (frozen) is SIGNAL-ONLY (no label field);
   `validate_subject_windows` is fail-closed against the pinned preprocessing config: exact 19-channel order, 19 channels, 128 Hz,
   512-sample windows, positive n_windows, matching `preprocessing_config_sha256`, and rejects a namespaced raw id. `read_subject_windows`
   is contracted to return a validated `SubjectWindows` at the real run.
3. **Pinned DSP preprocessing config + hash** — `preprocessing_config.PREPROCESSING_CONFIG` is fixed in code (19ch 10-20 montage,
   128 Hz resample, 0.5–45 Hz bandpass, 4 s / 512-sample non-overlapping windows, average reference, per-trial z-score, fail-closed
   bad-channel policy) → canonical-JSON → `config_sha256()`. The real mne DSP must produce windows conforming exactly; the hash is
   carried on every `SubjectWindows` and recorded by the run.
4. **Pinned EEGNet / source-state training spec** — `training_config.TRAINING_CONFIG` fixes architecture EEGNet (19×512, 2
   classes), adam lr 1e-3, batch 64, ≤100 epochs, early-stopping on val_loss (patience 15), best-val checkpoint, deterministic,
   torch_threads 1, class-conditional-Gaussian-tangent source state, `trains_on = FIT_train_val_only`, `reads_labels =
   FIT_training_only` → `config_sha256()`. The real trainer must train under exactly this spec.
5. **FIT training view (labels) separated from label-free embedding-dump view** — `fit_dataset_view.AuthorizedFitDatasetView`
   gains a closure-backed `read_label` that is FIT-only (CAL/EVAL/unknown → refused). `embedding_dataset_view.AuthorizedEmbeddingDatasetView`
   (used to dump routing features over ALL fold subjects after the encoder is frozen) exposes read_windows ONLY — it has NO
   read_label method — so the feature dump physically cannot read labels. `stage1b_embedding_dump` validates dump records are
   label-free (rejects label/y/y_te/diagnosis/target/case_control/labels) and asserts the dump is driven by an
   `AuthorizedEmbeddingDatasetView` (a FIT view is rejected as a dump driver).
6. **Hardened file writer** — `stage1b_file_artifact_writer.write_artifact_from_files(..., output_root=...)`: rejects any artifact
   path that is a symlink (`os.path.islink`, checked BEFORE containment, so a symlink can't smuggle bytes even if it resolves
   inside), enforces realpath containment inside `output_root`, rejects duplicate/shared paths across the 6 artifacts, rejects
   missing/empty files, and IGNORES any trainer-reported hash (streams sha256 from the bytes). `run_stage1b_real_build` binds
   `output_root` into the writer via `functools.partial`.
7. **All-or-none registry population** — `stage1b_registry_populate.populate_registry` runs the exact-30 set check BEFORE any
   `register()`, so any wrong set (fewer than 30 / extra / mismatched ref) raises AND leaves the fresh registry with ZERO entries —
   a partial substrate table can never exist.

The two numeric seams remain unwired and raise a clear "wired at the Stage-1B run" error; heavy imports (mne/torch) stay lazy
(inside the seam functions), so the whole substrate package imports with NO heavy dependency (verified).

## Guards (synthetic/temp-file; part of `acar/v5/tests/run_all.py`)
execution-context-required · preprocessing-config-pinned · reader-window-payload-schema · training-config-pinned ·
embedding-dump-label-free · file-writer-output-root-containment · file-writer-rejects-symlink-escape ·
registry-population-all-or-none. Two Stage-1B4 tests updated for the new contracts (dev-reader now takes a context;
FIT-view public surface now includes `read_label`). Full v5 suite = **55 guard modules, green py3.9 + py3.13**.

## Still forbidden in Stage-1B5 (unchanged)
real DEV read · OpenNeuro/Zenodo access · BIDS indexing on real paths · EEGNet/spectral-z training on real data · source-state
fitting on real data · embedding dump from real data · registry population from real artifacts · SLURM submission · candidate
selection · S1/S2/S3 robustness · held-out/external read · lockbox consumption.

## Next gate (separate authorization; NOT started)
**Stage-1B real run** = wire the two remaining seams in a reviewed patch — mne DSP in `read_subject_windows` (→ validated
`SubjectWindows` per `preprocessing_config`); EEGNet/source-state training in `train_fold` per `training_config`, emitting the 6
output files consumed by the file writer + the label-free embedding dump — then a run authorization pinning
`implementation_base_sha` to that reviewed commit + a captured runtime lock (SLURM, acar-v5 env), invoking
`run_stage1b_real_build(...)` on the pinned DEV cohorts.
