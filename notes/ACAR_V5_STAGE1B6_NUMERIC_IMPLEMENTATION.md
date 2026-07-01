# ACAR V5 — Stage-1B6 numeric implementation + embedding-dump orchestration (CODE + SYNTHETIC/FIXTURE/TEMP-FILE TESTS ONLY; NO REAL RUN)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B6 turns the pinned Stage-1B5 *contracts* into an auditable numeric
CODE PATH and, crucially, wires the mandatory **two-phase** substrate build (FIT-only training → freeze → ALL-fold label-free
feature dump) that Stage-2 selection depends on — WITHOUT executing on real DEV data. The two heavy operations (mne DSP; EEGNet /
source-state fit + embedding) are driven through an INJECTABLE numeric backend / mne adapter, so a synthetic fake exercises the
entire path deterministically with no torch/mne and no real read. Still NO real DEV read, NO OpenNeuro/Zenodo, NO BIDS index on
real paths, NO training/source-state fit on real data, NO real registry population, NO SLURM, NO selection, NO S1/S2/S3, NO
held-out/external, NO lockbox.

## The eight points (per review of ec1452c)
1. **Two-phase embedding-dump orchestration** (`stage1b_embedding_orchestrator.build_fold_raw`): Phase A runs `trainer.train_fold`
   over ONLY the FIT (train∪val) keys via `AuthorizedFitDatasetView`; it returns the 5 encoder/source-state/config artifacts and is
   asserted to emit NO feat_dump. Phase B builds an `AuthorizedEmbeddingDatasetView` over ALL fold subjects (train∪val∪cal∪eval —
   which is every disease subject) and hands it to a SEPARATE `dumper.dump_embeddings`, which reads windows only (no read_label) and
   emits feat_dump. The two outputs are merged. CAL/EVAL are therefore reachable ONLY through the label-free embedding view; the
   trainer is structurally unable to emit the feature dump. A dumper is REQUIRED for `execute=True` (contract + `UnwiredEmbeddingDumper`
   default in the CLI). Guards: train-then-dump-order (+ real-seam end-to-end) and embedding-dump-all-fold-subjects.
2. **Typed SIGNAL payload** (`subject_windows.SubjectWindows.windows`): the payload now carries the ACTUAL array
   [n_windows,19,512]; `validate_subject_windows` fail-closes on missing / wrong shape / non-float dtype / NaN / Inf (numpy imported
   LAZILY inside the validator only). Guard: subject-windows-actual-payload.
3. **Real mne DSP seam** (`real_mne_reader`): lazy mne; deterministic recording discovery (sorted, recursive); `raw_to_windows`
   applies the pinned pipeline (pick+reorder to canonical 19ch → average reference → 0.5–45 Hz → resample 128 → 4 s/512 non-overlap
   windows → per-trial z-score, microvolt) and returns a validated SubjectWindows with NO label. It is driven in tests by a FAKE
   mne-Raw / fake-mne adapter (no real mne, no real DEV). `real_dev_reader.read_subject_windows` now delegates to it. Interpretation
   (reviewable): a recording must CONTAIN all 19 montage channels (missing → fail-closed); they are picked+reordered to canonical
   order (a permuted input yields canonical output); extra non-montage channels are dropped by the pick. Guard: mne-reader-fixture.
4. **Pinned label source** (`stage1b_label_source`): only {control:0, case:1} (with a small alias set); unknown / missing / empty /
   ambiguous / non-string → `LabelSourceError` (fail-closed, never defaulted); a deterministic stdlib participants.tsv reader
   (fail-closed on missing file/subject, no group column, duplicate subject). `real_dev_reader.read_subject_label` delegates to it and
   is reachable ONLY via `AuthorizedFitDatasetView.read_label` (the embedding view has no label path). Guard: label-loading-fit-only.
5. **Real EEGNet/source-state trainer** (`real_eegnet_trainer` + `real_trainer`): `real_trainer.RealSubstrateTrainer` is the VIEW
   BOUNDARY — it reads signal+labels ONLY via `dataset_view.read_windows`/`read_label`, performs no filesystem scan and no direct
   reader call, computes the per-ref output dir via the layout helper, and delegates the ALREADY-READ FIT data to
   `real_eegnet_trainer.train_encoder_and_source_state` (lazy torch in the backend; deterministic+seeded), which emits the 4 model
   files + the pinned preprocessing_config + a training_config sidecar into the per-ref dir and does NOT emit feat_dump. The numeric
   backend is injectable (a `FakeEegnetBackend` drives it in tests); the default `TorchEegnetBackend` leaves the actual EEGNet fit /
   embedding as the seam wired at the authorized real run. `RealEmbeddingDumper` is the label-free dump counterpart.
6. **Per-ref output containment** (`stage1b_output_layout` + `stage1b_file_artifact_writer`): with a run_id, every artifact file must
   be a NON-symlink regular file under `output_root/run_id/safe_ref_slug(ref)` (islink checked before containment; realpath boundary;
   `safe_ref_slug` rejects non-canonical refs / unsafe chars); no path may be reused within a ref, and GLOBAL cross-ref uniqueness is
   enforced at finalize. Guards: per-ref-output-containment, global-artifact-path-uniqueness.
7. **Config files validated, not just hashed** (`stage1b_finalize`): the preprocessing_config file content must equal
   `preprocessing_config.canonical_json()` and its recorded hash must equal `config_sha256()`; the training_config sidecar must equal
   `training_config.canonical_json()`. `training_config_sha256` is recorded in registry META (added additively via
   `populate_registry(extra_meta_by_ref=...)`; it can NEVER override the six registry hash fields, which live in `hashes`). Guard:
   config-files-canonical.
8. **Finalize BARRIER, all-or-none** (`stage1b_finalize.finalize_and_populate`): exactly-30 count check → global path uniqueness →
   canonical config sidecars, validated IN FULL BEFORE any `registry.register()`; only then populate (itself all-or-none) and write a
   FINALIZED marker (file-backed builds only). Any pre-populate failure leaves the registry EMPTY and writes no marker. `stage1b_build`
   now routes every fold through the orchestrator and finalizes through this barrier. Guard: finalize-barrier-before-registry.

The two numeric seams still return through the injectable backend/adapter and the default (torch/mne) backends leave the true model
fit / DSP read as the seam wired at the authorized Stage-1B run; every acar.v5.substrate module still imports with NO heavy
dependency (torch/mne/numpy lazy inside functions — verified).

## Guards (synthetic/fixture/temp-file; part of `acar/v5/tests/run_all.py`)
subject-windows-actual-payload · mne-reader-fixture-contract · label-loading-fit-only · train-then-label-free-dump-order (incl real
seam) · embedding-dump-all-fold-subjects · per-ref-output-containment · global-artifact-path-uniqueness · config-files-canonical ·
finalize-barrier-before-registry. Existing Stage-1B2/1B3/1B4/1B5 guards updated for the new trainer↔dumper split + per-ref layout +
delegated seams (dev-reader delegates to real_mne_reader; FIT-view surface += read_label; factory/real-run paths take a dumper
factory; lazy-import guard points at the seam modules). Full v5 suite = **64 guard modules, green py3.9 + py3.13**.

## Adversarial review hardening (post-implementation, before commit)
A 6-lens adversarial review (label-firewall / containment / finalize-all-or-none / gate-ordering / config-DSP / completeness), each
finding independently refuted, surfaced 5 real defense-in-depth gaps — all fixed (gate-ordering and DSP lenses came back clean):
- **Label-firewall closure leak (critical):** the label-free embedding view is closure-backed, so an introspecting dumper could
  reach the captured reader via `read_windows.__closure__` and call `read_subject_label`. Fixed: the embedding view now FAILS CLOSED
  unless handed a WINDOWS-ONLY reader (no `read_subject_label`); readers expose `windows_only()` (a facade bound only to the
  execution context — no path to labels, even via a bound method's `__self__`); the orchestrator builds the embedding view from it. A
  guard walks the view's closure and asserts no captured object (incl `__self__`) exposes `read_subject_label`.
- **Partial registry population (critical):** a mid-loop `register()` failure could leave a partial registry, breaking the
  all-or-none promise. Fixed: `populate_registry` is now ATOMIC (tracks added refs and rolls them back on any exception).
- **Training_config sidecar containment (high):** the sidecar was content/hash-validated but not containment-checked. Fixed:
  finalize now asserts the training_config sidecar (and preprocessing_config) are non-symlink files under the per-ref output dir.
- **Reader contract missing read_subject_label (high, ×2):** `require_reader` didn't require `read_subject_label` and
  `UnwiredDevReader` lacked it (AttributeError instead of fail-closed). Fixed: added to the contract + a fail-closed unwired method.

## Still forbidden in Stage-1B6 (unchanged)
real DEV read · OpenNeuro/Zenodo access · BIDS indexing on real paths · EEGNet/spectral-z training on real data · source-state
fitting on real data · embedding dump from real data · registry population from real artifacts · SLURM submission · candidate
selection · S1/S2/S3 robustness · held-out/external read · lockbox consumption.

## Next gate (separate authorization; NOT started)
**Stage-1B real run** = fill the two numeric backends (mne DSP already structured in `real_mne_reader.raw_to_windows`; EEGNet /
source-state fit + frozen-encoder embedding in `TorchEegnetBackend.fit`/`embed`) in a reviewed patch, then a run authorization
pinning `implementation_base_sha` to that reviewed commit + a captured runtime lock (SLURM, acar-v5 env), invoking
`run_stage1b_real_build(...)` on the pinned DEV cohorts (factories = make_real_dev_reader / make_real_trainer /
make_real_embedding_dumper).
