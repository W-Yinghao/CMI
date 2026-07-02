# ACAR V5 — Stage-1B8 numeric-backend implementation + final real-run hardening (CODE + SYNTHETIC/FIXTURE/TEMP-FILE TESTS ONLY; NO REAL RUN)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B8 implements the two numeric backends and closes the last real-run
integrity gaps, so a Stage-1B real run becomes "authorize + run". Everything is exercised on synthetic/fixture/temp-file inputs; torch
exists only on py3.13 (the backend fixture test skips cleanly on py3.9). Still NO real DEV read, NO OpenNeuro/Zenodo, NO real registry
population, NO SLURM, NO Stage-2 selection, NO S1/S2/S3, NO external/held-out.

## The six points (per review of 5051ada)
1. **Numeric backends implemented** — `eegnet_architecture.build_eegnet` (compact deterministic EEGNet, 19×512→2; `encode()` =
   pre-classifier features) + canonical pickle-free state (de)serialization; `source_state` (class-conditional Gaussian over encoder
   features: per-class means + ridged pooled covariance + priors, deterministic serialization); `torch_eegnet_backend.TorchEegnetBackend`
   — torch LAZY inside methods; `fit` is deterministic (seeded shuffle + `use_deterministic_algorithms`), trains only on FIT-train,
   early-stops on FIT-val, and returns the 4 model artifacts as DETERMINISTIC pickle-free bytes; `embed_from_artifacts` LOADS the
   frozen encoder from the handle's checkpoint FILE (no shared in-memory trainer state). `real_eegnet_trainer.TorchEegnetBackend` now
   re-exports this real backend. Verified on py3.13: same seed → byte-identical checkpoint + source-state; embed rows == n_windows,
   dim>0, finite. Guard: backend-fit-embed-fixture (torch; skips on py3.9).
2. **Channel-policy consistency** — `preprocessing_config` now declares `required_channels=all_19_canonical_present`,
   `extra_channel_policy=drop_non_canonical_after_required_19_present`, `duplicate_channel_policy=fail_closed`,
   `channel_output_order=canonical_pinned`; `real_mne_reader` matches exactly (missing→fail, duplicate→fail, extras dropped,
   permuted→canonical output). Guard: channel-policy-config-matches-reader.
3. **Raw sidecar hashing + provenance** — `raw_recording_manifest.build_manifest` resolves + hashes FORMAT SIDECARS (BrainVision
   .eeg/.dat via `DataFile=`, .vmrk via `MarkerFile=`, enforcing bare-basename same-dir; EEGLAB `.fdt`), fail-closing on
   missing/symlinked/escaping sidecars, and marks each entry primary/sidecar; `real_mne_reader.preprocess_subject` propagates the
   `manifest_sha256` into `SubjectWindows.provenance`, tying the payload to the exact raw bytes. Guards:
   raw-sidecar-manifest-brainvision-eeglab, raw-manifest-hash-propagated.
4. **Feature-dump completeness at finalize** — beyond schema, `stage1b_finalize._validate_feature_dumps` (via
   `stage1b_feature_dump_writer.load_feature_dump`) checks dump ref/disease/fold/seed match the ref, the subject set equals the
   expected fold subjects, each subject's `split_role` matches, and each subject's `window_id`s are exactly 0..n-1 (contiguous,
   unique). The expected manifest (`role_by_subject` from the authoritative split) is built per ref in `stage1b_build`. Guards:
   feature-dump-expected-subject-role-completeness, feature-dump-window-id-contiguous.
5. **Embedding rows == n_windows** — `dump_fold_embeddings` fail-closes unless each subject's embedding matrix is finite 2-D with
   `dim>0` and `rows == SubjectWindows.n_windows`. Guard: backend-embedding-rows-match-windows.
6. **Subject eligibility before split** — `subject_eligibility.assert_all_eligible` runs in `stage1b_build` BEFORE the split; every
   subject must have a resolvable control/case label, checked via the reader's `subject_label_resolvable(...)` which returns a
   BOOLEAN ONLY (the label value never leaves the reader → no leak into routing/dump; the embedding view still has no label path, the
   FIT view remains the only label-VALUE path). An ineligible subject aborts the build before any split/train/dump. Guard:
   subject-eligibility-before-split.

Every `acar.v5.substrate` module still imports with NO heavy dependency (torch/mne/numpy lazy inside functions — verified on both
interpreters), including the new torch/backend modules.

## Guards (synthetic/fixture/temp-file; part of `acar/v5/tests/run_all.py`)
backend-fit-embed-fixture (torch, skips on py3.9) · backend-embedding-rows-match-windows · channel-policy-config-matches-reader ·
raw-sidecar-manifest-brainvision-eeglab · raw-manifest-hash-propagated · feature-dump-expected-subject-role-completeness ·
feature-dump-window-id-contiguous · subject-eligibility-before-split. Full v5 suite = **79 guard modules, green py3.9 + py3.13**
(the torch backend fixture runs for real on py3.13).

## Adversarial review hardening (post-implementation, before commit)
A 6-lens adversarial review (numeric-backend / channel-policy / raw-sidecar / feat-dump-completeness / eligibility-labelfw /
gate-imports-completeness), each finding independently refuted, surfaced 3 real findings (3 dismissed as false positives) → fixes:
- **BrainVision header parsing didn't match mne (high, ×2):** the sidecar parser matched `DataFile=`/`MarkerFile=` case-sensitively
  and without tolerating whitespace around `=`, while mne's configparser is case-insensitive + whitespace-stripping — a partial
  sidecar miss is a silent audit gap. Fixed: parse case-insensitively + whitespace-tolerant (bare-basename/same-dir still enforced).
- **Finalize checked window contiguity but not the expected count (high):** a contiguous-but-too-few dump (e.g. only window 0 for a
  3-window subject) passed. Fixed: the dumper now reports the authoritative per-subject window count (from the reader's
  SubjectWindows), threaded via the orchestrator into `expected_by_ref`, and finalize verifies each subject's dump record count
  equals it (in addition to contiguity). Guard added.
Also hardened: the torch backend now raises if `set_deterministic(seed)` was not called before `fit()` (determinism contract).

## Still forbidden in Stage-1B8 (unchanged)
real DEV read · OpenNeuro/Zenodo access · BIDS indexing on real paths · EEGNet/source-state training on real data · embedding dump
from real data · registry population from real artifacts · SLURM submission · candidate selection · S1/S2/S3 robustness ·
held-out/external read · lockbox consumption.

## Next gate (separate authorization; the FIRST real-data step)
**Stage-1B real run** = a run authorization pinning `implementation_base_sha` to this reviewed commit + a CAPTURED runtime lock
(SLURM, acar-v5 env with torch), invoking `run_stage1b_real_build(plan, auth, lock, output_root=, dev_reader_factory=
make_real_dev_reader, trainer_factory=make_real_trainer, dumper_factory=make_real_embedding_dumper)` on the pinned DEV cohorts. The
numeric backends are now implemented; the real run reads DEV EEG, trains the 30 fold substrates, dumps the label-free features, and
populates the registry behind the finalize barrier. This is the first step that touches real data and requires its own explicit
authorization.
