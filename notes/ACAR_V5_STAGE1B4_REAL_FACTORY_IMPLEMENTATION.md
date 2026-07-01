# ACAR V5 — Stage-1B4 real reader/trainer + file-artifact-writer (CODE + SYNTHETIC/TEMP-FILE TESTS ONLY; NO REAL RUN)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B4 supplies the real-wiring seam (factories, file-backed writer) and
proves the remaining safety properties, WITHOUT executing on real DEV data. Still NO real DEV read, NO BIDS indexing on real
paths, NO training/source-state fit on real data, NO registry population from real artifacts, NO SLURM, NO external. Real
reader/trainer are constructed only by factories, only after the gate; their heavy imports (mne/torch) are lazy; the actual signal
read + numeric training remain seams wired at the authorized Stage-1B run.

## The five points (per review of c4b48a3)
1. **Production real-run entry accepts ONLY factories** — `stage1b_build.run_stage1b_real_build(plan, auth, lock, *,
   dev_reader_factory, trainer_factory, artifact_writer=<file writer>)` has NO `dev_reader`/`trainer` object params (a
   preconstructed object can't be handed to a real run → nothing is instantiated before the gate). `run_stage1b_build` keeps the
   object path for synthetic tests.
2. **Reader returns RAW ids** — `dev_reader_contract` doc corrected; `real_dev_reader.RealBidsDevReader.list_subjects` returns raw
   `sub-*` ids (plain dir listing); the subject index rejects a namespaced raw id (containing "/"), fail-closed.
3. **Dataset view exposes no raw roots** — `fit_dataset_view.AuthorizedFitDatasetView` is now CLOSURE-backed: the reader +
   cohort paths live in a closure, so the view object has NO `_reader`/`_cohort_paths` attribute; public surface = read_windows +
   allowed_subject_keys + reads. (Fixed the prior `... or True` no-op test.)
4. **File-backed artifact writer** — `stage1b_file_artifact_writer.write_artifact_from_files` streams sha256 over the trainer's
   output FILES (rejecting missing/empty), ignores any reported hash; the real build uses it by default.
5. **Real trainer reads only via the view** — `real_trainer.RealSubstrateTrainer.train_fold(disease, fold, seed,
   train_subject_keys, val_subject_keys, dataset_view)` accesses signal ONLY through `dataset_view.read_windows`; it performs no
   filesystem scan and calls no reader method directly.

Real reader/trainer heavy imports are LAZY (mne inside `read_subject_windows`, torch inside `train_fold`); both raise a clear
"wired at the Stage-1B real run" seam for the signal-read / numeric-training that must not run on real data yet.

## Guards (synthetic; part of `acar/v5/tests/run_all.py`)
real-run-requires-factories (+ end-to-end file-backed synthetic build) · dev-reader-returns-raw-ids · dataset-view-public-surface ·
file-artifact-hashes-streamed · real-trainer-no-raw-root-scan · real-factories-lazy-imports. Full v5 suite = 47 guard modules,
green py3.9 + py3.13.

## Still forbidden in Stage-1B4 (unchanged)
real DEV read · OpenNeuro/Zenodo access · BIDS indexing on real paths · EEGNet/spectral-z training on real data · source-state
fitting on real data · embedding dump from real data · registry population from real artifacts · SLURM submission · candidate
selection · S1/S2/S3 robustness · held-out/external read · lockbox consumption.

## Next gate (separate authorization; NOT started)
**Stage-1B real run** = wire the two remaining seams (mne DSP in `read_subject_windows`; EEGNet/source-state training + file
emission in `train_fold`) in a reviewed patch, then a run authorization pinning `implementation_base_sha` to that reviewed commit
+ a captured runtime lock (SLURM, acar-v5 env), invoking `run_stage1b_real_build(...)` on the pinned DEV cohorts.
