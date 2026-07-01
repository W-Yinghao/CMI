# ACAR V5 — Stage-1B3 real-wiring implementation (CODE + SYNTHETIC/FAKE TESTS ONLY; NO REAL RUN)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B3 implements the architecture that makes the eventual real build
safe-by-construction — proving (on synthetic fakes) that even once a real reader/trainer is wired, it cannot init before the gate,
cannot let inconsistent cohort paths change the subject universe, cannot collapse subject ids, cannot touch CAL/EVAL, and cannot
forge artifact hashes. Still NO real DEV read, NO training, NO real artifact creation, NO registry population from real artifacts,
NO SLURM, NO external. The orchestration path is pure stdlib; the only heavy import (torch) is LAZY inside the capture probe.

## The five hardenings
1. **Factory / gate-before-instantiation** (`stage1b_build.run_stage1b_build(..., dev_reader_factory, trainer_factory)`): the
   gate runs FIRST; the real reader/trainer are constructed by their factories ONLY after it passes (no pre-gate model init / GPU
   probe / BIDS scan). Object params remain for the synthetic test path; factories XOR objects.
2. **Per-disease source-path consistency** (`stage1b_full_build_manifest.validate_source_paths_consistent`, wired into the
   full-build gate): all 15 fold/seed refs of a disease must declare the IDENTICAL `source_paths_by_cohort`, so the subject
   universe (listed once per disease) can never disagree with a later ref's paths.
3. **Canonical SubjectKey index** (`subject_index.py`): raw per-cohort ids → `"{disease}/{cohort}/{raw}"`; a raw id shared across
   cohorts yields DISTINCT keys (no collapse); duplicate (cohort, raw) or key → fail-closed. Splits run on canonical keys.
4. **Authorized FIT dataset view** (`fit_dataset_view.py`): the trainer receives ONLY the FIT (train∪val) keys + a view whose
   `read_windows(key)` refuses any CAL/EVAL/unknown key — it never gets raw cohort roots. "CAL/EVAL never passed" is upgraded to
   "CAL/EVAL cannot be read".
5. **Computed (not trusted) artifact hashes + registry population** (`stage1b_artifact_writer.py`, `stage1b_registry_populate.py`):
   the writer computes the 6 registry hashes from the trainer's output BYTES (ignoring any reported hash strings); the registry is
   populated exactly once per canonical ref (no silent overwrite), 30 entries total.

New trainer contract signature: `train_fold(disease, fold, seed, train_subject_keys, val_subject_keys, dataset_view)` → raw build
output (bytes payloads). Real reader/trainer remain a later patch (`Unwired*` defaults fail closed; CLI `--execute` cannot read).

## Guards (synthetic; part of `acar/v5/tests/run_all.py`)
factory-gate-before-instantiation · source-paths-consistent-across-refs · subject-key-canonicalization · duplicate-subject-keys-
rejected · dataset-view-rejects-cal-eval-reads · registry-population-exact-30 · artifact-hashes-computed-not-trusted ·
real-wiring-imports-lazy. Full v5 suite = 41 guard modules, green on py3.9 and py3.13.

## Still forbidden in Stage-1B3 (unchanged)
real DEV read · OpenNeuro/Zenodo access · EEGNet/spectral-z training · source-state fitting · embedding dump from real data ·
registry population from real artifacts · candidate selection · S1/S2/S3 robustness · held-out/external read · lockbox consumption.

## Next gate (separate authorization; NOT started)
**Stage-1B real run** = a reviewed patch supplying the real BIDS/mne DEV reader factory + EEGNet/source-state trainer factory
(lazy heavy imports), then a run authorization pinning `implementation_base_sha` to THAT reviewed commit + a captured runtime lock
(SLURM, acar-v5 env), invoking `run_stage1b_build(..., execute=True, dev_reader_factory=<real>, trainer_factory=<real>)`.
