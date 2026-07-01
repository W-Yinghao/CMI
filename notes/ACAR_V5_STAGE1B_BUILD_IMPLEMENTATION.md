# ACAR V5 — Stage-1B2 build-code implementation (CODE + SYNTHETIC TESTS ONLY; NO REAL RUN)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B2 implements the Stage-1B build *code path* and proves it cannot
read before the gate, cannot build fewer than 30 substrates, cannot contaminate FIT/CAL/EVAL, and cannot create artifacts outside
the registry contract — all on synthetic fakes. It performs NO real DEV read, NO training, NO artifact creation, NO registry
population from real artifacts, NO SLURM, NO external access. Pure stdlib on the orchestration path (the real reader/trainer,
which import torch/mne, are a separate later patch).

## Modules
- `substrate/dev_reader_contract.py` — DEV reader INTERFACE + `UnwiredDevReader` (fail-closed default) + `require_reader`. The real
  BIDS/mne reader is later; importing this reads nothing.
- `substrate/train_contract.py` — trainer INTERFACE + `UnwiredTrainer` + `require_trainer`. The real EEGNet/source-state trainer is
  later (lazy torch); importing this trains nothing.
- `substrate/stage1b_artifacts.py` — `validate_artifact_manifest`: each built fold substrate must carry the COMPLETE V5 registry
  hash set (encoder_state_dict / encoder_checkpoint_file / source_state_artifact / source_state_file / preprocessing_config /
  feat_dump, all 64-hex) + matching (ref, disease, fold, seed).
- `substrate/stage1b_runtime_capture.py` — `build_lock` (pure) + `capture_runtime_lock` (LAZY torch probe; not called at import or
  in tests; runs only at a real Stage-1B on the training node).
- `substrate/stage1b_build.py` — the ORCHESTRATOR `run_stage1b_build(plan, auth, lock, *, execute=False, dev_reader, trainer)`:
  1. calls `require_stage1b_full_build_ready(...)` FIRST — before any read/list/init;
  2. default `execute=False` → `STAGE1B_BUILD_DRYRUN` (validates the gate; reads nothing);
  3. `execute=True` requires a wired reader+trainer (CLI default = `Unwired*` → fails closed), builds EXACTLY the 30 fold refs,
     hands the trainer ONLY split FIT (train/val) subjects (CAL/EVAL never passed), and validates each artifact's hash set.
  CLI: default dry-run; `--execute` needs `--auth-json --runtime-lock-json --full-build-manifest-json` and still cannot read real
  data in Stage-1B2 (unwired reader/trainer).

## Extra hardening (Step 1B2)
`stage1b_full_build_manifest.validate_cohort_source_path` upgraded from substring containment to **path-segment exactness**
(`ds002778_old` ≠ `ds002778`).

## Guards (synthetic; part of `acar/v5/tests/run_all.py`)
`test_stage1b_build_default_dry_run_no_read` (execute=False reads nothing) ·
`test_stage1b_build_requires_full_gate_before_read` (gate before read; unwired reader blocks execute) ·
`test_stage1b_artifact_manifest_hash_set_complete` (complete registry hash set required) ·
`test_stage1b_outputs_exact_30_refs` (executed synthetic build → exactly 30 valid artifacts) ·
`test_stage1b_split_discipline_enforced` (trainer receives exactly split FIT train/val) ·
`test_stage1b_no_cal_eval_fit_contamination` (CAL/EVAL never reach the trainer) ·
`test_stage1b_no_selection_or_external_imports` (importing the build pulls no torch/mne/cmi/acar.v3/numpy; source has no
selection/robustness/external/real-loader calls).

Full v5 suite = 33 guard modules, green on py3.9 and py3.13.

## Still forbidden in Stage-1B2 (unchanged)
real DEV read · OpenNeuro/Zenodo access · EEGNet/spectral-z training · source-state fitting · embedding dump from real data ·
registry population from real artifacts · candidate selection · S1/S2/S3 robustness · held-out/external read · lockbox consumption.

## Next gate (separate authorization; NOT started)
**Stage-1B real run** = wiring the real BIDS/mne DEV reader + EEGNet/source-state trainer (their own reviewed patch), then a
tightly-scoped run authorization whose `implementation_base_sha` pins THIS reviewed build code, a captured runtime lock (SLURM,
acar-v5 env), and the real 30-substrate build via `run_stage1b_build(..., execute=True, dev_reader=<real>, trainer=<real>)`.
