# ACAR V5 — Stage-1B15 production reader repair-staging wiring (CODE + SYNTHETIC/FIXTURE TESTS ONLY; NO REAL RUN)

```
Stage-1B real-run authorization pinned to 3fe8852 was SUPERSEDED before launch.
Reason: the production reader did not pass staging_dir into preprocess_subject, so the reviewed BrainVision repair was NOT active in
        the real build path (ds003944/ds003947 marker-less + ds004000 sub-042 broken-pointer would crash at read).
Stage-1B15 wires the production reader to the reviewed repair path using an explicit, validated, per-run repair staging root.

No real build is authorized by this patch.
```

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B15 closes the single run-dooming gap the pre-launch readiness preflight
found (see `ACAR_V5_STAGE1B_REALRUN_BLOCKED_STAGING_WIRING.md`): the production DEV reader called `preprocess_subject(...)` WITHOUT a
`staging_dir`, so the Stage-1B12/1B13/1B14 BrainVision header repair (marker synth / pointer rewrite / channels.tsv ordinal rename) was
never invoked in the real build. **Code + synthetic/fixture/temp-file tests only; no real DEV read, no signal load, no DSP, no training,
no embedding, no artifact/registry, no SLURM.**

## 1. Explicit, validated, per-run repair staging root (`substrate/stage1b_repair_staging.py`, pure/stdlib)
The production real build now takes an EXPLICIT `repair_staging_root` — an ephemeral SCRATCH dir where the reviewed BrainVision header
repair materializes repaired headers/markers. It is validated **fail-closed** (after the gate, before any factory/read) and created
only then:
- a non-empty **absolute** path; **not a symlink**;
- **never overlaps** (inside / equal / containing, on `os.sep` component boundaries) any approved raw cohort source path — scratch
  never touches the raw DEV tree;
- **never overlaps** the run's hash-bound artifact root `output_root/run_id` — scratch is never part of the registered package;
- **absent OR an empty real directory** at launch.
It is SCRATCH ONLY: never a registered artifact, never in `registry.json` / `FINALIZED.json`.

## 2. Production entry requires it; the context carries it
`run_stage1b_real_build(plan, authorization, runtime_lock, *, output_root, repair_staging_root, dev_reader_factory, trainer_factory,
dumper_factory)` now **requires** `repair_staging_root` (raises `Stage1bBuildError` if falsy — no silent default temp dir). In
`run_stage1b_build`'s factory branch, AFTER `require_stage1b_full_build_ready` and the fresh-run-root check, the root is validated +
created (`RS.create_repair_staging_root`, with the approved raw source paths gathered from the plan) and threaded into a new
`Stage1BExecutionContext.repair_staging_root` field handed to the factories.

## 3. Both production read paths use the repair (via a per-call staging subdir)
`real_dev_reader` — BOTH `RealBidsDevReader.read_subject_windows` (FIT training reads) and the label-incapable
`WindowsOnlyReader.read_subject_windows` (label-free dump reads) — delegate to a shared `_read_windows_with_repair(ctx, ...)` which:
- **fail-closes** (`RealReaderError`) if the gate-issued context carries no valid repair staging root;
- creates a **fresh PER-CALL** `tempfile.TemporaryDirectory(dir=repair_staging_root)` (so the same subject read across folds / seeds /
  FIT+dump never collides on repaired-header filenames), passes it as `staging_dir` to `real_mne_reader.preprocess_subject(...)`, and
  **removes it after the read** (the returned SubjectWindows holds the windows in memory; the repaired headers are scratch).

## 4. Label firewall unchanged
`WindowsOnlyReader` still carries ONLY the (label-free) execution context — no `read_subject_label`, no reference to the label-capable
`RealBidsDevReader`; the context itself has no label capability. The embedding view still fails closed for a label-capable reader and,
built from the staged windows-only facade, reaches no label through its `read_windows` closure.

## Verification
Full v5 suite = **154 guard modules, green py3.9 + py3.13** (13 new Stage-1B15 suites, incl. an END-TO-END guard that reads a
generic-ordinal ds003944 BrainVision fixture through the PRODUCTION `RealBidsDevReader.read_subject_windows` and gets a validated
canonical SubjectWindows with the channels.tsv rename recorded + the raw tree untouched — the exact class that crashed the pre-wiring
build). Every `acar.v5.substrate` module imports with **NO** heavy dependency (mne stays lazy inside `preprocess_subject`). A two-lens
adversarial review (staging-validation safety + reader wiring; label-firewall + purity + completeness) found **NO blocker/high/medium
defect** — the repair-cannot-be-bypassed, staging-never-leaks-into-package, per-call-cleanup, label-firewall, purity, and gate-first /
fresh-run-root / all-or-none ordering properties all hold. Two low items were addressed: (a) the in-memory `raw_manifest_sha256`
provenance for marker-synth cohorts embedded the ephemeral staging path (never persisted, but non-reproducible) → now uses the stable
synthesized-marker content hash (`generated_marker_sha256`), with a reproducibility guard; (b) added an end-to-end guard that a real
build ABORTS fail-closed (`RepairStagingError`) on an invalid staging root before any factory/read.

## Still forbidden in Stage-1B15 (unchanged)
real DEV read · mne preload on real recordings · get_data/load_data on real recordings · DSP on real data · training · embedding dump ·
registry population · SLURM · Stage-2 · S1/S2/S3 · external/held-out · lockbox · the real build.

## Next gates (separate authorizations)
No new full real-data channel preflight is required — Stage-1B14P already proved data ↔ repair-policy compatibility across all 539
recordings; the missing piece was production reader wiring, now proven by synthetic guards. If Stage-1B15 is reviewed clean, a NEW
Stage-1B real-run authorization pinned to the reviewed Stage-1B15 commit can be issued, with the production call adding
`repair_staging_root=REPAIR_STAGING_ROOT` (an explicit scratch root, e.g. `/scratch/$USER/acar_v5_stage1b_repair/<run_id>`). The
post-run report must additionally record: `repair_staging_root`, whether it was empty at launch, and whether it was cleaned after
success. The `3fe8852` and `0ab40ec` real-run authorizations remain superseded.
