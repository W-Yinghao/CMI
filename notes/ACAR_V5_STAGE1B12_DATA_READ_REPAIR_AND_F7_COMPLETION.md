# ACAR V5 — Stage-1B12 data-read repair + conditional ds004367 F7 completion (CODE + SYNTHETIC/FIXTURE TESTS ONLY; NO REAL RUN)

```
Stage-1B11P failed at three data-read/channel-integrity classes.
Labels passed completely (456/456).
Stage-1B12 is a reviewed pre-run data-read repair / completion amendment.

Authorized:
  ds003944/ds003947 marker-less BrainVision minimal read repair (synthesize a minimal marker; never infer events)
  ds004000 sub-042 exact broken-internal-pointer repair (repoint at the existing BIDS sibling files)
  ds004367 conditional F7 interpolation (only with the F7-0/F7-1 duplicate-variant pattern)

Not authorized:
  cohort dropping · subject dropping · reduced-channel substrate · modern-montage re-pin · F7-0/F7-1 remapping ·
  keep-first de-duplication · general BrainVision repair · the real build
```

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B12 fixes the three header-READABILITY / channel-integrity defects the
Stage-1B11P metadata/header/geometry preflight surfaced (see `ACAR_V5_STAGE1B11P_METADATA_GEOMETRY_PREFLIGHT_RESULT.md`) via a
REVIEWED, whitelisted repair + completion layer. **No real DEV signal load, no DSP on real data, no interpolation on real signals, no
training, no embedding, no registry, no SLURM.** The raw signal is NEVER modified, copied, or read by the repair.

## 1. BrainVision READ-REPAIR (`substrate/brainvision_read_repair.py`; pure/stdlib — no mne/numpy)
An EPHEMERAL, audited *header* repair so the pinned mne (1.12.1) can OPEN a header-defective recording. Two exact, whitelisted modes;
everything else fails closed (`plan_repair` returns `None`, and the reader opens the original header — which mne rejects if it is
genuinely unreadable):
- **`missing_markerfile_minimal_vmrk`** — cohorts `{ds003944, ds003947}` only, when the `.vhdr` declares a `DataFile` but **no
  `MarkerFile`** and there is **no `.vmrk`** on disk (the exact 143-recording defect). Synthesize a MINIMAL marker file (a single
  `New Segment` at position 1 — **no** task/stimulus/event inference) and a repaired header pointing `DataFile` at the ORIGINAL `.eeg`
  and `MarkerFile` at the synthesized marker (both absolute paths).
- **`broken_internal_pointer_rewrite`** — `ds004000`/`sub-042`'s TWO exact recordings
  (`sub-042_task-{proposer,responder}_run-1_eeg.vhdr`) only, when the declared `DataFile`/`MarkerFile` do not exist → a repaired
  header pointing at the EXISTING BIDS sibling data (`<stem>.eeg|.dat`) + marker (`<stem>.vmrk`). **No** marker synthesized.

Fail-closed invariants (enforced + tested): no plan for any other cohort/subject/recording (no general repair); repaired files land
ONLY under the caller's ephemeral **staging dir** (a staging dir equal to / inside / containing the raw recording dir is rejected —
never write into the raw tree); targets must be existing NON-symlink regular files; every repair emits a **manifest**
(`original_header_sha256`, `repaired_header_sha256`, `generated_marker_sha256`|None, `repair_mode`, cohort/subject/recording, targets,
policy hash) and `assert_manifest_consistent` re-verifies the on-disk hashes so the reader consumes the repaired header ONLY after the
manifest validates (a tampered header is rejected).

## 2. Conditional ds004367 F7 completion (`montage_completion.py` + `preprocessing_config`)
`allowed_missing_by_cohort` gains `ds004367: [F7]`, but F7 is **conditional**: `conditional_montage_completion =
{ds004367: {channel: F7, require_variant_names: [F7-0, F7-1]}}`. F7 is interpolated for a ds004367 recording only when the raw header
carries BOTH `F7-0` and `F7-1` and no canonical `F7`; otherwise **fail-closed** (being on the whitelist is not, by itself,
authorization). `F7-0`/`F7-1` are non-canonical (no `standard_1020` position) → they normalize to `None` and are **never**
aliased / kept-first / averaged into `F7`; the alias layer reports canonical F7 MISSING, and F7 is interpolated from good-position
donors like any other whitelisted missing channel. A new **drop-unknown-position-channels** step removes non-canonical channels
without a finite standard position (F7-0/F7-1, GSR/ECG…) BEFORE `interpolate_bads`, so they cannot poison the interpolation matrix
with NaN coordinates (donor_policy `unknown_position_channels_ignored`); it can never drop a canonical channel or the channel being
interpolated, and the ds004584-Pz / ds004000-F3F4P3P4 paths are unchanged. Output stays the canonical old-10-20 19 channels; F7
interpolation is audited in `SubjectWindows.provenance` + `.montage_completion`. Policy version bumped to
`ACAR_V5_STAGE1B12_MONTAGE_COMPLETION_V2`.

## 3. Provenance + feature-dump schema **V3** (`feature_dump_schema.py` + writer + trainer)
`SubjectWindows` gains a `read_repair` field (`{repaired:[recording…], by_recording:[manifest…]}`); provenance now records
`brainvision_read_repair_policy_sha256` + the read-repaired list. Feature-dump `SCHEMA_VERSION = ACAR_V5_STAGE1B_FEAT_DUMP_V3`: the
header adds `brainvision_read_repair_policy_sha256`, `raw_header_repair_manifest_sha256` (both hex64), and
`brainvision_read_repair_by_recording` (JSON map `subject::recording -> {repair_mode, *_sha256}`) alongside the V2
montage-completion map. It stays LABEL-FREE (the nested-key scan covers the repair map); the writer defaults the new fields (empty map
+ sentinel manifest hash); the trainer emits the per-recording map + an order-independent manifest-set hash from
`SubjectWindows.read_repair`. A V2-shaped dump (missing the V3 fields) is rejected. So Stage-2 can distinguish native /
markerfile-synthesized / pointer-repaired / F7-interpolated (ds004367) / Pz-interpolated (ds004584) / F3F4P3P4-interpolated
(ds004000) recordings.

## 4. Reader wiring (`real_mne_reader.preprocess_subject(..., staging_dir=...)`)
The default `staging_dir=None` path is byte-for-byte the previous behavior (`build_manifest` + `_read_raw`). When a `staging_dir` is
given, the reader discovers recordings by listing (`discover_raw_recordings`), applies the reviewed read-repair per recording into
staging, opens the repaired-or-original header, audits the original header + effective data/marker targets, aggregates
`read_repair`, and STILL validates the final canonical SubjectWindows. The real-run invocation (which passes a staging dir) is a
SEPARATE later authorization.

## Verification
Full v5 suite = **119 guard modules, green py3.9 + py3.13** (12 new Stage-1B12 suites; repair tests use synthetic BrainVision
triplets + real mne header reads; F7 tests use synthetic mne RawArrays). Every `acar.v5.substrate` module imports with **NO** heavy
dependency (`brainvision_read_repair` is pure/stdlib; mne/numpy stay lazy). An adversarial 5-lens review (read-repair safety /
header-rewrite correctness / F7 completion / schema-V3 label-firewall / wiring-purity-completeness), each finding independently
refuted, was run before commit.

## Still forbidden in Stage-1B12 (unchanged)
real DEV signal load · DSP on real data · interpolation on real signals · training · embedding dump · registry population · SLURM ·
Stage-2 selection · S1/S2/S3 · external/held-out · lockbox · cohort/subject dropping · reduced-channel substrate · modern-montage
re-pin · F7-0/F7-1 remapping · keep-first de-duplication · general BrainVision repair · the real build.

## Next gates (separate authorizations)
1. **Stage-1B12P** — metadata/header/repair/geometry preflight (read-only, incl. materializing the ephemeral repaired headers for a
   header-read test), classifying each recording `native_19_pass` / `montage_completion_required` / `read_repair_required` /
   `read_repair_plus_montage_completion_required` / `fail`. No signal load / interpolation-on-real / DSP / training / embedding /
   registry.
2. **Stage-1B real run** — only if Stage-1B12P passes: a NEW authorization pinned to the reviewed Stage-1B12 commit (full 40-hex
   `implementation_base_sha`) + a captured runtime lock. The `0ab40ec` authorization remains superseded and must not be used.
