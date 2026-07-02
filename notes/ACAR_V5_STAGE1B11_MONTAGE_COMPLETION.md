# ACAR V5 — Stage-1B11 reviewed montage-completion + channel-source adjudication (CODE + SYNTHETIC/FIXTURE TESTS ONLY; NO REAL RUN)

```
Stage-1B10P failed at channel compatibility.
Labels passed completely (all 7 cohorts / 456 subjects / 0 failures).
Stage-1B11 is a reviewed pre-run DSP amendment.
Canonical output remains 19 old-10-20.
Only ds004584:Pz and ds004000:F3/F4/P3/P4 are allowed for montage completion.
No cohort is dropped. No subject is dropped. No reduced-channel substrate is authorized.
No real build is authorized by this patch.
```

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B11 fixes the three channel-compatibility gaps the Stage-1B10P metadata
preflight surfaced (see `ACAR_V5_STAGE1B10P_METADATA_CHANNEL_PREFLIGHT_RESULT.md`) via a REVIEWED montage-completion + adjudication
layer. No real DEV signal load, no DSP on real data, no training, no embedding, no registry, no SLURM.

## 1. Canonical output montage UNCHANGED
`CHANNELS_19` stays the old-10-20 logical order (`Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 O2`). It is NOT re-pinned to
modern names.

## 2. Pinned montage-completion policy (`preprocessing_config`, part of `preprocessing_config_sha256`)
`montage_completion_policy_version = ACAR_V5_STAGE1B11_MONTAGE_COMPLETION_V1`;
`allowed_missing_by_cohort = {ds004584: [Pz], ds004000: [F3,F4,P3,P4]}`; `max_interpolated_canonical_channels_per_recording = 4`;
`interpolation_method = mne_interpolate_bads_spherical_spline_standard_1020`; `interpolation_mode = accurate`;
`min_donor_channels = 8`; `donor_policy = good_position_eeg_channels_only; interpolated_channels_not_donors;
noncanonical_donors_dropped_after_canonical_output; unknown_position_channels_ignored`.

## 3. Pinned pipeline order (`real_mne_reader._windows_from_raw`)
raw header names → (2) case-normalize + Stage-1B10 aliases → (3) detect missing canonical → (4) if the missing set is
cohort-whitelisted (and ≤ max, and no duplicate logical), interpolate it (add flat channels, `set_montage('standard_1020')`, mark
bad, `interpolate_bads('accurate')`, finite-check) else FAIL → (5) pick/reorder canonical 19 (donors + extras dropped) → (6) average
reference → (7) 0.5–45 Hz → (8) resample 128 → (9) 512-sample non-overlap windows → (10) per-trial z-score. The recording keeps its
MODERN names through interpolation (old T3/T4/T5/T6 are NOT in `standard_1020`, but the whitelisted missing channels Pz/F3/F4/P3/P4 are
montage-name-agnostic and present in `standard_1020`); the alias-pick then emits the old-canonical 19 in canonical order.

## 4. Interpolation provenance (audit; native vs completed)
`SubjectWindows.provenance` records `channel_alias_policy_sha256`, `montage_completion_policy_sha256`, `raw_manifest_sha256`, and
`interpolated=[...] / n_interpolated / donor_count`; `SubjectWindows.montage_completion` carries the structured record (incl.
per-recording). The feature-dump schema is bumped to **V2**: its header adds `channel_alias_policy_sha256`,
`montage_completion_policy_sha256`, and `montage_completion_by_subject` (a JSON map subject → {interpolated, n_interpolated,
donor_count}) so Stage-2 can audit whether each feature vector came from native or interpolated-completed data.

## 5. ds004367 duplicate F7 — raw-header adjudication (no keep-first)
`channel_aliases.adjudicate_channel_source(channels_tsv_names, raw_header_names)`: the RAW HEADER is decisive (the DSP consumes raw
header names). A duplicate logical channel in the raw header → FAIL (signal-level ambiguity); a duplicate only in channels.tsv with a
clean raw header → non-fatal WARN (metadata inconsistency); both clean → PASS. No keep-first / no silent de-dup. The DSP reader itself
fail-closes on a raw-header duplicate logical channel. Whether ds004367's 15 metadata-duplicate recordings PASS or FAIL is decided at
the Stage-1B11P preflight by reading their RAW headers.

## Verification
Full v5 suite = **107 guard modules, green py3.9 + py3.13** (the montage-interpolation tests run on real mne — 1.8.0 on 3.9, 1.12.1 on
3.13 — against SYNTHETIC RawArrays). Every `acar.v5.substrate` module imports with NO heavy dependency (mne/numpy stay lazy inside the
interpolation branch). 10 new guard suites.

## Still forbidden in Stage-1B11 (unchanged)
real DEV signal load · DSP on real data · interpolation on real signals · training · embedding dump · registry population · SLURM ·
Stage-2 selection · S1/S2/S3 · external/held-out · lockbox.

## Next gates (separate authorizations)
1. **Stage-1B10P → re-run as Stage-1B11P** (metadata/header/geometry preflight, read-only): for every recording report
   `native_19_pass` / `montage_completion_required (exact missing set)` / `fail (reason)` — including reading the ds004367 RAW headers
   to adjudicate the F7 duplicate. No signal load, no interpolation on real signals, no DSP/train/embed/registry.
2. **Stage-1B real run** — only if Stage-1B11P passes: a NEW authorization pinned to the reviewed Stage-1B11 commit (full 40-hex
   `implementation_base_sha`) + a captured runtime lock. The `0ab40ec` authorization remains superseded and must not be used.
