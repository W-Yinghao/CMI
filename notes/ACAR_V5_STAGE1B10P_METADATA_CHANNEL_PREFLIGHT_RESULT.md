# ACAR V5 — Stage-1B10P metadata/channel preflight RESULT

```
status: FAIL
implementation_base_sha: 79447dcad7fce65c6468a89dd2bc518a6b9c8672
protocol_tag_target_sha: 4278435975a72b1127803dd2cffab420c083e430
scope: participants.tsv + channels.tsv / raw headers (preload=False) ONLY
       NO signal load · NO get_data · NO DSP · NO filtering/resampling · NO full-file hashing · NO build_manifest
       NO training · NO embedding · NO artifact · NO registry · NO SLURM · NO Stage-2 · NO external · NO lockbox
tool: acar/v5/stage1b10p_preflight.py (read-only; re-runnable)
```

The preflight was run on the 7 frozen DEV cohorts. It reads only participants.tsv (label) and channels.tsv (channel names), falling
back to the raw HEADER with `preload=False` (channel names only — no signal loaded) where a channels.tsv is absent. Recording
discovery used `raw_recording_manifest.discover_raw_recordings` (path listing only — NO hashing; `build_manifest` was NOT used).
Channel resolution used the Stage-1B10 `channel_aliases` semantics **per recording** (not a subject-union rule); labels used
`cohort_label_spec`.

## Per-cohort table
```
 disease   cohort     subj  lbl_ok  lbl_fail  ctrl  case  recs  ch_ok  ch_fail  channel_src               extras
 PD        ds002778     31      31         0    16    15    46     46        0   channels.tsv               1012
 PD        ds003490     50      50         0    25    25    75     75        0   channels.tsv               3600
 PD        ds004584    149     149         0    49   100   149      0      149   channels.tsv               6737
 SCZ       ds003944     82      82         0    32    50    82     82        0   channels.tsv               3690
 SCZ       ds003947     61      61         0    30    31    61     61        0   channels.tsv               2745
 SCZ       ds004000     43      43         0    24    19    86      0       86   mne_header(preload=False)  9900
 SCZ       ds004367     40      40         0    23    17    40     25       15   channels.tsv               1945
 TOTAL    (7 cohorts)  456     456         0   ...   ...   539    289      250
```
observed channel naming families: **modern_10_10 (T7/T8/P7/P8) × 537** (every recording uses modern names).

## What PASSED
- **Labels: 100% resolved — all 7 cohorts, all 456 subjects, 0 label failures.** The Stage-1B10 cohort-label spec (per-cohort
  column/id-prefix + case-insensitive column matching) resolves every subject; control/case counts are sensible. **Blocker 1 (labels)
  is fully fixed.**
- **Channel-alias layer works:** ds002778, ds003490, ds003944, ds003947 (and the 25 passing ds004367 recordings) all resolve their
  MODERN names (T7/T8/P7/P8 → T3/T4/T5/T6; FP1/FP2 → Fp1/Fp2) to the canonical 19 in canonical order, with extras dropped. **Blocker 2
  (modern names) is fixed** for the cohorts that actually record the full montage.

## What FAILED (250 channel failures across 3 cohorts) — NOT alias/code bugs; genuine electrode-set differences
1. **ds004584 (PD) — all 149 recordings MISSING `Pz`.** The montage records P3/P4/P7/P8/POz but not Pz; there is no alias source for
   Pz. (`missing=['Pz']`.)
2. **ds004000 (SCZ) — all 86 recordings MISSING `F3, F4, P3, P4`** (sparser/clinical montage; no channels.tsv → read from raw header).
   (`missing=['F3','F4','P3','P4']`.)
3. **ds004367 (SCZ) — 15 of 40 recordings have a DUPLICATE `F7`** (its channels.tsv lists `F7` twice in some task recordings, e.g.
   `task-rdk`); the pinned `duplicate_logical_channel_policy = fail_closed` correctly rejects them. The other 25 recordings resolve.
   (`duplicate=['F7']`.)

Representative failure rows:
```
disease  cohort    subject   source                         failure_type       message
PD       ds004584  sub-001   sub-001_task-Rest_eeg.set      missing_canonical  missing=['Pz']
SCZ      ds004000  sub-000   <raw header>                   missing_canonical  missing=['F3','F4','P3','P4']
SCZ      ds004367  sub-S03   sub-S03_task-rdk_channels.tsv  duplicate_logical  duplicate=['F7']
```

## Interpretation — STOP (do NOT build)
The preflight is FAIL. This is exactly the pre-submit check working: from metadata alone it proves that, under the pinned 19-channel
10-20 substrate, **3 of the 7 cohorts cannot currently produce the canonical montage** — two are missing canonical electrodes
(ds004584 Pz; ds004000 F3/F4/P3/P4) and one has a duplicate channel label in some task recordings (ds004367 F7). These are SCIENTIFIC
montage realities, not code/alias bugs. Resolving them requires reviewed decisions (each changes the substrate or the cohort universe)
and a new authorization — e.g. one or more of: montage-completion (interpolate missing electrodes), a per-recording duplicate-channel
policy or task restriction for ds004367, accepting a reduced-channel substrate for the sparse cohorts, or dropping/deferring the
incompatible cohorts. Per the Stage-1B10P instruction, NO patch / rerun / cohort restriction / subject drop / mapping change was made;
the result is reported for review.
