# ACAR V5 — Stage-1B13P metadata / header / repair / name / geometry PREFLIGHT result (READ-ONLY; NOT a Stage-1B build)

```
STATUS: FAIL
implementation_base_sha: 2e8b1face003775ae5b3e6a06b7e576dd7863463   (Stage-1B13)
protocol_tag_target_sha: 4278435975a72b1127803dd2cffab420c083e430   (tag acar-v5-protocol)
tool: acar/v5/stage1b13p_preflight.py   env: acar-v4-regen (py3.13, mne 1.12.1)
```

Stage-1B13P is the read-only preflight authorized after Stage-1B13. For every recording of the 7 frozen DEV cohorts it planned the
reviewed BrainVision repair (marker synth / pointer rewrite / **channels.tsv channel-name rename**), materialized the ephemeral
repaired header into a staging dir, opened the (repaired-or-original) header at **`preload=False`**, adjudicated duplicates, computed
the missing canonical channels + `standard_1020` donor geometry, and classified each recording into the 8 authorized classes. **No
signal was loaded** (`preload=False` only), no DSP, no real interpolation, no `complete_missing_channels`/`preprocess_subject`/
`_windows_from_raw` on real data, no full raw-file hashing, no training/embedding/registry/SLURM. The only files written are the
ephemeral repaired headers/markers under the staging dir.

## Verdict
**FAIL.** The Stage-1B13 channels.tsv-rename design is **validated on real data** — 78 pure-generic ds003944 recordings are renamed
and resolve all 19 canonical, and every previously-reviewed case still resolves — **except for one NEW blocker (65 recordings)** that
only became visible now: 4 ds003944 + all 61 ds003947 recordings use **type-prefixed ordinal** header names (`EOG<pos>` / `ECG<pos>`
for the eye/cardiac channels instead of `EEG<pos>`), which are NOT exactly generic-sequential `EEG<i>`, so the pinned rename detection
fail-closes on them. Labels remain 100% resolvable (456/456, 0 failures).

## Per-cohort summary
| disease | cohort   | n_subj | lbl_ok | lbl_fail | recs | native19 | montComp | readRep | nameRep | rep+name | rep+comp | rep+name+comp | fail |
|---------|----------|-------:|-------:|---------:|-----:|---------:|---------:|--------:|--------:|---------:|---------:|--------------:|-----:|
| PD      | ds002778 |     31 |     31 |        0 |   46 |       46 |        0 |       0 |       0 |        0 |        0 |             0 |    0 |
| PD      | ds003490 |     50 |     50 |        0 |   75 |       75 |        0 |       0 |       0 |        0 |        0 |             0 |    0 |
| PD      | ds004584 |    149 |    149 |        0 |  149 |        0 |      149 |       0 |       0 |        0 |        0 |             0 |    0 |
| SCZ     | ds003944 |     82 |     82 |        0 |   82 |        0 |        0 |       0 |       0 |       78 |        0 |             0 |    4 |
| SCZ     | ds003947 |     61 |     61 |        0 |   61 |        0 |        0 |       0 |       0 |        0 |        0 |             0 |   61 |
| SCZ     | ds004000 |     43 |     43 |        0 |   86 |        0 |       84 |       0 |       0 |        0 |        2 |             0 |    0 |
| SCZ     | ds004367 |     40 |     40 |        0 |   40 |       25 |       15 |       0 |       0 |        0 |        0 |             0 |    0 |
| **all** |          | **456**| **456**|      **0**|**539**|    **146**|    **248**|    **0**|    **0**|    **78**|     **2**|         **0** | **65**|

(Admissible = 539 − 65 = **474**: 146 native + 248 montage-completion + 78 read-repair+channel-name-repair + 2 read-repair+completion.)

## Repair summary — the Stage-1B13 rename WORKS on real data
```
n_missing_markerfile_minimal_vmrk     = 65    (the 65 type-prefixed-ordinal recordings fell back to the marker-only fix)
n_broken_internal_pointer_rewrite     = 2     (ds004000 sub-042 proposer + responder)
n_channel_names_from_channels_tsv     = 78    (pure-generic ds003944 recordings renamed from channels.tsv, row order)
n_repair_manifest_validated           = 145
n_repaired_header_preload_false_pass  = 145
n_repaired_header_preload_false_fail  = 0
```
All 145 header-defective recordings open at `preload=False` after the ephemeral repair (0 fail); every manifest re-verified. The **78
pure-`EEG001..EEG0NN`** ds003944 recordings are renamed from channels.tsv by row order and resolve all 19 canonical →
`read_repair_plus_channel_name_repair_required`. ds004000 sub-042's 2 recordings → `read_repair_plus_montage_completion_required`;
ds004584 Pz (149) + ds004000 F3/F4/P3/P4 (84) + ds004367 conditional F7 (15) unchanged; PD ds002778/ds003490 native.

## The one remaining blocker (65 recordings) — type-prefixed ordinal header names
`failure_type histogram: header_channel_names_non_canonical × 65` (ds003944 4/82, ds003947 61/61). **No other failure type.**

The 65 failing headers are NOT pure generic-sequential `EEG<i>`: every channel name is `<PREFIX><ordinal>` where the **number equals the
1-based data-column position**, but the eye/cardiac channels carry `EOG`/`ECG` prefixes rather than `EEG`. Exact, uniform pattern
(verified over all 65; `0` "other"):
- **ds003944 (4 recs)**: `sub-1983` (`EOG61, EOG62, ECG63`), `sub-2140A`, `sub-2221A`, `sub-2223` (each `EOG62`);
- **ds003947 (61 recs, ALL)**: `EOG62` at position 62 (uniform across every subject).

The pinned Stage-1B13 detection (`_is_generic_sequential`: the i-th name must be exactly `EEG<i>`) requires the `EEG` prefix on
**every** channel, so it correctly refuses these headers and falls back to the marker-only fix, leaving the generic names → the pinned
reader (which uses `raw.ch_names`) sees 0/19 canonical while channels.tsv resolves 19/19. This is the same class of defect as the
Stage-1B12P `header_channel_names_non_canonical` blocker, but with a **type-prefixed** ordinal placeholder instead of a pure `EEG`
one — invisible until the 78 pure-generic recordings proved the rename path itself is sound.

Note (for the reviewed decision, NOT implemented here): because the ordinal always equals the data-column position and channels.tsv is
data-ordered, the SAME row-order rename would be valid for these 65 if the generic-pattern detection were widened from `EEG<i>` to
`<ALPHA-PREFIX><ordinal == position>` (still a strict, auditable, no-fuzzy, row-order rule). That is a **repair expansion** requiring a
new authorization (e.g. Stage-1B14), out of scope for this preflight.

### Failure table (grouped; all 65 identical failure_type)
| disease | cohort | subjects | recording | failure_type | message |
|---------|--------|----------|-----------|--------------|---------|
| SCZ | ds003944 | sub-1983, sub-2140A, sub-2221A, sub-2223 | `sub-*_task-Rest_eeg.vhdr` | `header_channel_names_non_canonical` | header names `EEG001..`/`EOG0NN`/`ECG0NN` resolve 0/19; channels.tsv resolves 19/19 |
| SCZ | ds003947 | all 61 | `sub-*_task-rest_eeg.vhdr` | `header_channel_names_non_canonical` | header names `EEG001..`/`EOG062` resolve 0/19; channels.tsv resolves 19/19 |

## Interpretation
- **Labels**: fully compatible (456/456).
- **Header openability**: fixed — all 145 previously-unreadable recordings open at `preload=False` (0 fail).
- **channels.tsv rename**: validated — the 78 pure-generic ds003944 recordings rename correctly and resolve all 19 canonical; the
  Stage-1B10/1B11/1B12/1B13 channel design holds for the 474 admissible recordings.
- **The blocker is a NARROWER generic-pattern than pinned**: 65 recordings use type-prefixed ordinal placeholders
  (`EOG<pos>`/`ECG<pos>`), which the pinned `EEG<i>`-only detection fail-closes on. Widening the detection is a reviewed repair
  expansion, not a preflight action.

## Decision (per the Stage-1B13P authorization: "If Stage-1B13P fails: Stop and report")
Stopping and reporting. **No** cohort dropping, **no** subject dropping, **no** repair expansion (incl. **no** widening of the
generic-header detection to accept `EOG`/`ECG`-prefixed ordinals), **no** interpolation-whitelist expansion, **no** channels.tsv
fallback expansion, **no** fuzzy matching, **no** rerun, **no** real build — **without review**. The Stage-1B real build remains **not
authorized** (the superseded `0ab40ec` authorization stays superseded). The next step requires an explicit reviewed decision on whether
to widen the generic-sequential detection to `<ALPHA-PREFIX><ordinal == position>` (row-order, no-fuzzy) for ds003944/ds003947.

## Still forbidden (unchanged; honored by this preflight)
no signal preload · no get_data · no raw.load_data · no DSP · no filtering · no resampling · no interpolation on real signals · no
`complete_missing_channels` on real data · no full raw-file hashing · no `preprocess_subject`/`_windows_from_raw` on real recordings ·
no training · no embedding · no artifact writing except the ephemeral repaired-header staging files · no registry · no
`registry.json`/`FINALIZED.json` · no SLURM · no Stage-2 · no G1–G6 · no S1/S2/S3 · no held-out/external · no lockbox.
