# ACAR V5 — Stage-1B14P metadata / header / repair / name / geometry PREFLIGHT result (READ-ONLY; NOT a Stage-1B build)

```
STATUS: PASS
implementation_base_sha: 3fe885245133e2bc141651955da33bb7fd7adeac   (Stage-1B14)
protocol_tag_target_sha: 4278435975a72b1127803dd2cffab420c083e430   (tag acar-v5-protocol)
tool: acar/v5/stage1b14p_preflight.py   env: acar-v4-regen (py3.13, mne 1.12.1)
```

Stage-1B14P is the read-only preflight authorized after Stage-1B14. For every recording of the 7 frozen DEV cohorts it planned the
reviewed BrainVision repair (marker synth / pointer rewrite / **channels.tsv rename over the widened {EEG,EOG,ECG} ordinal-placeholder
pattern**), materialized the ephemeral repaired header into a staging dir, opened the (repaired-or-original) header at **`preload=False`**,
adjudicated duplicates, computed the missing canonical channels + `standard_1020` donor geometry, and classified each recording into
the 8 authorized classes. **No signal was loaded** (`preload=False` only), no DSP, no real interpolation, no
`complete_missing_channels`/`preprocess_subject`/`_windows_from_raw` on real data, no full raw-file hashing, no
training/embedding/registry/SLURM. The only files written are the ephemeral repaired headers/markers under the staging dir. (Only the
report-metadata `IMPLEMENTATION_BASE_SHA` literal + the report-only subtype counters changed in the tool; no production substrate code
was touched.)

## Verdict
**PASS.** For the first time, **all 539 recordings across the 7 frozen DEV cohorts are admissible under a reviewed non-fail class**
(0 failures). Labels are 100% resolvable (456/456, 0 failures). The Stage-1B14 widened ordinal-placeholder rename renamed **all 143
ds003944/ds003947 recordings** (78 pure-EEG-ordinal + 65 type-prefixed-ordinal) from their BIDS `channels.tsv`, resolving all 19
canonical channels; the two ds004000 sub-042 broken-pointer recordings were repaired + montage-completed; ds004584 Pz, ds004000
F3/F4/P3/P4, and ds004367 conditional F7 completion all resolve; PD ds002778/ds003490 are native.

## Per-cohort summary
| disease | cohort   | n_subj | lbl_ok | lbl_fail | recs | native19 | montComp | readRep | nameRep | rep+name | rep+comp | rep+name+comp | fail |
|---------|----------|-------:|-------:|---------:|-----:|---------:|---------:|--------:|--------:|---------:|---------:|--------------:|-----:|
| PD      | ds002778 |     31 |     31 |        0 |   46 |       46 |        0 |       0 |       0 |        0 |        0 |             0 |    0 |
| PD      | ds003490 |     50 |     50 |        0 |   75 |       75 |        0 |       0 |       0 |        0 |        0 |             0 |    0 |
| PD      | ds004584 |    149 |    149 |        0 |  149 |        0 |      149 |       0 |       0 |        0 |        0 |             0 |    0 |
| SCZ     | ds003944 |     82 |     82 |        0 |   82 |        0 |        0 |       0 |       0 |       82 |        0 |             0 |    0 |
| SCZ     | ds003947 |     61 |     61 |        0 |   61 |        0 |        0 |       0 |       0 |       61 |        0 |             0 |    0 |
| SCZ     | ds004000 |     43 |     43 |        0 |   86 |        0 |       84 |       0 |       0 |        0 |        0 |             2 |    0 |
| SCZ     | ds004367 |     40 |     40 |        0 |   40 |       25 |       15 |       0 |       0 |        0 |        0 |             0 |    0 |
| **all** |          | **456**| **456**|      **0**|**539**|    **146**|    **248**|    **0**|    **0**|    **143**|     **2**|         **0** | **0**|

(`rep+name` = read_repair_plus_channel_name_repair_required; `rep+comp` = read_repair_plus_montage_completion_required. Class totals:
146 native + 248 montage-completion + 143 read-repair+channel-name-repair + 2 read-repair+montage-completion = **539**, fail = **0**.)

Note: ds003944/ds003947 are marker-less, so their channels.tsv rename ALSO synthesizes a minimal marker (a read-repair) → the composite
class is `read_repair_plus_channel_name_repair_required` (verified: for sub-1448 the plan sets `marker_target=""` → a marker is
synthesized → `generated_marker_sha256` is set → `marker_repair=True`).

## Repair summary — every real header-defect is now a reviewed repair
```
n_missing_markerfile_minimal_vmrk     = 0     (no recording falls back to marker-only any more; the 65 type-prefixed now rename)
n_broken_internal_pointer_rewrite     = 2     (ds004000 sub-042 proposer + responder)
n_channel_names_from_channels_tsv     = 143   (ALL ds003944 82 + ds003947 61 renamed from channels.tsv, row order)
  n_pure_eeg_ordinal                  = 78    (EEG001..EEG0NN headers)
  n_type_prefixed_ordinal             = 65    (EOG<pos>/ECG<pos> on the eye/cardiac channels — the Stage-1B13P blocker)
n_repair_manifest_validated           = 145
n_repaired_header_preload_false_pass  = 145
n_repaired_header_preload_false_fail  = 0
```
All 145 header-defective recordings (143 ordinal-placeholder renames + 2 broken-pointer rewrites) open at `preload=False` after the
ephemeral repair (0 fail); every manifest re-verified (incl. the subtype re-derivation). The 78 pure-EEG + 65 type-prefixed split
matches the Stage-1B13P characterization exactly.

## Montage summary
```
ds004584: missing {Pz}          × 149   donor_estimate min 61 / median 61
ds004000: missing {F3,F4,P3,P4} × 86    donor_estimate min 35 / median 35   (incl. sub-042's 2 repaired recordings)
ds004367: missing {F7}          × 15    donor_estimate min 61 / median 61   (conditional F7 via the F7-0/F7-1 variant pattern)
unreviewed missing-channel cases = 0
```

## Failure table
None — no recording failed under any class.

## Interpretation
- **Labels**: fully compatible (456/456).
- **Header openability**: all 145 header-defective recordings open at `preload=False` (0 fail).
- **Channel naming / geometry**: the full Stage-1B10..1B14 channel design now covers 100% of the real DEV data — aliasing (modern
  10-10 → old 10-20), montage completion (ds004584 Pz, ds004000 F3/F4/P3/P4, ds004367 conditional F7), marker/pointer read-repair
  (ds003944/ds003947 marker synth, ds004000 sub-042 pointer rewrite), and channels.tsv row-order rename over the widened
  {EEG,EOG,ECG} ordinal-placeholder pattern (ds003944/ds003947, pure + type-prefixed).
- **No new blocker surfaced.** Every recording maps to exactly one reviewed non-fail class.

## Decision (per the Stage-1B14P authorization: "If Stage-1B14P passes: Stop and report. Do not start Stage-1B real build")
Stopping and reporting. **The Stage-1B real build is NOT started here** — it remains a SEPARATE authorization. Per the reviewer, a PASS
across all 539 recordings makes a new Stage-1B real-run authorization pinned to `3fe885245133e2bc141651955da33bb7fd7adeac` (+ a captured
runtime lock) the next candidate step; the superseded `0ab40ec` authorization stays superseded. Awaiting that explicit real-run
authorization.

## Still forbidden (unchanged; honored by this preflight)
no signal preload · no get_data · no raw.load_data · no DSP · no filtering · no resampling · no interpolation on real signals · no
`complete_missing_channels` on real data · no full raw-file hashing · no `preprocess_subject`/`_windows_from_raw` on real recordings ·
no training · no embedding · no artifact writing except the ephemeral repaired-header staging files · no registry · no
`registry.json`/`FINALIZED.json` · no SLURM · no Stage-2 · no G1–G6 · no S1/S2/S3 · no held-out/external · no lockbox.
