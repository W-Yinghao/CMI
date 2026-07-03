# ACAR V5 — Stage-1B12P metadata / header / repair / geometry PREFLIGHT result (READ-ONLY; NOT a Stage-1B build)

```
STATUS: FAIL
implementation_base_sha: b67ca3b95f3767043f92a574fb6b13732ae799ab   (Stage-1B12)
protocol_tag_target_sha: 4278435975a72b1127803dd2cffab420c083e430   (tag acar-v5-protocol)
tool: acar/v5/stage1b12p_preflight.py   env: acar-v4-regen (py3.13, mne 1.12.1)
```

Stage-1B12P is the read-only pre-run preflight authorized after Stage-1B12. For every recording of the 7 frozen DEV cohorts it: (1)
planned the reviewed BrainVision read-repair (`brainvision_read_repair.plan_repair`) and, when a plan applied, **materialized an
ephemeral repaired header/marker into a staging dir** (`apply_repair`, manifest re-verified), (2) opened the (repaired-or-original)
header at **`preload=False`** to read channel names, (3) ran the raw-header-decisive duplicate adjudication, (4) computed the missing
canonical channels + `standard_1020` donor geometry, and classified each recording `native_19_pass` / `montage_completion_required` /
`read_repair_required` / `read_repair_plus_montage_completion_required` / `fail`. **No signal was loaded** (no `get_data`/`load_data`,
`preload=False` only), **no DSP, no real interpolation, no full raw-file hashing** (only the small `.vhdr`/synth-`.vmrk` headers are
hashed), no `preprocess_subject`/`_windows_from_raw`/`complete_missing_channels` on real recordings, no training/embedding/registry/
SLURM. The only files written are the ephemeral repaired headers/markers under the staging dir.

## Verdict
**FAIL.** The Stage-1B12 read-repair + conditional-F7 + completion design is **validated on real data** — 145/145 repairs open, and
every previously reviewed channel case now resolves — **except for one NEW blocker (143 recordings)** that only became visible once the
ds003944/ds003947 headers were made openable: their `.vhdr` channel names are **generic placeholders** (`EEG001…EEG064`), so 0/19
canonical channels resolve from the header. Labels remain 100% resolvable (456/456, 0 failures).

## Per-cohort summary
| disease | cohort   | n_subj | lbl_ok | lbl_fail | recs | native19 | mont_compl | read_rep | rep+compl | fail |
|---------|----------|-------:|-------:|---------:|-----:|---------:|-----------:|---------:|----------:|-----:|
| PD      | ds002778 |     31 |     31 |        0 |   46 |       46 |          0 |        0 |         0 |    0 |
| PD      | ds003490 |     50 |     50 |        0 |   75 |       75 |          0 |        0 |         0 |    0 |
| PD      | ds004584 |    149 |    149 |        0 |  149 |        0 |        149 |        0 |         0 |    0 |
| SCZ     | ds003944 |     82 |     82 |        0 |   82 |        0 |          0 |        0 |         0 |   82 |
| SCZ     | ds003947 |     61 |     61 |        0 |   61 |        0 |          0 |        0 |         0 |   61 |
| SCZ     | ds004000 |     43 |     43 |        0 |   86 |        0 |         84 |        0 |         2 |    0 |
| SCZ     | ds004367 |     40 |     40 |        0 |   40 |       25 |         15 |        0 |         0 |    0 |
| **all** |          | **456**| **456**|      **0**|**539**|    **146**|    **248** |     **0** |      **2**| **143**|

## Repair summary — the Stage-1B12 read-repair WORKS on real data
```
n_missing_markerfile_minimal_vmrk     = 143   (ds003944 82 + ds003947 61)
n_broken_internal_pointer_rewrite     = 2     (ds004000 sub-042 proposer + responder)
n_repair_manifest_validated           = 145
n_repaired_header_preload_false_pass  = 145
n_repaired_header_preload_false_fail  = 0
```
All 145 header-defective recordings from Stage-1B11P (classes **A** marker-less BrainVision and **C** broken sidecar pointers) are now
**openable** at `preload=False` via the reviewed ephemeral repair, and every repair manifest re-verified. The Stage-1B11P openability
blockers are fixed.

## Montage summary — the Stage-1B11/1B12 channel design resolves on real data
```
ds004584: missing {Pz}          × 149 recordings   donor_estimate min 61 / median 61   (Pz completion)
ds004000: missing {F3,F4,P3,P4} × 86  recordings   donor_estimate min 35 / median 35   (incl. sub-042's 2 repaired recordings)
ds004367: missing {F7}          × 15  recordings   donor_estimate min 61 / median 61   (F7 via the F7-0/F7-1 variant pattern)
ds004367 F7-variant-pattern completions = 15       (every F7-missing ds004367 recording DOES carry F7-0 + F7-1 — the conditional gate matches the data exactly)
```
- ds004367's 15 F7-missing recordings — which **failed** in Stage-1B11P (F7 not whitelisted) — are now `montage_completion_required`
  via the reviewed conditional F7 completion; the other 25 are `native_19_pass`.
- ds004000 sub-042's 2 broken-pointer recordings are `read_repair_plus_montage_completion_required` (pointer-rewrite + F3/F4/P3/P4).
- ds004584 (Pz) and the rest of ds004000 (F3/F4/P3/P4) are unchanged from Stage-1B11P.

## The one remaining blocker (143 recordings) — generic header channel names
`failure_type histogram: header_channel_names_non_canonical × 143` (ds003944 82/82, ds003947 61/61). **No other failure type** — no
raw-header duplicate, no repaired-header-unreadable, no insufficient donor geometry, no unreviewed missing set elsewhere.

Once the marker-less headers are made openable by the Stage-1B12 repair, mne returns the `.vhdr [Channel Infos]` names, which for these
two cohorts are **generic placeholders `EEG001…EEG064`** (verified: the `.vhdr` literally declares `Ch1=EEG001,,`). The real electrode
names (`FP1, F7, T7/T8/P7/P8, …` — which the Stage-1B10 alias layer resolves to all 19 canonical) live **only in the BIDS
`channels.tsv`** (verified: `channels.tsv` resolves **19/19** canonical for both cohorts). The pinned Stage-1B12 reader
(`real_mne_reader`) uses `raw.ch_names` from the header, **not** `channels.tsv`, so under the pinned code all 19 canonical channels are
"missing" for these 143 recordings → they cannot be windowed.

This is a **new, distinct blocker class** that was invisible at Stage-1B11P (those headers could not be opened at all) and that
Stage-1B12 did **not** address — Stage-1B12 fixed header *openability* (marker/pointer defects), not header *channel-naming*. It is a
genuine code↔data incompatibility, not a preflight artifact: the raw `.vhdr` header carries no electrode identity.

### Failure table (grouped; all 143 identical failure_type)
| disease | cohort | subjects | recording | failure_type | message |
|---------|--------|----------|-----------|--------------|---------|
| SCZ | ds003944 | all 82 | `sub-*_task-Rest_eeg.vhdr` | `header_channel_names_non_canonical` | header names `EEG001..` resolve 0/19 canonical; channels.tsv resolves 19/19 |
| SCZ | ds003947 | all 61 | `sub-*_task-rest_eeg.vhdr` | `header_channel_names_non_canonical` | header names `EEG001..` resolve 0/19 canonical; channels.tsv resolves 19/19 |

## Interpretation
- **Labels**: fully compatible (456/456).
- **Header openability**: fixed — the Stage-1B12 read-repair opens all 145 previously-unreadable recordings (0 fail).
- **Channel resolution where the header carries real names**: fully compatible — ds004584 Pz, ds004000 F3/F4/P3/P4, ds004367 F7
  (conditional), and all native cohorts resolve; the Stage-1B10/1B11/1B12 channel + completion design is **validated on real headers**
  for the 396 admissible recordings (146 native + 248 completion + 2 repair+completion = 396; i.e. 539 − 143 fail).
- **The blocker is header *channel-naming***: ds003944/ds003947 encode the electrode identities only in `channels.tsv`; the pinned
  reader does not consult it. A `channels.tsv`-driven header rename would be a NEW reviewed repair mode — out of scope here.

## Decision (per the Stage-1B12P authorization: "If Stage-1B12P fails: Stop and report")
Stopping and reporting. **No** cohort dropping, **no** subject dropping, **no** repair expansion, **no** interpolation-whitelist
expansion, **no** F7 remapping, **no** de-duplication, **no** `channels.tsv`-rename repair, **no** rerun, **no** real build — **without
review**. The Stage-1B real build remains **not authorized** (the superseded `0ab40ec` authorization stays superseded). The next step
requires an explicit reviewed decision on the one remaining blocker class (ds003944/ds003947 generic header channel names vs the
authoritative `channels.tsv`).

## Still forbidden (unchanged; honored by this preflight)
no signal preload · no get_data · no raw.load_data · no DSP · no filtering · no resampling · no interpolate_bads on real signal · no
full raw-file hashing · no `preprocess_subject`/`_windows_from_raw`/`complete_missing_channels` on real recordings · no training · no
embedding · no artifact writing except the ephemeral repaired-header staging files · no registry · no `registry.json`/`FINALIZED.json`
· no SLURM · no Stage-2 · no G1–G6 · no S1/S2/S3 · no held-out/external · no lockbox.
