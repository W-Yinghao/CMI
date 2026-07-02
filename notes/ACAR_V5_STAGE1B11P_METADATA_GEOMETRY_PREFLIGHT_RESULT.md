# ACAR V5 — Stage-1B11P metadata / header / geometry PREFLIGHT result (READ-ONLY; NOT a Stage-1B build)

```
STATUS: FAIL
implementation_base_sha: fd0e1a77139fb8261588a8ba4ced3b6102255451   (Stage-1B11)
protocol_tag_target_sha: 4278435975a72b1127803dd2cffab420c083e430   (tag acar-v5-protocol)
tool: acar/v5/stage1b11p_preflight.py   env: acar-v4-regen (py3.13, mne 1.12.1)
```

Stage-1B11P is the read-only pre-run preflight authorized after Stage-1B11. It reads **only** metadata + headers + standard
geometry — for every recording of the 7 frozen DEV cohorts it resolved the cohort label (`cohort_label_spec`), read the
`channels.tsv` names, read the raw **header** channel names via `mne.io.read_raw_*(preload=False)`, ran the RAW-HEADER-decisive
duplicate adjudication (`channel_aliases.adjudicate_channel_source`), applied the Stage-1B10 aliases, computed the missing canonical
channels, and classified each recording `native_19_pass` / `montage_completion_required` / `fail` — with a `standard_1020` donor
geometry estimate for the completion cases. **No signal was loaded** (no `get_data` / `load_data` / DSP / filter / resample), **no
interpolation was run on real signals**, no full-file hashing, no `preprocess_subject` / `_windows_from_raw` /
`complete_missing_channels` on real raw data, no training, no embedding, no artifact, no registry, no SLURM.

## Verdict
**FAIL.** Labels are 100% resolvable (all 456 subjects, 0 label failures — as at Stage-1B10P). Channel/header compatibility does
**not** pass: **160 of 539 recordings** cannot be admitted by the pinned Stage-1B11 pipeline. Two of the three PD cohorts pass
natively and the third (ds004584) passes by the whitelisted Pz completion; but **the SCZ arm is largely blocked** — two SCZ cohorts
are 100% unreadable by the pinned mne, and two more have per-recording defects. Each failure below was independently reproduced with a
bare `mne` call (not a preflight-tool artifact).

## Per-cohort summary
| disease | cohort   | n_subj | lbl_ok | lbl_fail | recs | native19 | completion | tsv_warn | fail |
|---------|----------|-------:|-------:|---------:|-----:|---------:|-----------:|---------:|-----:|
| PD      | ds002778 |     31 |     31 |        0 |   46 |       46 |          0 |        0 |    0 |
| PD      | ds003490 |     50 |     50 |        0 |   75 |       75 |          0 |        0 |    0 |
| PD      | ds004584 |    149 |    149 |        0 |  149 |        0 |        149 |        0 |    0 |
| SCZ     | ds003944 |     82 |     82 |        0 |   82 |        0 |          0 |        0 |   82 |
| SCZ     | ds003947 |     61 |     61 |        0 |   61 |        0 |          0 |        0 |   61 |
| SCZ     | ds004000 |     43 |     43 |        0 |   86 |        0 |         84 |        0 |    2 |
| SCZ     | ds004367 |     40 |     40 |        0 |   40 |       25 |          0 |       15 |   15 |
| **all** |          | **456**| **456**|      **0**|**539**|    **146**|    **233** |     **15**| **160**|

Observed channel-naming family among the 394 header-readable recordings: **modern 10-10 (T7/T8/P7/P8) ×394** — confirming the
Stage-1B10 alias layer (T7/T8/P7/P8→T3/T4/T5/T6) is the correct and necessary mapping; no recording uses old-10-20 temporal names.
(The 145 header-unreadable recordings are not counted in the naming tally.)

## Montage-completion-required recordings (233) — geometry OK
| cohort   | missing canonical | n recs | min donor_estimate | min_donor req | status |
|----------|-------------------|-------:|-------------------:|--------------:|--------|
| ds004584 | [Pz]              |    149 |                 61 |             8 | montage_completion_required |
| ds004000 | [F3, F4, P3, P4]  |     84 |                 35 |             8 | montage_completion_required |

Both completion sets are ⊆ the reviewed per-cohort whitelist (`ds004584:{Pz}`, `ds004000:{F3,F4,P3,P4}`), ≤ max (4), and have donor
counts far above `min_donor_channels=8`. These would be admissible; they are reported here only, not interpolated (no real-signal DSP
in this preflight).

## Failures (160) — all reproduced with a bare `mne` call
### A. Marker-less BrainVision — pinned mne cannot read the header (143 recs; ds003944 ×82, ds003947 ×61)
`failure_type = raw_header_read_failed`; `message = No option 'markerfile' in section: 'Common Infos'`.
The `.vhdr` `[Common Infos]` declares `DataFile` but **no `MarkerFile`**, and there is **no `.vmrk`** on disk. Direct
`mne.io.read_raw_brainvision(vhdr, preload=False)` **and** `preload=True` (exactly what the real build's `real_mne_reader._read_raw`
calls) both raise `configparser.NoOptionError`. This is a hard incompatibility between the **pinned build environment (mne 1.12.1)**
and these recordings — not a tool artifact, not a preload effect, not a channel-name issue. Every recording in **both** cohorts is
affected (100% of ds003944 and ds003947).
- ds003944 (82/82): sub-1448 sub-1824 sub-1971 sub-1983 sub-1989 sub-1990 sub-1998 sub-1999 sub-2000 sub-2002 sub-2009 sub-2015 sub-2017 sub-2018 sub-2020 sub-2025 sub-2028 sub-2034 sub-2038 sub-2041 sub-2044 sub-2054 sub-2067 sub-2070 sub-2071 sub-2072 sub-2073 sub-2081 sub-2082 sub-2086 sub-2089 sub-2092 sub-2093 sub-2095 sub-2096 sub-2105 sub-2109 sub-2110 sub-2117 sub-2122 sub-2124 sub-2125 sub-2128 sub-2129 sub-2131 sub-2133 sub-2134 sub-2135 sub-2136 sub-2137 sub-2138 sub-2140A sub-2141 sub-2142 sub-2143 sub-2149 sub-2170A sub-2171 sub-2174 sub-2174A sub-2176 sub-2176A sub-2177 sub-2177A sub-2178 sub-2179 sub-2184A sub-2187 sub-2193 sub-2193A sub-2195 sub-2196 sub-2197 sub-2199 sub-2213 sub-2214A sub-2217A sub-2218 sub-2221 sub-2221A sub-2222 sub-2223
- ds003947 (61/61): sub-2235A sub-2237A sub-2238A sub-2240A sub-2245A sub-2246A sub-2249A sub-2252A sub-2259A sub-2262A sub-2266A sub-2267A sub-2268A sub-2269A sub-2276A sub-2279A sub-2290A sub-2318A sub-2320A sub-2336A sub-2342A sub-2350A sub-2351A sub-2355A sub-2356A sub-2367A sub-2372A sub-2379A sub-2384A sub-2387A sub-2389A sub-2392A sub-2397A sub-2403A sub-2406A sub-2418A sub-2419A sub-2426A sub-2430A sub-2431A sub-2442A sub-2448A sub-2456A sub-2476A sub-2477A sub-2479A sub-2480A sub-2494A sub-2496A sub-2498A sub-2503A sub-2505A sub-2512A sub-2514A sub-2518A sub-2526A sub-2530A sub-2538A sub-2574A sub-2578A sub-2581A

### B. Canonical F7 absent in the raw header (15 recs; ds004367)
`failure_type = missing_not_whitelisted`; `message = missing ['F7'] not in cohort whitelist []`.
The raw `.set` header of these 15 recordings contains **`F7-0` and `F7-1`** (two non-canonical, re-referenced/split variants) and
**zero** channels named `F7`; `channels.tsv` lists `F7` twice. So the Stage-1B10P metadata-level "duplicate F7" is a **channels.tsv
artifact** — the raw-header truth is that these recordings have **no canonical F7 electrode**. The adjudication is therefore `WARN`
(tsv duplicate, raw header has no *logical* duplicate because `F7-0`/`F7-1` are non-canonical and dropped) and the recording then
fails the missing-canonical check. The alias layer correctly does **not** guess that `F7-0` or `F7-1` *is* `F7` (that would be a
keep-first, explicitly forbidden). The other **25** ds004367 recordings carry `F7` exactly once → `native_19_pass` (e.g. sub-S16).
- ds004367 (15/40): sub-S01 sub-S02 sub-S03 sub-S04 sub-S05 sub-S06 sub-S07 sub-S08 sub-S09 sub-S10 sub-S11 sub-S12 sub-S13 sub-S14 sub-S15

### C. Broken internal header sidecar pointers (2 recs; ds004000 sub-042)
`failure_type = raw_header_read_failed`; `message = No such file or directory: .../019_P1.vmrk` (and `019_R1.vmrk`).
sub-042's two `.vhdr` files internally declare `DataFile=019_P1.dat` / `MarkerFile=019_P1.vmrk` (and the responder-run analogue),
but the files on disk were BIDS-renamed to `sub-042_task-proposer_run-1_eeg.{dat,vmrk}` — the header's internal pointers were not
updated, so mne (and `raw_recording_manifest.resolve_sidecars`) cannot resolve them. Genuine per-recording data defect. The other
84 ds004000 recordings pass as `montage_completion_required (F3/F4/P3/P4)`.

## Interpretation
- **Labels**: fully compatible (456/456, 0 failures) — unchanged from Stage-1B10P.
- **Channels/geometry where the header IS readable**: fully compatible — the Stage-1B10 alias layer covers 100% of the readable
  cohorts (all modern 10-10), and the Stage-1B11 whitelisted montage completion (ds004584:Pz, ds004000:F3/F4/P3/P4) has ample donor
  geometry. The reviewed Stage-1B10/1B11 channel design is **validated** against real headers for the 379 readable admissible
  recordings.
- **The blocker is header *readability* and per-recording data integrity, not the channel policy**: 145 recordings cannot be parsed
  by the pinned mne (143 marker-less BrainVision + 2 broken sidecar pointers) and 15 genuinely lack a canonical F7. These are
  properties of the data (and of mne 1.12.1's BrainVision strictness), surfaced before any signal was touched.

## Decision (per the Stage-1B11P authorization: "If Stage-1B11P fails: Stop and report")
Stopping and reporting. **No** cohort dropping, **no** subject dropping, **no** whitelist expansion, **no** interpolation-policy
change, **no** duplicate-channel de-duplication, **no** F7-variant remapping, **no** marker-file synthesis, **no** rerun — **without
review**. The Stage-1B real build remains **not authorized** (the superseded `0ab40ec` authorization stays superseded). The next step
requires an explicit reviewed decision on how to treat the three blocker classes (A marker-less BrainVision, B absent-F7 ds004367, C
broken-sidecar sub-042), which is out of scope for this read-only preflight.

## Still forbidden (unchanged; honored by this preflight)
no signal preload · no get_data · no raw.load_data · no DSP windows · no filtering · no resampling · no interpolation on real signals ·
no `raw_recording_manifest` full-file hashing · no `preprocess_subject`/`_windows_from_raw`/`complete_missing_channels` on real raw ·
no EEGNet/source-state training · no embedding dump · no artifact writing · no registry population · no `registry.json`/`FINALIZED.json`
· no SLURM · no Stage-2 selection · no G1–G6 · no S1/S2/S3 · no held-out/external · no lockbox.
