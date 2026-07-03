# ACAR V5 — Stage-1B13 reviewed channels.tsv-driven BrainVision channel-NAME repair (CODE + SYNTHETIC/FIXTURE TESTS ONLY; NO REAL RUN)

```
Stage-1B12P failed only because ds003944/ds003947 BrainVision headers expose generic EEG001..EEG0NN names.
channels.tsv resolves 19/19 canonical channels for those cohorts.
Stage-1B13 authorizes a narrow channels.tsv-driven name repair for those two cohorts only.

Authorized:
  channels.tsv row-order rename for generic BrainVision headers in ds003944/ds003947.

Not authorized:
  use of channels.tsv as a global fallback · use of channels.tsv to override real header names · fuzzy channel matching ·
  cohort dropping · subject dropping · reduced-channel substrate · the real build.
```

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B13 fixes the single remaining blocker the Stage-1B12P preflight surfaced
(see `ACAR_V5_STAGE1B12P_METADATA_REPAIR_GEOMETRY_PREFLIGHT_RESULT.md`): after the Stage-1B12 read-repair makes the ds003944/ds003947
headers openable, mne returns generic placeholder names `EEG001..EEG0NN` (the `.vhdr` literally declares `Ch1=EEG001,,`), so 0/19
canonical channels resolve from the header — while the BIDS `channels.tsv` holds the real electrode labels (19/19 resolve). **No real
DEV signal load, no DSP, no interpolation on real signals, no training, no embedding, no registry, no SLURM.** The raw signal is NEVER
modified, copied, or read by the repair.

## 1. New repair mode `channel_names_from_channels_tsv_for_generic_brainvision` (`substrate/brainvision_read_repair.py`)
A third whitelisted BrainVision read-repair mode (composing with the Stage-1B12 marker fix). Authorized **only** when ALL hold:
- `cohort ∈ {ds003944, ds003947}` (pinned `channel_name_repair_cohorts`); format BrainVision `.vhdr`;
- the `.vhdr [Channel Infos]` names are EXACTLY generic-sequential (the i-th channel is `EEG<i>`, any zero-padding);
- a BIDS `channels.tsv` exists with a `name` column; its row count == the header channel count; its names are unique after
  strip+casefold; and all 19 canonical channels resolve via the Stage-1B10 aliases with no logical duplicate.

The mapping is **row order only** — staged header `Ch_i` ← `channels.tsv` row `i` (BIDS requires channels.tsv rows in EEG-data-file
order). No fuzzy/heuristic/partial matching. If any condition fails → **no rename** (fall back to the marker-only fix so the generic
defect is surfaced downstream, or `None`). A header that already carries real (non-generic) names is **never** overridden (the raw
header stays decisive; `channels.tsv` is never a global fallback). Everywhere outside these two cohorts the raw header remains
decisive and `channels.tsv` may only warn/audit, never rename.

**Ephemeral + auditable:** the repaired header (and, when marker-less, a synthesized minimal marker) is written ONLY under the caller's
staging dir; the original `.vhdr`/`.eeg`/`channels.tsv` are byte-identical afterward. Only the `.vhdr` and `channels.tsv` TEXT are read
(never the `.eeg`). Every rename emits a manifest — `channel_name_source=channels.tsv`, `channels_tsv_sha256`,
`channel_name_mapping_sha256` (the row-ordered `Ch_i → name` pairs), `original_header_channel_names_sha256`,
`repaired_header_channel_names_sha256`, `channel_name_repair_policy_sha256` — and `assert_manifest_consistent` re-verifies the on-disk
repaired header + `channels.tsv` and that the repaired names resolve the canonical 19 (a tampered header OR a mutated `channels.tsv` is
rejected).

## 2. Provenance + feature-dump schema **V4**
`SubjectWindows.provenance` now also records `channel_name_repair_policy_sha256` + the renamed-recording list; the reader's
`read_repair` record gains `channel_name_repaired`. Feature-dump `SCHEMA_VERSION = ACAR_V5_STAGE1B_FEAT_DUMP_V4`: the header adds
`channel_name_repair_policy_sha256` (hex64) + `channel_name_repair_by_recording` (JSON map `subject::recording ->
{channel_name_source, channels_tsv_sha256, channel_name_mapping_sha256, original/repaired header-name sha256}`) alongside the V3
read-repair + V2 completion maps. It stays LABEL-FREE (the nested-key scan covers the rename map); the writer defaults the new fields;
the trainer emits the rename map for ONLY the mode-C recordings. A V3-shaped dump (missing the V4 fields) is rejected. So Stage-2 can
distinguish native / markerfile-repaired / pointer-repaired / **channels.tsv-renamed generic** / montage-completed recordings.

## 3. Stage-1B13P preflight tool (`acar/v5/stage1b13p_preflight.py`) — code only, NOT run on real data here
A read-only 8-way classifier (`native_19_pass` / `montage_completion_required` / `read_repair_required` /
`channel_name_repair_required` / `read_repair_plus_channel_name_repair_required` / `read_repair_plus_montage_completion_required` /
`read_repair_plus_channel_name_repair_plus_montage_completion_required` / `fail`). It plans + materializes the ephemeral repaired header,
opens it at `preload=False` ONLY, adjudicates duplicates, computes missing canonical + `standard_1020` donor geometry. It loads no
signal, runs no DSP/interpolation, does no full raw-file hashing, and never calls `preprocess_subject`. A synthetic fixture test proves
it classifies a generic-header + valid-`channels.tsv` recording as `read_repair_plus_channel_name_repair_required` (and a generic +
INVALID `channels.tsv` as `header_channel_names_non_canonical`, surfaced for review). **Running it on the real DEV cohorts is the
separate Stage-1B13P authorization.**

## Verification
Full v5 suite = **131 guard modules, green py3.9 + py3.13** (12 new Stage-1B13 suites; rename tests use synthetic generic BrainVision
triplets + a BIDS `channels.tsv` + real mne header reads). Every `acar.v5.substrate` module + both preflight tools import with **NO**
heavy dependency (`brainvision_read_repair` now imports the pure `channel_aliases`; mne/numpy stay lazy). An adversarial 5-lens review
(rename safety / rename correctness / schema-V4 label-firewall / preflight classification / wiring-purity-completeness) surfaced **one
confirmed high-severity fail-closed gap — FIXED**: a `channels.tsv` with a non-latin-1 channel name (BrainVision headers are latin-1)
could pass validation (the 19 canonical resolve from the ascii names) yet crash `apply_repair`'s `text.encode("latin-1")` with an
unhandled `UnicodeEncodeError`. Now `_validate_channels_tsv_for_rename` rejects any non-latin-1 name (→ no rename / marker-only
fallback) AND `apply_repair` converts an encode error into `BrainvisionReadRepairError` (defense-in-depth), with a new guard suite.

## Still forbidden in Stage-1B13 (unchanged)
use of `channels.tsv` as a global fallback · overriding real header names · fuzzy channel matching · cohort/subject dropping ·
reduced-channel substrate · modern-montage re-pin · real DEV signal load · DSP on real data · interpolation on real signals ·
training · embedding dump · registry population · SLURM · Stage-2 · S1/S2/S3 · external/held-out · lockbox · the real build.

## Next gates (separate authorizations)
1. **Stage-1B13P** — run the read-only metadata/header/repair/name/geometry preflight on the 7 real DEV cohorts (materializing the
   ephemeral repaired headers for a `preload=False` open test), classifying every recording into the 8 classes. ds003944/ds003947 are
   expected to be `read_repair_plus_channel_name_repair_required`.
2. **Stage-1B real run** — only if Stage-1B13P passes across all 539 recordings: a NEW authorization pinned to the reviewed Stage-1B13
   commit (full 40-hex `implementation_base_sha`) + a captured runtime lock. The `0ab40ec` authorization remains superseded.
