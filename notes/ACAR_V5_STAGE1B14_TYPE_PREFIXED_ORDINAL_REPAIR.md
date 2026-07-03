# ACAR V5 — Stage-1B14 type-prefixed ordinal BrainVision channel-name repair (CODE + SYNTHETIC/FIXTURE TESTS ONLY; NO REAL RUN)

```
Stage-1B13P failed only because 65 ds003944/ds003947 recordings use type-prefixed ordinal placeholders.
Stage-1B13 pure-EEG ordinal repair is validated (78/78 ds003944 pure-generic recordings renamed).
Stage-1B14 authorizes widening the placeholder detector from EEG<i> only to {EEG,EOG,ECG}<i>, with the integer i equal to
the 1-based data-column position for every channel.

Authorized:
  row-order channels.tsv rename for ds003944/ds003947 BrainVision ordinal-placeholder headers ({EEG,EOG,ECG}<i>, i==position).

Not authorized:
  arbitrary alpha prefixes · fuzzy matching · channels.tsv global fallback · channels.tsv override of real header names ·
  cohort dropping · subject dropping · reduced-channel substrate · the real build.
```

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B14 fixes the single remaining blocker the Stage-1B13P preflight surfaced
(see `ACAR_V5_STAGE1B13P_METADATA_REPAIR_NAME_GEOMETRY_PREFLIGHT_RESULT.md`): 4 ds003944 + all 61 ds003947 recordings use
**type-prefixed ordinal** header names — `<PREFIX><position>` where the integer equals the 1-based data-column position but the
eye/cardiac channels carry `EOG`/`ECG` prefixes instead of `EEG` (e.g. `EOG062`, `ECG063`) — which the Stage-1B13 `EEG<i>`-only
detector correctly refused. **No real DEV signal load, no DSP, no interpolation on real signals, no training, no embedding, no
registry, no SLURM.** The raw signal is NEVER modified, copied, or read by the repair.

## 1. Widened ordinal-placeholder detector (`brainvision_read_repair.py` + `preprocessing_config.py`)
The Stage-1B13 `_is_generic_sequential` (i-th name must be exactly `EEG<i>`) is replaced by `_ordinal_header_info`, which accepts a
header as an ordinal placeholder iff **every** channel name matches `^(EEG|EOG|ECG)0*[1-9][0-9]*$` (prefix from the PINNED set
`channel_name_repair_allowed_ordinal_prefixes = [EEG, EOG, ECG]`) **and** the parsed integer equals the 1-based position for every
channel. It returns a **subtype** — `pure_eeg_ordinal` (all EEG) or `type_prefixed_ordinal` (some EOG/ECG) — and the per-position
prefixes. Everything else about the Stage-1B13 repair is unchanged: the same row-order rename (`Ch_i ← channels.tsv row i`), the same
channels.tsv gate (exists / name column / row count == header count / unique after strip+casefold / latin-1 encodable / all 19
canonical resolve via the Stage-1B10 aliases with no logical duplicate), the same ephemeral staging-only materialization (raw tree
byte-identical; only `.vhdr` + `channels.tsv` text read, never the `.eeg`), composing with the marker fix.

**Fail-closed (no rename → marker-only fallback or `None`)** on: any non-`{EEG,EOG,ECG}` prefix (e.g. `GSR`/`MISC`/`REF`); any integer
≠ its position; any non-ordinal / real name among the channels (no partial rename); a header that already carries real names (never
override — the raw header stays decisive; a bare `ECG`/`EOG` real name with no ordinal is not a placeholder); any cohort other than
ds003944/ds003947; or an invalid channels.tsv. The mode identifier string
(`channel_names_from_channels_tsv_for_generic_brainvision`) is **unchanged** for provenance continuity; the widening is captured by the
bumped `channel_name_repair_policy_version = ACAR_V5_STAGE1B14_ORDINAL_CHANNEL_NAME_REPAIR_V2`, the pinned prefix set, and the subtype.

## 2. Subtype provenance + feature-dump schema **V5**
Every mode-C repair manifest now records `channel_name_repair_subtype` + `original_header_ordinal_prefixes`; `apply_repair`
**re-derives** them from the original header and refuses if they disagree with the plan, and `assert_manifest_consistent` re-derives
them again on consumption (mismatch → reject), on top of the Stage-1B13 hash checks + the repaired-names-resolve-19 check. Feature-dump
`SCHEMA_VERSION = ACAR_V5_STAGE1B_FEAT_DUMP_V5`: the header adds `channel_name_repair_subtype_by_recording` (JSON map
`subject::recording -> {subtype, ordinal_prefixes}`), label-free (the nested-key scan covers it); the writer defaults it; the trainer
emits it for ONLY the mode-C recordings. A V4-shaped dump (missing the V5 field) is rejected. So Stage-2 can distinguish native /
marker-repaired / pointer-repaired / **pure-EEG-ordinal-renamed** / **type-prefixed-ordinal-renamed** / montage-completed recordings.

## 3. Stage-1B14P preflight tool (`acar/v5/stage1b14p_preflight.py`) — code only, NOT run on real data here
The SAME read-only 8-way classifier as Stage-1B13P; with the widened detector a type-prefixed ordinal recording now classifies as
`read_repair_plus_channel_name_repair_required` (and an arbitrary-prefix one still as `fail(header_channel_names_non_canonical)`,
surfaced for review). preload=False ONLY; no signal/DSP/interpolation/full-file-hashing/`preprocess_subject`/training/registry/SLURM.
**Running it on the real DEV cohorts is the separate Stage-1B14P authorization.**

## Verification
Full v5 suite = **141 guard modules, green py3.9 + py3.13** (10 new Stage-1B14 suites; rename tests use synthetic BrainVision triplets
with type-prefixed ordinal headers + a BIDS `channels.tsv` + real mne header reads). Every `acar.v5.substrate` module + both preflight
tools import with **NO** heavy dependency (`brainvision_read_repair` imports only the pure `re` + `channel_aliases`; mne/numpy stay
lazy). An adversarial 5-lens review (detector-widening safety / subtype-manifest integrity / schema-V5 label-firewall / preflight +
non-regression / purity-config-completeness) surfaced **one confirmed medium finding — FIXED**: the Stage-1B14P preflight tool's
ephemeral staging-dir prefix was copy-pasted as `acar_v5_1b13p_staging_` (cosmetic — misleading in logs/reports); corrected to
`acar_v5_1b14p_staging_`. No functional/safety findings.

## Still forbidden in Stage-1B14 (unchanged)
arbitrary alpha prefixes · fuzzy matching · channels.tsv global fallback · overriding real header names · cohort/subject dropping ·
reduced-channel substrate · real DEV signal load · DSP on real data · interpolation on real signals · training · embedding · registry ·
SLURM · Stage-2 · S1/S2/S3 · external/held-out · lockbox · the real build.

## Next gates (separate authorizations)
1. **Stage-1B14P** — run the read-only 8-way preflight on the 7 real DEV cohorts (materializing the ephemeral repaired headers for a
   `preload=False` open test). Expected, if no new blocker: all 539 recordings admissible under a reviewed non-fail class
   (ds003944/ds003947 → `read_repair_plus_channel_name_repair_required`). Set the tool's `IMPLEMENTATION_BASE_SHA` to the reviewed
   Stage-1B14 commit before that run.
2. **Stage-1B real run** — only if Stage-1B14P passes across all 539 recordings: a NEW authorization pinned to the reviewed Stage-1B14
   commit (full 40-hex `implementation_base_sha`) + a captured runtime lock. The `0ab40ec` authorization remains superseded.
