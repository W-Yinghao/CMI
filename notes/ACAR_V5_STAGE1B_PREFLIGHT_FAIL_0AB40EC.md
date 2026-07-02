# ACAR V5 — Stage-1B real-run authorization 0ab40ec: SUPERSEDED BY PREFLIGHT STRUCTURAL INCOMPATIBILITY

```
STAGE1B_REALRUN_AUTH_0AB40EC_SUPERSEDED_BY_PREFLIGHT_FAIL
NO SLURM
NO TRAINING
NO EMBEDDING
NO ARTIFACTS
NO REGISTRY
NO EXTERNAL
NO LOCKBOX
```

The first Stage-1B real-run authorization (pinned to `implementation_base_sha = 0ab40ecc1b3a071cc2670dfb6a469fa9d09408b9`) was NOT
executed. A metadata-only preflight against the frozen DEV cohorts (read of `participants.tsv` + `channels.tsv` only — NO signal
load, NO DSP, NO training, NO embedding, NO artifacts, NO registry, NO SLURM, NO code change) found TWO structural code↔data
incompatibilities that would make the run fail before any training or artifact. The authorization is therefore superseded, not a
failed build.

## The two blockers (recorded exactly)

**Blocker 1 — cohort label-schema mismatch (fatal at `subject_eligibility`, before any split/train).** The pinned `stage1b_label_source`
read only a lowercase column in {group, diagnosis, dx, participant_group} with a fixed global alias set. The 7 frozen DEV cohorts each
encode the control/case label differently, and ALL 7 fail as-pinned:

| cohort | label location | values | why it failed |
| --- | --- | --- | --- |
| PD ds002778  | (no column) subject-id prefix | `hc`(16)/`pd`(15) | no group column at all |
| PD ds003490  | column `Group` | `CTL`/`PD` | column case; `CTL` not an alias |
| PD ds004584  | column `GROUP` | `Control`/`PD` | column case |
| SCZ ds003944 | column `type` | `Control`/`Psychosis` | `type` not a known column; `Psychosis` not an alias |
| SCZ ds003947 | column `type` | `Control`/`Psychosis` | same |
| SCZ ds004000 | column `group` | `HC`/`P` | `P` not a case alias |
| SCZ ds004367 | column `Group` | `Control`/`Patient` | column case |

**Blocker 2 — modern channel-name mismatch (fatal at DSP, if labels were bypassed).** The pinned `preprocessing_config.CHANNELS_19`
uses the OLD 10-20 temporal names `T3/T4/T5/T6` (and `Fp1/Fp2`), and the reader required exact name presence before the pick. The real
recordings use MODERN 10-10 names — ds002778 has `…T7…P7…T8…P8…` (no T3/T4/T5/T6), ds003944 has `FP1/FP2` (upper-case P) and
`T7/T8/P7/P8` — so `_windows_from_raw` computed `missing=[T3,T4,T5,T6]` (and `Fp1/Fp2` for ds003944) → `RealMneReaderError` on every
subject.

## Datasets confirmed present (real BIDS)
The 7 cohorts DO exist as real raw-BIDS trees under `/projects/EEG-foundation-model/datalake/raw/scps/{PD,SCZ}/dsXXXXXX/` with
recordings at `sub-*/ses-*/eeg/*` (e.g. `.bdf`) + `participants.tsv`. This is a code↔data schema mismatch, not a data-availability
problem — re-running the same commit would not fix it; it required a reviewed code change.

## Resolution
Fixed by the reviewed **Stage-1B10** adaptation layer (channel-alias + cohort-label spec) — see
`ACAR_V5_STAGE1B10_CHANNEL_LABEL_ADAPTATION.md`. A NEW Stage-1B real-run authorization must be issued against the Stage-1B10 commit
(and, per the reviewer, a tiny Stage-1B10P metadata/channel preflight first).
