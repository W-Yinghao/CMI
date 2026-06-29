# ACAR v4 — held-out cohort audit (METADATA-ONLY)

```
STATUS   : METADATA-ONLY AUDIT — primary-source RE-VERIFIED 2026-06-29 (no predictor, no adaptation, no ΔR, no endpoint)
PURPOSE  : produce the admissible held-out site list for ACAR_FROZEN_v4.md §4
EXTERNAL : NO MODELING READ. Verified from published pages / participants.tsv / dataset_description.json only.
DATE     : 2026-06-29
```

The external Arm B may use **only** sites marked ADMISSIBLE here, after §4 is filled, the protocol tagged
`acar-v4-protocol`, the external CLI committed, and sign-off given. DEV cohorts to stay disjoint from: **PD** ds002778,
ds003490, ds004584 · **SCZ** ds003944, ds003947, ds004000, ds004367.

## Verified candidates (primary source, 2026-06-29)

| disease | dataset / DOI | n (per class) | ch / Fs | task / raw | license | acquisition units | disjoint vs DEV | VERDICT |
|---------|---------------|---------------|---------|------------|---------|-------------------|-----------------|---------|
| PD  | OpenNeuro **ds007526** v1.0.2 (doi:10.18112/openneuro.ds007526.v1.0.2) | **116 PD + 28 HC** (participants.tsv `group`) | (verify ch) / (verify Fs) | resting-state + walking → **resting-only**; raw (BIDS 1.10.0) | **CC0** | single (Tel Aviv Sourasky; Katzir/Vered/Maidan) | different accession + institution (DEV PD = UCSD ds002778 etc.) → disjoint | **ADMISSIBLE** (single PD site) |
| SCZ | Zenodo **14808296** | **38 SCZ + 39 HC** | 64ch / 1000 Hz | resting (eyes-closed); raw `sch_resting_64ch_ec.zip` (2.4 GB) + clinical `.xlsx` | **CC-BY-4.0** | single setup (Semmelweis) | different repo (Zenodo vs OpenNeuro) → disjoint | **ADMISSIBLE** (single SCZ site) |
| SCZ | Zenodo **14178398** (ASZED) | 76 SCZ + 77 HC (153) | 16ch (prior) / 200 & 256 Hz (prior) | resting eyes-closed among 4 paradigms; raw (verify) | CC-BY-4.0 | **2 Nigerian units/devices** (Contec KT-2400 200 Hz; BrainMaster Discovery24-E 256 Hz — prior, re-verify) | different repo → disjoint | **PROVISIONAL — NOT admitted**: integrity review unresolved; ch/Fs/device + raw not confirmed on the Zenodo page |
| PD  | OpenNeuro **ds007020** | participants.tsv = `participant_id, survival_status` only (living/deceased); all `sub-PD####`; **no HC** | — / — | — | — | — | — | **EXCLUDED** — no usable HC-vs-PD diagnosis target (mortality-only; conflict resolved by participants.tsv) |

Notes verified this audit: ds007526 `participants.tsv` columns = `participant_id, subject_id, group, updrs_part_iii,
updrs_total, moca, age, sex, disease_duration, ledd, pigd_score, td_score, ctt` (group ∈ {PD, HC}). ds007020 has no
diagnosis column. Zenodo 14808296 is a single-setup resting eyes-closed SCZ dataset with raw EEG + a separate clinical
spreadsheet (the per-subject diagnosis labels live in that .xlsx — to be read at Arm-B label time, metadata-only).
Still to confirm before the external read (metadata-only, at the runner preflight): ds007526 channel count / Fs and the
exact resting-run file selection; Zenodo 14808296 per-subject id↔diagnosis mapping from the .xlsx; no re-released/derived
subject overlap with DEV.

## Split feasibility (metadata-only; frozen seed 0, CAL frac 0.40, min 20/20)
```
ds007526 (PD)        : 144 subjects (116 PD + 28 HC) → CAL ≈ 58 / EVAL ≈ 86; both classes present in both → EVALUABLE.
                       (28 HC is small; the runner preflight must confirm ≥1 HC and ≥1 PD in BOTH CAL and EVAL under seed 0.)
Zenodo 14808296 (SCZ): 77 subjects (38 SCZ + 39 HC) → CAL ≈ 31 / EVAL ≈ 46; both classes present → EVALUABLE.
ASZED (SCZ)          : PROVISIONAL — per-acquisition-unit stratum; each unit must independently meet min 20/20; gated on
                       integrity review. Not counted until admitted.
```

## Admissible held-out set (this audit) → fills ACAR_FROZEN_v4.md §4
```
PD  : (ds007526, PD)        — single PD site → PD external confirmation is SINGLE-SITE confirmatory (NOT replication).
SCZ : (zenodo14808296, SCZ) — single admissible SCZ site at present (ASZED pending integrity) → SCZ also single-site.
```
**Both diseases are single-site at present.** External Arm B would therefore be single-site-per-disease confirmatory; a
second SCZ site (ASZED, after integrity clearance) or a second PD site would be a later, separately-dated protocol
amendment — not required to tag, but the single-site limitation must be stated. ds007020 EXCLUDED; ASZED PROVISIONAL.
