# ACAR v4 — held-out cohort audit (METADATA-ONLY)

```
STATUS   : METADATA-ONLY AUDIT (no predictor, no adaptation, no ΔR, no endpoint — metadata only)
PURPOSE  : produce the admissible held-out site list for ACAR_FROZEN_v4.md §4 (the hard blocker to tagging)
EXTERNAL : NO DATA READ FOR MODELING. This file records only published metadata + disjointness reasoning.
DATE     : 2026-06-29
```

The external Arm B may use **only** sites listed ADMISSIBLE here, after this file is completed, the protocol tagged
`acar-v4-protocol`, and sign-off given. **Every prior verdict below is carried over from the v3 lockbox audit and must
be RE-VERIFIED against the primary source before tag** (sample sizes, labels, montage/Fs, license, raw availability,
acquisition units, and overlap with the seven DEV cohorts). Nothing here authorizes a modeling read.

DEV cohorts to stay disjoint from: **PD** ds002778, ds003490, ds004584 · **SCZ** ds003944, ds003947, ds004000, ds004367.

## Required metadata fields (per candidate; fill from primary source)
```
disease · dataset id / DOI · n subjects (per class: PD/HC or SCZ/HC) · modality · task (resting?) · raw available? ·
license · channels / sampling rate · subject-id scheme · acquisition units (sites/devices) · overlap risk vs DEV ·
admissibility verdict + reason
```

## Candidates (prior v3 verdicts — RE-VERIFY; metadata only)

| disease | dataset / DOI | prior verdict (v3) | n (class) | ch / Fs | task / raw | license | acquisition units | overlap vs DEV | RE-VERIFY before tag |
|---------|---------------|--------------------|-----------|---------|------------|---------|-------------------|----------------|----------------------|
| SCZ | Zenodo 14808296 | **ADMISSIBLE** | 38 SCZ + 39 HC | 64ch / 1000 Hz | resting · raw | CC-BY | single (verify) | none known vs ds0039xx/ds004000/ds004367 | ☐ confirm n/labels/raw/license; ☐ confirm disjoint from DEV SCZ |
| SCZ | ASZED 14178398 | **PROVISIONAL — integrity review required** | (verify) | 16ch / 200&256 Hz | resting (verify) · raw | (verify) | **≥2 Nigerian units/devices** (200 vs 256 Hz) | none known | ☐ 2026-05 integrity preprint still UNVERIFIED → blocks upgrade; ☐ per-unit stratum (§3a) if used |
| PD | OpenNeuro ds007526 | **PROVISIONAL — admissible** | 116 PD + 28 HC | 65ch / 250 Hz | resting · raw | CC0 | single (verify) | verify disjoint from ds002778/ds003490/ds004584 | ☐ confirm HC-vs-PD diagnosis label usable; ☐ confirm no subject overlap |
| PD | OpenNeuro ds007020 | **EXCLUDED** | 94 (mortality living/deceased) | — / 500 Hz | — | — | — | UCSD-overlap inference withdrawn | EXCLUDE unless primary metadata proves a usable HC-vs-PD diagnosis target AND no DEV overlap |

## Disjointness / overlap reasoning (to complete)
- Confirm no shared subjects/recordings between each candidate and the DEV cohorts (id scheme + provenance; OpenNeuro
  accession disjoint is necessary but not sufficient — check for re-released/derived subjects).
- ASZED multi-unit: if two devices/units, the DEFAULT stratum is `(acquisition_unit, disease)` (site-local claim);
  pooling units is allowed only as an explicit mixture-exchangeability claim (declare per site).

## PD single-site contingency
If, after re-verification, **only ds007526** is admissible for PD, then PD external confirmation can be **SINGLE-SITE
only** — reported as single-site confirmatory, NOT cross-site replication (per ACAR_FROZEN_v4.md §3d). A second
independent PD HC-vs-PD resting-raw site is desirable but not yet identified; sourcing it (metadata only) is open.

## Output of this audit (fills ACAR_FROZEN_v4.md §4)
On completion, list per admissible site: dataset id, disease, n per class, acquisition-unit structure, the chosen
`(site/unit, disease)` stratum, and the split parameters (CAL frac 0.40, seed 0, min 20/20). Until then §4 stays TBD and
the protocol is NOT tagged.
