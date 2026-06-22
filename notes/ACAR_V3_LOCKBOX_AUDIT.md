# ACAR v3 — candidate lockbox METADATA AUDIT (admissibility only)

**Status:** `METADATA-ONLY / NO ACAR INFERENCE / NO ADAPTATION OUTCOME / NO ENDPOINT ACCESSED`
**Date:** 2026-06-22 · **Gates:** `notes/ACAR_V3_FREEZE_SKELETON.md` §S10.

Scope: determine **admissibility** of the four candidate held-out cohorts using public metadata only — license, raw
availability, montage, sampling rate, resting condition, subject IDs, **binary HC-vs-patient compatibility**, sample
size, overlap with DEV, preprocessing compatibility, and **CAL feasibility (≥30 CAL subjects after subject-hash
split)**. No ACAR inference was run; no adaptation outcomes were inspected.

**Source caveat.** The two Zenodo records were fetched directly. The two **OpenNeuro** pages are JavaScript-rendered
and did not return content; their fields below come from **secondary sources (web search / linked papers)** and are
marked `[secondary]`. Before any lockbox designation these MUST be confirmed against the dataset's **primary**
`participants.tsv` + `dataset_description.json` (plain-text, fetchable via the OpenNeuro file API). Verdicts here are
**provisional** accordingly.

## Per-candidate findings

### PD — OpenNeuro ds007020 — "EEG Mortality Dataset in Parkinson's Disease"  → **NOT ADMISSIBLE (unconfirmed)**
- HC-vs-patient: `[secondary]` search returns "15 PD + 16 healthy controls, Scripps Clinic La Jolla" — **this is the
  signature of the UC San Diego resting PD cohort, which is the DEV cohort `ds002778`.** Strong **subject-overlap
  risk** with DEV (likely the same UCSD lineage, possibly a mortality follow-up of those subjects). Cannot confirm a
  *distinct* HC-vs-patient cohort from metadata.
- CAL feasibility: if ~31 subjects total, a subject-hash split yields ≤ ~18 CAL subjects → **FAILS the ≥30-CAL gate**.
  (A larger Iowa/Narayanan mortality cohort is also plausible; the search is contradictory.)
- **Verdict:** per the pre-committed rule (S10.1, "treat ds007020 as unqualified unless metadata confirms a usable,
  *distinct* HC-vs-patient structure"), and given the overlap risk + likely CAL-size failure → **EXCLUDED** unless
  primary `participants.tsv` decisively shows (a) both HC and PD groups, (b) **no** subject overlap with ds002778,
  and (c) ≥30 CAL subjects. Default = unqualified.

### PD — OpenNeuro ds007526 — "Resting-State & Walking EEG in Parkinson's Disease" → **PROVISIONALLY ADMISSIBLE**
- HC-vs-patient: `[secondary]` **116 PD + 30 HC** → binary contrast present. ✓
- Resting condition: present (resting + treadmill walking; **use resting only**, pre-specified). ✓
- CAL feasibility: 146 subjects → subject-hash split CAL ≈ 80–90 → **≥30 ✓**. Caveat: HC is the minority (30); after
  split ~18 HC in CAL / ~12 HC in EVAL — workable but HC-light (note for batch class balance / harm-AUROC evaluability).
- Channels / sampling rate / montage / **license** / raw-format: **not stated** in secondary sources → **VERIFY**.
- Overlap with DEV: distinct study (gait lab; likely Tel-Aviv/Vered MDS 2024) → low risk, but **verify subject IDs**.
- **Verdict:** **PROVISIONALLY ADMISSIBLE (PD)**, conditional on primary-metadata verification of montage/Fs/license,
  raw availability, and a subject-ID overlap check vs the seven DEV cohorts.

### SCZ — Zenodo 10.5281/zenodo.14808296 — resting-state SCZ vs HC → **ADMISSIBLE**
- HC-vs-patient: **38 SCZ + 39 HC** → binary contrast. ✓
- Montage/Fs: **64-channel, 1000 Hz** → 10–20-compatible, resamplable to the DEV pipeline. ✓
- Resting condition: **resting-state**. ✓ · License: **CC-BY-4.0**. ✓ · Raw EEG: **available** (ZIP). ✓
- CAL feasibility: 77 subjects → CAL ≈ 45 → **≥30 ✓**.
- Overlap with DEV: distinct (Zenodo, not the four OpenNeuro SCZ cohorts) → low risk; verify subject scheme.
- **Verdict:** **ADMISSIBLE (SCZ)** — strongest candidate; confirm internal file format + subject IDs at ingest.

### SCZ — Zenodo 10.5281/zenodo.14178398 — "ASZED" (African Schizophrenia EEG Dataset) → **PROVISIONALLY ADMISSIBLE**
- HC-vs-patient: **76 SCZ + 77 HC = 153** → binary contrast. ✓
- Resting condition: present (resting + cognitive/MMN/ASSR; **use resting only**, pre-specified). ✓
- License: **CC-BY-4.0**. ✓ · CAL feasibility: 153 subjects → CAL ≈ 90 → **≥30 ✓** (largest).
- Montage / sampling rate / raw-format: **not stated** → **VERIFY**.
- Population: African indigenous cohort → strong distribution shift vs DEV (useful for external robustness; Arm-B
  coverage is within-site so exchangeability is unaffected).
- Overlap with DEV: none (distinct population). ✓
- **Verdict:** **PROVISIONALLY ADMISSIBLE (SCZ)**, conditional on montage/Fs/raw-format verification.

## Admissibility summary vs the two-site rule (S6, resolution 8)

| disease | candidate | verdict | binding gap |
|---|---|---|---|
| PD | ds007020 | **EXCLUDED** (overlap risk + CAL-size; unqualified per S10.1) | — |
| PD | ds007526 | **PROVISIONALLY ADMISSIBLE** (verify montage/Fs/license/overlap) | only candidate |
| SCZ | Zenodo 14808296 | **ADMISSIBLE** | — |
| SCZ | ASZED 14178398 | **PROVISIONALLY ADMISSIBLE** (verify montage/Fs/raw) | — |

**Decisive finding:** PD currently has **at most ONE** admissible held-out site (ds007526). Per the S6 two-site rule
this means PD would fall to the **single-site contingency** → G2 for PD would be *site-specific, no within-disease
external replication*. **Recommendation:** source **one additional independent PD lockbox** with HC+PD, resting, raw,
≥30 CAL subjects, and no overlap with the DEV cohorts, before freezing — otherwise pre-register PD as explicitly
single-site. SCZ is on track for two sites (one solid + one pending verification).

## Verification checklist before lockbox designation (still metadata-only)

1. ds007526 + ASZED: fetch primary `dataset_description.json` / `participants.tsv` (OpenNeuro file API) and the
   Zenodo file manifest for: channel montage, sampling rate, raw-signal format, license confirmation, resting label.
2. ds007020: fetch primary `participants.tsv`; decide overlap-vs-`ds002778` and HC presence; default **excluded**.
3. Subject-ID / recording overlap check of every retained site against the seven DEV cohorts (no shared subjects).
4. Confirm ≥30 CAL subjects under the pre-registered subject-hash split for each retained site (record the count;
   prefer well above the ~9 mathematical minimum).
5. Source an additional PD site (or pre-register PD single-site).

No ACAR inference, adaptation-outcome inspection, or endpoint access was or will be performed before
`ACAR_FROZEN_v3.md` is committed and tagged on the `acar` lineage on a clean worktree.

## Sources
- ds007020: https://openneuro.org/datasets/ds007020 · mortality paper https://www.medrxiv.org/content/10.1101/2025.07.07.25331047
- ds007526: https://openneuro.org/datasets/ds007526/versions/1.0.1 · subtypes paper https://movementdisorders.onlinelibrary.wiley.com/doi/abs/10.1002/mds.70348
- SCZ resting: https://zenodo.org/records/14808296
- ASZED: https://zenodo.org/records/14178398
