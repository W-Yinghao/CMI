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

### PD — OpenNeuro ds007020 — "EEG Mortality Dataset in Parkinson's Disease" → **EXCLUDED / UNQUALIFIED** (Amendment 1)
- Corrected facts: public release has **94 subjects**, **500 Hz**, resting EEG; the public `participants.tsv` carries
  **mortality labels (`living`/`deceased`)** and **no documented HC-vs-PD diagnosis mapping** or other clinical labels.
- Decisive failure = **S10.1 target compatibility**: no confirmed usable HC-vs-Patient label matching the DEV target.
  The earlier "15 PD + 16 HC / UCSD-`ds002778` signature / overlap / <30 CAL" inference is **withdrawn** (secondary
  conflation); no overlap claim is made without primary subject-identity evidence.
- **Verdict:** **EXCLUDED / UNQUALIFIED — public release does not document a usable HC-vs-PD target label; no overlap
  claim is made without primary subject-identity evidence.** Reconsiderable only if a primary release adds a usable
  HC/PD diagnosis mapping (then still requires overlap + ≥30-CAL checks).

### PD — OpenNeuro ds007526 — "Resting-State & Walking EEG in Parkinson's Disease" → **PROVISIONALLY ADMISSIBLE**
- Corrected facts (primary snapshot, Amendment 1): **144 subjects = 116 PD + 28 HC** (not 30); **277 recordings**;
  **65 channels**; **250 Hz**; license **CC0**; conditions **resting + walking** (use **resting only**).
- HC-vs-patient: binary contrast present (116 PD + 28 HC). ✓ · montage/Fs/license/raw now largely known (verify raw
  format + exact released version).
- CAL feasibility: 144 subjects → CAL ≈ 80–90 → **≥30 ✓**. Caveat: HC minority (28) → ~17 HC CAL / ~11 HC EVAL —
  workable but HC-light (note for batch class balance / harm-AUROC evaluability).
- Overlap with DEV: distinct study (gait lab) → low risk; **verify subject IDs** vs the seven DEV cohorts.
- **Verdict:** **PROVISIONALLY ADMISSIBLE (PD)**, conditional on: freeze exact OpenNeuro version, participants
  mapping, resting-recording completeness, DEV subject/recording-overlap check, and post-split CAL/EVAL counts.

### SCZ — Zenodo 10.5281/zenodo.14808296 — resting-state SCZ vs HC → **ADMISSIBLE**
- HC-vs-patient: **38 SCZ + 39 HC** → binary contrast. ✓
- Montage/Fs: **64-channel, 1000 Hz** → 10–20-compatible, resamplable to the DEV pipeline. ✓
- Resting condition: **resting-state**. ✓ · License: **CC-BY-4.0**. ✓ · Raw EEG: **available** (ZIP). ✓
- CAL feasibility: 77 subjects → CAL ≈ 45 → **≥30 ✓**.
- Overlap with DEV: distinct (Zenodo, not the four OpenNeuro SCZ cohorts) → low risk; verify subject scheme.
- **Verdict:** **ADMISSIBLE (SCZ)** — strongest candidate; confirm internal file format + subject IDs at ingest.

### SCZ — Zenodo 10.5281/zenodo.14178398 — "ASZED" (African Schizophrenia EEG Dataset) → **PROVISIONAL / DATA-INTEGRITY REVIEW REQUIRED** (Amendment 1)
- HC-vs-patient: **76 SCZ + 77 HC = 153** → binary contrast. ✓ · License **CC-BY-4.0**. ✓ · resting present
  (+ cognitive/MMN/ASSR; **use resting only**). · CAL feasibility (pooled) ≈ 90 → ≥30 ✓.
- Corrected facts: data span **two Nigerian acquisition sites / hospital units, two devices, 16 channels**, sampling
  **200 Hz and 256 Hz** (mixed). → **Site/stratum issue (S10):** the acquisition unit/device is the calibration
  stratum; each unit must independently meet CAL feasibility, **or** the Arm-B coverage claim is only for the pooled
  two-unit mixture (not a single-hospital site-local claim). The 16-ch / mixed-Fs montage also needs explicit
  preprocessing-compatibility mapping to the DEV pipeline.
- **Integrity flag:** a 2026-05 preprint alleges signal-level reuse / domain inversion / biomarker instability in
  ASZED (claims signal-level hashing independent of EDF headers). I found **no verifiable archived full text or
  independent reproduction** — this is **not** sufficient to deem the data invalid, but it is sufficient to **block
  upgrade** to admissible pending the authors' response / a version fix / public audit materials. Any payload-level
  check must be **separately pre-registered as a content-blind integrity audit** — it must NOT silently widen the
  current metadata-only audit's authority.
- **Verdict:** **PROVISIONAL / DATA-INTEGRITY REVIEW REQUIRED (SCZ)** — hold pending integrity clarification +
  per-unit stratum/CAL verification + montage/Fs mapping.

## Admissibility summary vs the two-site rule (S6, resolution 8)

| disease | candidate | verdict | binding gap |
|---|---|---|---|
| PD | ds007020 | **EXCLUDED / UNQUALIFIED** (no documented usable HC-vs-PD label; mortality labels only) | — |
| PD | ds007526 | **PROVISIONALLY ADMISSIBLE** (144=116 PD+28 HC, 65ch, 250Hz, CC0; verify version/mapping/overlap) | only candidate |
| SCZ | Zenodo 14808296 | **ADMISSIBLE** (38+39, 64ch, 1000Hz, resting, CC-BY, raw) | — |
| SCZ | ASZED 14178398 | **PROVISIONAL / DATA-INTEGRITY REVIEW REQUIRED** (2 units/devices, 16ch, 200/256Hz; integrity flag) | — |

**Decisive finding:** PD currently has **at most ONE** admissible held-out site (ds007526; ds007020 excluded on the
**label** gate). Per the S6 two-site rule PD would fall to the **single-site contingency** → G2 for PD *site-specific,
no within-disease external replication*. **Recommendation:** source **one additional independent PD lockbox** (HC+PD,
resting, raw, ≥30 CAL, no DEV overlap) before freeze; do **not** force two sites with small/overlapping/label-
incompatible data — otherwise pre-register PD single-site. SCZ has **one solid** (14808296) + **one held for
integrity review** (ASZED); if ASZED does not clear, SCZ may also fall to single-site. The stronger framing if the
gaps persist: **SCZ = replicated external claim, PD = single-site confirmatory.**

## Verification checklist before lockbox designation (still metadata-only)

1. ds007526 + ASZED: fetch primary `dataset_description.json` / `participants.tsv` (OpenNeuro file API) and the
   Zenodo file manifest for: channel montage, sampling rate, raw-signal format, license confirmation, resting label.
2. ASZED: clarify the **two acquisition units/devices** (per-unit montage/Fs, per-unit CAL feasibility) and the
   integrity flag; if a payload-level check is needed, **separately pre-register a content-blind integrity audit**.
3. Subject-ID / recording overlap check of every retained site against the seven DEV cohorts (no shared subjects).
4. Confirm ≥30 CAL subjects under the pre-registered subject-hash split for each retained site/unit (record counts;
   prefer well above the ~9 mathematical minimum).
5. Source an additional PD site (or pre-register PD single-site). ds007020 stays excluded unless a primary release
   documents a usable HC/PD label.

No ACAR inference, adaptation-outcome inspection, or endpoint access was or will be performed before
`ACAR_FROZEN_v3.md` is committed and tagged on the `acar` lineage on a clean worktree.

## Sources
- ds007020: https://openneuro.org/datasets/ds007020 · metadata (94 subj/500Hz/mortality labels) https://eegdash.org/api/dataset/eegdash.dataset.DS007020.html · mortality paper https://www.medrxiv.org/content/10.1101/2025.07.07.25331047
- ds007526: https://openneuro.org/datasets/ds007526/versions/1.0.2 · subtypes paper https://movementdisorders.onlinelibrary.wiley.com/doi/abs/10.1002/mds.70348
- SCZ resting: https://zenodo.org/records/14808296
- ASZED: https://zenodo.org/records/14178398 · integrity-flag preprint (UNVERIFIED, no archived full text found): ResearchGate figure ref 394114307
- NB: OpenNeuro `participants.tsv`/`dataset_description.json` still to be fetched for primary confirmation (Amendment 1 used eegdash + OpenNeuro snapshot + Zenodo records).
