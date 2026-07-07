# S2P_01 — TUEG Corpus Inventory (Phase 9A)

**Project S2P — Phase 9A.** Inventory of the processed pretraining corpus for the subject-scaling study. Source:
`/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG` (config `4704743c`). Pure inventory; no
pretraining. Script `s2p/scripts/tueg_inventory.py`; artifacts `results/s2p_inventory/`.

## Headline — corpus side GREEN
| property | value |
|---|---|
| dataset / config | **TUEG** / 4704743c |
| **unique subjects** | **13,446** |
| recordings | 57,700 |
| **total hours** | **23,187** |
| sfreq | 200 Hz (preproc: 0.5–45 Hz band, EEG-only) |
| channels stored | 33 (TUH 10-20 `-LE` montage + extras: EKG/PHOTIC/SP1/SP2/…) |
| **subject IDs reliable** | **YES** — processed `subject` index + `infos.json.original_subjects` maps `sub_N` → original TUH patient id (e.g. `aaaaaacj`) |
| **sample-by-subject feasible** | **YES** — `recordings/sub_<S>_ses_<E>_run_<R>.npy` + `metadata.parquet` (subject/session/run/n_timepoints/channels/filepath) |

**This decisively passes the S2P go/no-go gate** ("if subject ID unreliable → STOP"): IDs are stable and
subject-resolvable.

## Per-subject hours distribution
Most TUEG recordings are short; the long tail is what matters for a fixed-hours grid.
| ≥ threshold | subjects |
|---|---|
| ≥0.25 h | 13,211 |
| ≥0.5 h | 4,568 |
| ≥1 h | 2,691 |
| ≥2 h | 1,425 |
| ≥5 h | 778 |
| ≥10 h | 406 |
(median per-subject hours is low; mean pulled up by the tail; max large.)

## Fixed-hours subset feasibility (the key design input) — all feasible
`fixed_hours_feasible` = (≥ N subjects have ≥ H0/N hours). Every grid cell we need is feasible:
| H0 | N=32 | N=128 | N=512 |
|---|---|---|---|
| 250 h | cap 7.8 h, 509 avail ✓ | cap 1.95 h, 1444 ✓ | cap 0.49 h, 4614 ✓ |
| 500 h | cap 15.6 h, 280 ✓ | cap 3.91 h, 915 ✓ | cap 0.98 h, 2756 ✓ |
So **H0 ∈ {250, 500} h × N ∈ {32,128,512}** is comfortably supported; even N=2048 is feasible at H0≤500 h.

## Caveat (design decision, not a blocker) — 19-common-channel coverage 21%
Only **21%** of recordings contain the full **19 common 10-20 channels** (CBraMod/CodeBrain pretraining montage) —
TUH montages vary. Options for the pretraining channel pipeline (to be pinned in S2P_02/03):
- **(a) restrict** to the 19-common-covered recordings (~21% ⇒ ~4,900 h, ~2,800 subjects — still ample for the
  grid), maximally reproducible; **recommended default**; or
- **(b) channel-flexible** input (the encoders are conv-over-channel) with a pinned channel-selection/imputation
  rule; larger corpus but a preprocessing variable.
Whichever is chosen must be **hashed and fixed a priori** (subject-scaling contrast requires a constant channel
pipeline across N).

## Other configs present (for reference)
TUEG also under `0beed0bd/TUEG`, `55b3ae9c/TUEG` (different preprocessing hashes). We pin **4704743c** (200 Hz,
0.5–45 Hz) as the S2P substrate; alternatives noted for a sensitivity check only.

## Artifacts (`results/s2p_inventory/`)
`tueg_subject_manifest.csv` (per subject: n_recordings, n_sessions, total_hours, frac_common19),
`tueg_hours_by_subject.csv`, `fixed_hours_subset_feasibility.csv`, `tueg_inventory_summary.json`.
