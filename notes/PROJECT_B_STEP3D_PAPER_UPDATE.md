# Project B Step-3D: Paper Update

Merges the Step-3C **bounded real benchmark expansion** into the Project B paper package. Paper
integration only — no experiment re-run, no `h2cmi/**` or `cmi/**` change.

## What changed
- `PROJECT_B_PAPER_DRAFT.md` §10 split into **10.1 Step-3A bridge smoke** and **10.2 Step-3C bounded
  real benchmark expansion**; §11 claim boundary and §12 limitations updated for the real result.
- `PROJECT_B_CLAIM_BOUNDARY.md`: claimable 7 items
  (adds two Step-3C claims), not-claimable 8 items
  (adds "not a full MOABB benchmark" + two harmful-regime caveats).
- `PROJECT_B_REVIEWER_CHECKLIST.md`: adds Q8–Q10.
- New paper tables: Table 5 (real aggregate), Table 6 (real reason audit), Table 7 (representative
  real per-domain examples), plus `*_v2.tex` and `*_v2.json`.

## Headline real numbers (BNCI2014_004, baseline support mode)
- Raw offline TTA mean d_bAcc = -0.140 (harmful); OFFLINE_TTA
  rate = 0.00.
- Subject routing: coverage 0.50, identity_rate
  0.50, accepted_bAcc 0.618.
- Session routing: coverage 0.40, identity_rate
  0.40, accepted_bAcc 0.614.
- Dominant refusal driver `SUPPORT_MISMATCH`; LOW_ESS inactive; nested support inert.

## Boundary
Bounded real benchmark expansion, **not a full benchmark**. Demonstrates harm avoidance and
refusal/identity routing under a harmful-TTA regime; does not establish accuracy improvement over
identity or beneficial-TTA recovery on real EEG.

## Inputs (read-only, frozen)
- Step-2G: `/tmp/project_b_step2g_report` · Step-3A: `/tmp/project_b_step3a_real_bridge` · Step-3C: `/tmp/project_b_step3c_real_benchmark`
- Outputs: `/tmp/project_b_step3d_paper_update`
