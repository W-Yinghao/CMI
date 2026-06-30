# paper/cigl/ — CIGL manuscript draft workspace (Phase 4B)

**Draft workspace, not a submission.** Writing only — **no generated PDFs, no new experiments, no GPU**.

Every claim here must trace to the tracked evidence:
- Method/claims: `docs/CIGL_32_METHOD_FRAMING_AND_CLAIMS.md`
- Evidence index: `docs/CIGL_33_EVIDENCE_INDEX.md`
- Reviewer risks: `docs/CIGL_34_REVIEWER_RISK_REGISTER.md`
- Blueprint: `docs/CIGL_35_PAPER_BLUEPRINT.md`
- Reproducibility: `docs/CIGL_36_REPRODUCIBILITY_CHECKLIST.md`
- Primary results: `docs/CIGL_29_PHASE3A_J_RESULTS_BNCI2014_001.md` (BNCI2014_001),
  `docs/CIGL_31_PHASE3A_K_RESULTS_BNCI2015_001.md` (BNCI2015_001),
  `docs/CIGL_25_PHASE3A_H_...` (DGCNN leakage audit).

## Files

- `MANUSCRIPT_DRAFT.md` — first full paper skeleton (8 sections, paragraph-level prose, table/figure
  placeholders).
- `CLAIMS_AUDIT.md` — claim → evidence → allowed/forbidden wording → status.
- `TABLES_AND_FIGURES_PLAN.md` — T1–T5 / F1–F4 with source, claim supported, claim NOT supported, status.
- `RELATED_WORK_MATRIX.md` — positioning vs related work (`TODO: verify citation` where unverified).
- `OPEN_PAPER_BLOCKERS.md` — candidate blockers classified (must-fix / optional ablation / future / do-not-do).

## Hard rules (Phase 4B)

- **No** GPU / EEG training / new dataset / ablation / λ-grid / edge-CMI / CITA-DualPC-Tri-CMI changes.
- **No** invented citations or BibTeX — uncertain references are marked `TODO: verify citation`.
- The **bounded claim** is fixed: partial, reproducible graph/node leakage reduction at source-task
  retention, two MI datasets, DGCNN static-adjacency backbone, posterior-KL proxy. **No** SOTA /
  elimination / unbiased-CMI / edge-CMI / cross-architecture / beyond-MI / λ-robustness claim.
