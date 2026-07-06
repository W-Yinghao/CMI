# 07 Discussion Limitations

> Scope: single dataset/backbone; diagnostic-only; future work = calibration estimand + external protocol

- **[D1 · diagnostic_only]** The C19/C20 competence probe is DIAGNOSTIC-ONLY: it emits no selector and makes no deployment claim.  
  *evidence:* C19/C20 (0eebae5/7b38bee) — no_selector_artifact gate = True in C19 and C20
- **[D2 · diagnostic_only]** Target-oracle labels and source-signal identifiability analyses are diagnostic-only, joined post hoc.  
  *evidence:* C16/C17 (38206d6) — diagnostic_only_non_deployable=True throughout
- **[N1 · not_established]** No deployable target-free selector / competence detector is established (diagnostic-only; no-selector gate).  
  *evidence:* C19/C20 (7b38bee) — no-selector gate; C20 external validation not established
- **[N2 · not_established]** External / new-regime dataset generalization of the probe is NOT established.  
  *evidence:* C20 (7b38bee) — cross-regime pooled near chance; Holm 0/4
- **[N3 · not_established]** Support mismatch as a NATURAL cause in BNCI2014_001 is NOT demonstrated (masks are controlled).  
  *evidence:* C17/C18 (a8af8c6) — controlled masks only; BNCI001 not shown naturally support-mismatched
- **[N4 · not_established]** We do NOT make blanket over-claims (all-DG-failure, EEG-transfer-impossibility, or support-aware-invariance worthlessness); the falsification is protocol-scoped (single dataset / single backbone).  
  *evidence:* C14/C18 (8046929) — single dataset/backbone; falsification is protocol-scoped
- **[F1 · future_work]** Score CALIBRATION / a target-free competence ESTIMAND that transports cross-regime: within-target ranking appears stable (~0.64) and suggests a future score-calibration question.  
  *evidence:* C20 (7b38bee) — within-target mean AUC ~0.63-0.65 stable; pooled does not transport
- **[F2 · future_work]** External-dataset validation on a genuinely new cohort, via a pre-registered protocol (C20-B) -> C21+ execution ONLY after approval. BNCI2014_004 remains BARRED.  
  *evidence:* C20_EXTERNAL_DATASET_PROTOCOL (89b8a6a) — protocol only; no execution; BNCI2014_004 BARRED_pending_explicit_approval
- **[F3 · future_work]** Real support-mismatched clinical EEG validation (hierarchical D=(cohort,subject)) is future work.  
  *evidence:* C17/SCPS (a8af8c6) — PD/SCZ cohorts not provisioned; clinical loader offline-only

TODO: prose (this is a locked-evidence scaffold, not finished text).