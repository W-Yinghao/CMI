# S2P downstream audit plan (Phase 9A)

Per pretrained checkpoint x downstream dataset (SHU-MI primary decodable; PhysioNetMI large/weak; BNCI sanity):
frozen encoder -> F1 spatial -> fixed-a-priori PCA (source-only) -> linear head (source-val selected).
Metrics (reuse FSR-hardened): target bAcc/macro-F1; L1 = mean PAIRWISE subject separability (dimension-invariant,
2-way, run/session-held-out, on HELD-OUT source subjects); L4 task-head/subject-subspace alignment; L5 subject-
subspace erase vs VARIANCE-MATCHED null (per removed-var); L6 target consequence. Per-cell task gate (source-val
>=0.58) -> L4/L5/L6 interpreted else WEAK_TASK_NOT_INTERPRETED. Per-dataset only. Slopes vs log(N_subjects) with
CIs CLUSTERED by eval subject + hours covariate. Target labels final-scoring only.
