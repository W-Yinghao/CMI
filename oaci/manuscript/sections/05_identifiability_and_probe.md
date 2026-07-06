# 05 Identifiability And Probe

> Scope: C17 weak multivariate identifiability; C18 support-stress + endpoint-estimability; C19 pre-registered in-regime weak positive

- **[E4 · established]** No strong scalar source signal identifies target-good checkpoints; a WEAK multivariate source-only signal exists; class-boundary rotation is source-mirrored; source signals are calibration-biased.  
  *evidence:* C17_SOURCE_SIGNAL_IDENTIFIABILITY (a8af8c6) — LOTO probe AUC 0.6023 beats perm p=0.008; best scalar |ρ|<=0.236; boundary corr +0.547
- **[E5 · established]** The weak multivariate signal SURVIVES cell-present support stress and collapses only under cell DELETION, and there because the worst-domain accuracy ENDPOINT becomes non-estimable (estimator-level), not because the signal vanished; leakage and class-boundary mirror are support-robust.  
  *evidence:* C18_CONTROLLED_SUPPORT_MISMATCH_STRESS (8046929) — S2 0.603 / S3 0.562 beat perm; S4/S6/S7 bAcc->NaN; leakage estimability 1.0
- **[E6 · established]** A PRE-REGISTERED low-freedom robust-core probe (deletion-robust source observables) recovers weak IN-REGIME competence information (config hash frozen before the run).  
  *evidence:* C19_SOURCE_ONLY_COMPETENCE_PROBE (0eebae5) — robust-core LOTO 0.561 beats perm p=0.005 margin>=0.03 on S0/S2/S3; per-target 0.57-0.73

TODO: prose (this is a locked-evidence scaffold, not finished text).