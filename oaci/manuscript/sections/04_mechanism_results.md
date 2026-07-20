# 04 Mechanism Results

> Scope: C16: target-good checkpoints exist but source-unobservable; calibration barrier; SRC memorization

- **[E2 · established]** Target-accuracy-good OACI checkpoints EXIST but are invisible to simple source-audit selectors (source-side OBSERVABILITY failure), with a separate calibration barrier.  
  *evidence:* C16_TARGET_ORACLE_CEILING (38206d6) — 6/6 seed-levels reproducible bAcc gain via non-deployable target oracle; source-audit oracle fails
- **[E3 · established]** Selected OACI is calibration-improved / accuracy-flat (class-boundary rotation); SRC anti-transfer is source memorization.  
  *evidence:* C16_HARM_DECOMPOSITION (0eedfee) — mean ΔNLL -0.074, ΔbAcc -0.002; SRC memorization index +1.965 (6/6 flagged)

TODO: prose (this is a locked-evidence scaffold, not finished text).