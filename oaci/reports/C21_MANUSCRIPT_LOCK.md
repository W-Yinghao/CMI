# C21 — Manuscript Lock

**Title:** When Source-Side Signals Do Not Transfer: Falsification, Observability, and Estimand Boundaries in EEG Domain Generalization

> STORY FROZEN. No further empirical branches (no new experiments / no probe tuning / no estimand swap / no BNCI2014_004 / no seeds [3,4] / no retrain / no selector / no new DG penalty). The chain C14->C20 is the final paper object.

## Evidence chain
- **C14**: OACI-control / SRC-control closed by falsification battery  (verdict falsified)
- **C16**: target-accuracy-good checkpoints exist but source-unobservable; calibration barrier; SRC memorization  (6/6 oracle gain; source-audit oracle fails)
- **C17**: no strong scalar source signal; weak multivariate signal; boundary source-mirrored; calibration-biased  (LOTO 0.6023 p=0.008; boundary corr +0.547)
- **C18**: weak signal survives cell-present stress; deletion collapse = accuracy-endpoint non-estimability  (S2/S3 beat perm; leakage/boundary robust)
- **C19**: pre-registered robust-core probe recovers weak IN-REGIME competence  (pooled 0.561 p=0.005)
- **C20**: frozen cross-regime validation NOT established; signal largely regime-local  (pooled 0.50-0.54; Holm 0/4; Simpson)

## Canonical conclusion
> Under strict target isolation, the tested source-side DG control objectives do not transfer to reproducible target worst-domain improvements. However, the failure is not a simple absence of target-good states: OACI trajectories contain target-accuracy-good checkpoints that are invisible to simple source-audit selectors. Source-only competence information is weakly present in multivariate, deletion-robust observables and can be recovered by a pre-registered low-freedom diagnostic probe in-regime. Yet this signal does not establish stable cross-regime external validation: held-out support-stress regimes reveal that within-target ranking information does not transport as a pooled cross-target competence estimand. The contribution is therefore a falsification and observability framework, not a deployable target-free selector.
