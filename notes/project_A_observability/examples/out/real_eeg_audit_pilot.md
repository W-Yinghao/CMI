# Observability report — REAL-EEG PILOT (interface-only, placeholder metrics) — BNCI2014_001 2a LOSO

- claims: **8**  ·  allowed: **8**  ·  rejected: **0**
- forbidden-claim violations: **0** (must be 0)

| # | claim | regime | estimand | verdict | licensing / certificate | contracts (checkable / assumed) | reason |
|---|---|---|---|---|---|---|---|
| 1 | strict_dg.target_bacc | R0 | balanced_accuracy | 📊 reportable (oracle/eval-only) |  | — / — | oracle/evaluation-only target bAcc; not adaptation evidence; not identifiable from R0/R1 |
| 2 | strict_dg.worst_domain_bacc | R0 | balanced_accuracy | 📊 reportable (oracle/eval-only) |  | — / — | oracle/evaluation-only target bAcc; not adaptation evidence; not identifiable from R0/R1 |
| 3 | offline_tta.adaptation_gain | R1 | target_gain | 📊 reportable (oracle/eval-only) |  | — / — | oracle/evaluation-only; not adaptation evidence; not identifiable from R0/R1 |
| 4 | offline_tta.target_prior | R1 | target_prior | ✅ identifiable | TU-1 / CE-R1-2 | C1,C3 / C2 | target prior identifiable under TU-1 (C1∧C2∧C3) |
| 5 | online_tta.target_bacc | R1 | balanced_accuracy | 📊 reportable (oracle/eval-only) |  | — / — | oracle/evaluation-only target bAcc; not adaptation evidence; not identifiable from R0/R1 |
| 6 | leakage.site | R0 | leakage | ✅ identifiable | P0-2 | — / — | leakage is a diagnostic, not a risk/accuracy guarantee (fidelity governed by C5) |
| 7 | leakage.subject | R0 | leakage | ✅ identifiable | P0-2 | — / — | leakage is a diagnostic, not a risk/accuracy guarantee (fidelity governed by C5) |
| 8 | leakage.session | R0 | leakage | ✅ identifiable | P0-2 | — / — | leakage is a diagnostic, not a risk/accuracy guarantee (fidelity governed by C5) |

## Forbidden claims checked (05 §6)

- target concept shift detected from unlabeled target — **not made**
- source-only target safety certified — **not made**
- CMI/leakage guarantees accuracy — **not made**
- GLS gives the target prior source-only — **not made**
- R0 source metric reported as a target risk/gain/concept guarantee — **not made**
- R1 unlabeled-target balanced accuracy reported as identifiable target metric — **not made**

