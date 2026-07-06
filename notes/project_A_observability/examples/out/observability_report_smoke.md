# Observability report — Project A smoke — 5-claim ledger

- claims: **5**  ·  allowed: **3**  ·  rejected: **2**
- forbidden-claim violations: **0** (must be 0)

| # | claim | regime | estimand | verdict | licensing / certificate | contracts (checkable / assumed) | reason |
|---|---|---|---|---|---|---|---|
| 1 | source_loso_bacc | R0 | source_loso | ✅ allowed |  | — / — | source-side quantity; identifiable in R0+ (source law observed) |
| 2 | target_gain_r0 | R0 | target_gain | ⛔ rejected | CE-R0-2 | — / — | target risk/gain non-identifiable under R0 (TOS-1) _(demo, not a conclusion)_ |
| 3 | target_prior_r1 | R1 | target_prior | ✅ allowed | TU-1 / CE-R1-2 | C1,C3 / C2 | target prior identifiable under TU-1 (C1∧C2∧C3) |
| 4 | target_concept_r1 | R1 | target_concept | ⛔ rejected | CE-R1-1 | — / — | concept non-identifiable from unlabeled target (TU-2) _(demo, not a conclusion)_ |
| 5 | transport_r2 | R2 | target_transport | ✅ allowed | MP-1 / CE-MP-1 | C11,C8 / — | transport identifiable (or bounded) under MP-1 (C8∧C11) |

## Forbidden claims checked (05 §6)

- target concept shift detected from unlabeled target — **not made**
- source-only target safety certified — **not made**
- CMI/leakage guarantees accuracy — **not made**
- GLS gives the target prior source-only — **not made**
- R0 source metric reported as a target risk/gain/concept guarantee — **not made**
- R1 unlabeled-target balanced accuracy reported as identifiable target metric — **not made**

