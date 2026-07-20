# C19 — Source-Only Competence Probe (diagnostic-only, config `664007686afb520f`)

> Pre-registered low-freedom probe. NOT a target-free selector, NOT an OACI rescue, NOT a deployable competence detector. Tests whether the weak C17/C18 source-only signal survives pre-registration + feature discipline using DELETION-ROBUST observables, with endpoint-estimability as a first-class output.

- **CASE: `robust_core_recovers_weak_competence`**  ·  primary_success (robust-core beats perm+margin on S0/S2/S3): **True**
- next science: C20 = external / new-regime validation of the diagnostic probe (NOT a DG penalty, NOT a selector).

## Robust-core (primary) — LOTO vs within-fold permutation

| regime | n_used | n_feat | loto_auc | perm_mean | perm_p | margin≥.03 | passes |
|---|---:|---:|---:|---:|---:|:--:|:--:|
| S0_full_support | 1268 | 16 | +0.561 | +0.476 | +0.005 | True | True |
| S2_rare_cells | 1268 | 16 | +0.537 | +0.481 | +0.005 | True | True |
| S3_nonestimable_cells | 1268 | 16 | +0.533 | +0.471 | +0.005 | True | True |

## Endpoint-estimability gate (first-class output)

| regime | scored_rate | endpoint_nonestimable_rate (aug) |
|---|---:|---:|
| S0_full_support | +1.000 | +0.510 |
| S2_rare_cells | +1.000 | +0.510 |
| S3_nonestimable_cells | +1.000 | +0.510 |

## Endpoint-augmented (secondary, only where estimable)

| regime | n_endpoint_estimable | loto_auc | perm_p | passes |
|---|---:|---:|---:|:--:|
| S0_full_support | 621 | +0.553 | +0.005 | True |
| S2_rare_cells | 621 | +0.534 | +0.095 | False |
| S3_nonestimable_cells | 621 | +0.530 | +0.015 | True |

## Per-target heterogeneity (S0 robust-core)

- spread +0.161 (min +0.570, max +0.732) → heterogeneous: **False**

## Gates

- preregistration hash `664007686afb520f` · static excluded from primary **True** · fragile endpoints excluded **True** · targets post-hoc **True** · finite-filter **True** · no selector artifact **True**

> DIAGNOSTIC-ONLY. A positive result means only that a pre-registered low-freedom source-only diagnostic probe recovers weak competence information; it is NOT evidence of a deployment-ready target-free checkpoint chooser, and no such artifact is produced.