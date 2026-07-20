# C16-A — Target-oracle ceiling (does OACI's trajectory contain a target-good checkpoint?)

> The target oracle is a **NON-DEPLOYABLE, post-hoc diagnostic**: it reads target_audit only to test checkpoint EXISTENCE, never as a selection method.

- **CASE: `C3_calibration_not_discrimination`** — target oracle recovers accuracy but not calibration

| selector | K2 (worst-held-out-target) | reproduced | deployable |
|---|---|---|---|
| ERM | `stop_no_reproducible_gain` | — | yes |
| source_audit_oracle | `stop_no_reproducible_gain` | — | yes |
| target_oracle_bacc | `reproducible_gain` | ['worst_domain_bacc'] | NO (diagnostic) |
| target_oracle_joint | `stop_no_reproducible_gain` | — | NO (diagnostic) |

## What this splits from C10's case C

- source-audit oracle rescues K2: **False** (reproduces C10 case C)
- target oracle rescues worst-domain **bAcc**: **True** → target-accuracy-good checkpoints **DO exist** in the trajectory but are **not source-observable**
- target oracle rescues **joint** (bAcc+NLL): **False** → a separate **calibration barrier**: even the accuracy-optimal checkpoints do not jointly improve NLL

## Target-oracle bAcc ceiling — worst-held-out-target Δ vs ERM (per seed, level)

| seed | level | Δ worst bAcc | Δ worst NLL |
|---:|---:|---:|---:|
| 0 | 0 | +0.0191 | -0.4636 |
| 0 | 1 | +0.0035 | +0.2845 |
| 1 | 0 | +0.0313 | -0.1320 |
| 1 | 1 | +0.0365 | +0.1842 |
| 2 | 0 | +0.0295 | -0.4234 |
| 2 | 1 | +0.0139 | -0.4443 |

> **Mechanism.** The measurement→control decoupling for *accuracy* is a **source-side observability failure** (the good checkpoints exist; source signal cannot identify them), while *calibration* harm persists even at the target-accuracy ceiling. This is a diagnostic result; the target oracle is never deployable and makes no selection claim.