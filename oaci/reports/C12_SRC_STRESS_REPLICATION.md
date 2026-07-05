# C12 — SRC stress replication (BNCI2014-001 seed-0; targets [1, 3, 5], τ_lse [0.1, 0.3])

> Last SRC round. Narrow question: is target-001's SRC failure a single-fold fluke, or does source-side control fail to transfer across folds?

- configs loaded: **6** (12 target×temp×level cells)
- **VERDICT: `stop_SRC_pivot_measurement_only`**
- pivot triggers: target NLL blowup in 6/12 cells (SRC confidently wrong on target); source-side improvement did NOT transfer to target in 6 cells

## Table 1 — SRC τ=0.1 vs τ=0.3 (target worst NLL; blowup if > uniform 1.386 or > ERM+0.5)

| target | level | ERM NLL | SRC NLL τ0.1 | blow | SRC NLL τ0.3 | blow |
|---:|---:|---:|---:|:--:|---:|:--:|
| 1 | 0 | +1.2786 | +2.4487 | YES | +2.4454 | YES |
| 1 | 1 | +1.2208 | +1.2208 | - | +1.2208 | - |
| 3 | 0 | +0.9540 | +1.0820 | - | +1.2330 | - |
| 3 | 1 | +0.9149 | +0.9149 | - | +0.9149 | - |
| 5 | 0 | +2.4257 | +4.3498 | YES | +3.1311 | YES |
| 5 | 1 | +2.0667 | +2.0667 | YES | +2.0667 | YES |

## Table 2 — SRC target worst-domain Δ vs ERM (bAcc↑ NLL↓ better)

| target | temp | level | Δ bAcc | Δ NLL | fallback | risk-feasible |
|---:|---:|---:|---:|---:|:--:|:--:|
| 1 | 0.1 | 0 | -0.0451 | +1.1701 | - | True |
| 1 | 0.1 | 1 | +0.0000 | +0.0000 | ERM | True |
| 1 | 0.3 | 0 | -0.0538 | +1.1668 | - | True |
| 1 | 0.3 | 1 | +0.0000 | +0.0000 | ERM | True |
| 3 | 0.1 | 0 | +0.0122 | +0.1281 | - | True |
| 3 | 0.1 | 1 | +0.0000 | +0.0000 | ERM | True |
| 3 | 0.3 | 0 | -0.0104 | +0.2790 | - | True |
| 3 | 0.3 | 1 | +0.0000 | +0.0000 | ERM | True |
| 5 | 0.1 | 0 | -0.0156 | +1.9241 | - | True |
| 5 | 0.1 | 1 | +0.0000 | +0.0000 | ERM | True |
| 5 | 0.3 | 0 | -0.0052 | +0.7054 | - | True |
| 5 | 0.3 | 1 | +0.0000 | +0.0000 | ERM | True |

## Table 3 — fallback frequency

- SRC fell back to ERM in **6/12** cells; active (trained-ckpt) cells **6**

- τ=0.1: target NLL blowup in 3/6 cells
- τ=0.3: target NLL blowup in 3/6 cells

## Table 4 — source-side improvement vs target transfer

| target | temp | level | ΔsrcGuard NLL | Δtarget NLL | transferred? |
|---:|---:|---:|---:|---:|:--:|
| 1 | 0.1 | 0 | -1.1256 | +1.1701 | NO |
| 1 | 0.1 | 1 | +0.0000 | +0.0000 | - |
| 1 | 0.3 | 0 | -1.0504 | +1.1668 | NO |
| 1 | 0.3 | 1 | +0.0000 | +0.0000 | - |
| 3 | 0.1 | 0 | -0.9217 | +0.1281 | NO |
| 3 | 0.1 | 1 | +0.0000 | +0.0000 | - |
| 3 | 0.3 | 0 | -0.9979 | +0.2790 | NO |
| 3 | 0.3 | 1 | +0.0000 | +0.0000 | - |
| 5 | 0.1 | 0 | -1.3086 | +1.9241 | NO |
| 5 | 0.1 | 1 | +0.0000 | +0.0000 | - |
| 5 | 0.3 | 0 | -1.0134 | +0.7054 | NO |
| 5 | 0.3 | 1 | +0.0000 | +0.0000 | - |

## Interpretation

> SRC fails to transfer across folds (see pivot triggers). Combined with C10 (leakage/oracle) and C11 (endpoint), THREE source-side interventions fail -> **STOP SRC. Pivot to measurement-only / source-target-instability (C13 memo).** Keep support-aware leakage + K1/K2 as the falsification instrument; do not build another DG control penalty.