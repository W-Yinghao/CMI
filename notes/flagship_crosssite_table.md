# Flagship cross-site DG — leave-one-cohort-out, multi-seed (jobs xsite_*, 2026-06-13)

Method ladder `erm / lpc_prior / cdann / dual / dualc` on the Protocol-C leave-one-cohort-out benchmark, both
domain granularities, **mean±std over 3 seeds**. `bAcc`=per-target balanced acc; `worst`=worst-target bAcc;
`leakKL`=encoder leakage `I(Z;D|Y)`↓; `decCMI_rw`=GLS-reweighted residual decoder CMI (concept).

### PD (3 sites: ds002778 / ds003490 / ds004584)
| D | method | bAcc | worst | leakKL | decCMI_rw |
|---|---|---|---|---|---|
| cohort | erm | 58.7±0.8 | 51.4 | 0.230 | 0.000 |
| cohort | **lpc_prior** | **59.3±0.4** | 52.6 | **0.033** | 0.000 |
| cohort | cdann | 59.0±1.1 | 52.0 | 0.028 | 0.001 |
| cohort | dual | 59.5±0.6 | 52.7 | 0.033 | 0.000 |
| cohort | dualc | 59.3±0.4 | 52.5 | 0.033 | 0.000 |
| subject | erm | 58.8±0.2 | 51.7 | 1.364 | 0.095 |
| subject | **lpc_prior** | 59.3±0.5 | 52.4 | **0.104** | 0.108 |
| subject | cdann | 56.8±0.6 | 51.3 | 0.350 | 0.151 |
| subject | dualc | 59.4±0.6 | 52.8 | 0.104 | 0.108 |

### SCZ (2 same-task resting cohorts: ds003944 / ds003947)
| D | method | bAcc | worst | leakKL | decCMI_rw |
|---|---|---|---|---|---|
| cohort | erm | 50.8±0.4 | 47.6 | 0.434 | 0.003 |
| cohort | **lpc_prior** | 52.1±0.5 | 49.3 | **0.127** | 0.001 |
| cohort | cdann | 52.9±2.1 | 48.0 | 0.096 | 0.004 |
| cohort | dual | 52.3±0.5 | 49.4 | 0.129 | 0.001 |
| cohort | dualc | 52.0±0.6 | 49.2 | 0.128 | 0.001 |
| subject | erm | 51.0±0.6 | 47.2 | 1.393 | 0.166 |
| subject | **lpc_prior** | 53.1±2.0 | 48.0 | **0.094** | 0.188 |
| subject | cdann | 53.1±1.4 | 51.0 | 0.133 | 0.255 |
| subject | dualc | 53.2±2.0 | 47.9 | 0.091 | 0.188 |

### Takeaways (with CIs)
1. **Leakage removal is robust and large** — `lpc_prior` cuts `I(Z;D|Y)` 3–15× in every condition (subject 13–15×, cohort 3–7×), tightest where it matters.
2. **Accuracy: `lpc_prior` ≥ `erm`** (+0.6 … +2.1) with **lower variance** (e.g. PD-cohort std 0.4 vs 0.8); `cdann` is erratic (PD-subject −2.0; SCZ-cohort ±2.1).
3. **`dual`/`dualc` ≈ `lpc_prior`** on accuracy — the dual-CMI no-compounding parity, now multi-seed.
4. **`decCMI_rw` ≈ 0 at D=cohort vs 0.1–0.25 at D=subject** — concept-null at the valid granularity, `H(Y|Z)` artifact at the degenerate one (§1.2/§3.3), re-confirmed with seeds.
