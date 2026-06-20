# Tri-CMI — Auto-generated Results Summary

_Generated 2026-06-09 18:41 from 117 result files (`python -m analysis.summarize_results`). Metric: subject-level bAcc for SCPS, per-target for MCPS._

## 1. EEGNet: CE vs CE+CMI — one fair row per dataset (best-λ where a sweep exists)

| dataset | #λ | ERM | +CMI(best λ) | Δacc | ERM leak | +CMI leak |
|---|---|---|---|---|---|---|
| ADFTD | 4 | 53.0 | 59.5 (0.3) | +6.5 | 1.54 | 0.11 |
| ADFTD_bin | 4 | 83.0 | 83.4 (0.1) | +0.3 | 1.08 | 0.22 |
| BNCI2014_001 | 5 | 52.0 | 51.4 (0.05) | -0.7 | 1.20 | 0.61 |
| BNCI2014_001_imb | 2 | 46.5 | 44.9 (0.3) | -1.6 | 1.15 | 0.11 |
| BNCI2014_004 | 5 | 68.6 | 69.2 (0.05) | +0.6 | 0.53 | 0.08 |
| BNCI2014_004_imb | 2 | 65.7 | 66.3 (0.3) | +0.6 | 0.48 | 0.02 |
| Cho2017 | 2 | 64.7 | 64.5 (0.1) | -0.2 | 0.73 | 0.04 |
| DEAP ⚠1λ | 1 | 50.6 | 51.1 (0.5) | +0.4 | 0.94 | 0.00 |
| DEAP_arousal ⚠1λ | 1 | 50.1 | 50.0 (0.3) | -0.1 | 0.96 | 0.00 |
| DEAP_quadrant ⚠1λ | 1 | 25.6 | 25.6 (0.3) | +0.0 | 1.48 | 0.00 |
| Lee2019_MI | 2 | 69.1 | 68.9 (0.1) | -0.3 | 0.65 | 0.10 |
| MUMTAZ | 4 | 85.4 | 89.3 (0.5) | +3.8 | 1.59 | 0.02 |
| SEED | 3 | 55.3 | 54.1 (0.5) | -1.2 | 0.67 | 0.05 |
| SEED_IV | 2 | 35.0 | 34.9 (0.3) | -0.1 | 0.65 | 0.03 |
| Schirrmeister2017 | 2 | 61.4 | 60.0 (0.3) | -1.4 | 1.42 | 0.06 |
| TUAB ⚠1λ | 1 | 62.5 | 65.0 (0.3) | +2.5 | 1.42 | 0.04 |
| xdata ⚠1λ | 1 | 66.4 | 65.5 (0.3) | -0.8 | 0.82 | 0.04 |

*+CMI ≥ parity (Δ>−0.5) on 12/17 datasets at best-available λ; leakage drops everywhere. ⚠1λ = only one λ run (often 0.3, over-regularized) so not a fair best-λ.*

## 2. Worst-subject robustness (ERM vs lpc_prior, small λ)

| run | ERM worst | lpc_prior worst | Δ |
|---|---|---|---|
| ADFTD_EEGNet_lamsweep | 0.0 | 0.0 (0.05) | +0.0 |
| BNCI2014_001_EEGNet_lamsweep | 35.9 | 39.1 (0.05) | +3.1 |
| BNCI2014_004_EEGNet_lamsweep | 55.4 | 56.7 (0.05) | +1.2 |

## 3. Method & backbone comparisons (per-config)

### Framework zoo
- **BNCI2014_001_EEGNet_frameworks**: erm:0=42.1/lk1.18 · coral:1=42.7/lk0.77 · mmd:1=41.6/lk0.16 · irm:100=35.6/lk0.12 · vrex:10=37.1/lk0.74 · groupdro:0.01=38.4/lk1.06 · dann:1=40.7/lk0.42 · cdann:0.5=40.5/lk0.33
- **BNCI2014_004_EEGNet_frameworks**: erm:0=64.8/lk0.54 · coral:1=65.1/lk0.34 · mmd:1=64.9/lk0.07 · irm:100=64.4/lk0.06 · vrex:10=64.4/lk0.39 · groupdro:0.01=63.1/lk0.54 · dann:1=65.2/lk0.11 · cdann:0.5=66.5/lk0.11

### GNN benchmark
- **DEAP_quadrant_GraphCMI**: erm:0=26.6/lk1.47 · lpc_prior:0.3=25.3/lk0.01 · cdann:1=25.8/lk0.72

### Alignment
- **align_BNCI2014_001_LogCov_ha**: erm:0=32.7/lk1.18 · lpc_prior:0.3=31.2/lk0.02
- **align_BNCI2014_001_LogCov_none**: erm:0=36.8/lk1.13 · lpc_prior:0.3=36.9/lk0.03
- **align_BNCI2014_001_ea**: erm:0=48.8/lk0.79 · lpc_prior:0.1=47.3/lk0.23
- **align_BNCI2014_001_ea_strict**: erm:0=41.8/lk0.79 · lpc_prior:0.1=39.3/lk0.23
- **align_BNCI2014_001_none**: erm:0=43.2/lk1.06 · lpc_prior:0.1=40.8/lk0.27
- **align_BNCI2014_001_ra**: erm:0=48.8/lk0.84 · lpc_prior:0.1=46.2/lk0.23
- **align_BNCI2014_004_ea**: erm:0=63.8/lk0.43 · lpc_prior:0.1=64.1/lk0.07
- **align_BNCI2014_004_none**: erm:0=65.6/lk0.53 · lpc_prior:0.1=64.8/lk0.07
- **align_BNCI2014_004_ra**: erm:0=64.1/lk0.43 · lpc_prior:0.1=64.1/lk0.07
- **align_Lee2019_MI_ea**: erm:0=70.8/lk0.41 · lpc_prior:0.1=71.3/lk0.08
- **align_Lee2019_MI_none**: erm:0=69.5/lk0.63 · lpc_prior:0.1=69.4/lk0.06
- **align_Lee2019_MI_ra**: erm:0=70.3/lk0.44 · lpc_prior:0.1=69.6/lk0.06

### Self-supervised
- **BNCI2014_001_EEGNet_lpcssl**: erm:0=43.1/lk1.06 · simclr:1.0=40.3/lk1.62 · lpc_simclr:0.3:1.0=31.6/lk0.06 · byol:1.0=40.2/lk1.49 · lpc_byol:0.3:1.0=36.5/lk0.10 · lpc_prior:0.3=37.9/lk0.13
- **BNCI2014_001_EEGNet_ssl**: erm:0=43.0/lk1.06 · supcon:1.0=43.0/lk0.64 · simclr:0.5=41.0/lk1.51 · simclr:1.0=40.5/lk1.62 · byol:1.0=40.0/lk1.49 · lpc_prior:0.3=37.8/lk0.13
- **BNCI2014_004_EEGNet_lpcssl**: erm:0=65.4/lk0.52 · simclr:1.0=64.4/lk1.42 · lpc_simclr:0.3:1.0=63.0/lk0.01 · byol:1.0=65.2/lk1.25 · lpc_byol:0.3:1.0=65.4/lk0.04 · lpc_prior:0.3=64.4/lk0.03
- **BNCI2014_004_EEGNet_ssl**: erm:0=65.6/lk0.52 · supcon:1.0=65.9/lk0.29 · simclr:0.5=64.4/lk1.37 · simclr:1.0=64.6/lk1.42 · byol:1.0=65.4/lk1.25 · lpc_prior:0.3=64.5/lk0.03

### Route-2 FMCA
- **route2_BNCI2014_001_EEGNet**: erm:0=51.5/lk1.21 · lpc_prior:0.3=47.2/lk0.13 · fmca_chain:0.3=46.7/lk0.02 · fmca_diff:0.3=43.6/lk0.03 · fmca_strat:0.5=42.2/lk0.01
- **route2_BNCI2014_001_TSMNet**: erm:0=41.5/lk2.04 · lpc_prior:0.3=41.2/lk2.04 · fmca_chain:0.3=25.0/lk0.00 · fmca_strat:0.5=25.0/lk0.00
- **route2_BNCI2014_004_EEGNet**: erm:0=68.8/lk0.53 · lpc_prior:0.3=69.5/lk0.03 · fmca_chain:0.3=69.2/lk0.01 · fmca_diff:0.3=69.6/lk0.01 · fmca_strat:0.5=69.2/lk0.00

### Classical
- **classical_ADFTD**: TS+LR=58.0/lk0.00 · MDM=57.8/lk0.00 · CSP+LDA=43.8/lk0.00
- **classical_ADFTD_bin**: TS+LR=74.8/lk0.00 · MDM=80.9/lk0.00 · CSP+LDA=71.0/lk0.00
- **classical_BNCI2014_001**: TS+LR=36.3/lk0.00 · MDM=36.8/lk0.00 · CSP+LDA=37.8/lk0.00
- **classical_BNCI2014_004**: TS+LR=49.0/lk0.00 · MDM=48.0/lk0.00 · CSP+LDA=48.9/lk0.00
- **classical_Cho2017**: TS+LR=60.6/lk0.00 · MDM=52.7/lk0.00 · CSP+LDA=61.0/lk0.00
- **classical_DEAP**: TS+LR=50.2/lk0.00 · MDM=49.2/lk0.00 · CSP+LDA=49.7/lk0.00
- **classical_Lee2019_MI**: TS+LR=65.9/lk0.00 · MDM=54.5/lk0.00 · CSP+LDA=64.9/lk0.00
- **classical_MUMTAZ**: TS+LR=79.7/lk0.00 · MDM=65.5/lk0.00 · CSP+LDA=70.9/lk0.00
- **classical_SEED**: TS+LR=51.3/lk0.00 · MDM=45.5/lk0.00 · CSP+LDA=48.5/lk0.00
- **classical_Schirrmeister2017**: TS+LR=42.5/lk0.00 · MDM=35.4/lk0.00 · CSP+LDA=38.6/lk0.00
- **classical_TUAB**: TS+LR=47.5/lk0.00 · MDM=53.8/lk0.00 · CSP+LDA=45.0/lk0.00
