# C6 — BNCI2014_001 LOSO seed-0

> BNCI2014-001 LOSO seed-0 full-bootstrap staged run. This is not the final multi-seed, multi-dataset confirmatory efficacy result.

- folds: **9** (targets [1, 2, 3, 4, 5, 6, 7, 8, 9])
- protocol_family: `oaci-confirmatory-v2-pilot-BNCI2014_001`
- provenance_hash: `f2f021bdbaaacf4e0cbe275b29e3ed84909b057acefbce64bb180fd4987f5156`
- all_deep_verified: **True**; all_target_fit_empty: **True**
- per-fold context hashes (distinct per fold): target-001:`4ef4b941`, target-002:`aa2713ec`, target-003:`4b4309f3`, target-004:`5fc5e66c`, target-005:`1da10acd`, target-006:`ac9119da`, target-007:`e1ff2e11`, target-008:`45476ddb`, target-009:`a897de0b`

## k1 — leakage UCL: Δ = audit_ucl(OACI) − audit_ucl(ERM)  (lower ⇒ OACI leaks less)
### level 0
- mean 0.0123 · median 0.0073 · min -0.0183 · max 0.0861 · n 9 · folds Δ<0: **3/9**

| target | Δ audit_ucl |
|---:|---:|
| 1 | 0.0861 |
| 2 | -0.0040 |
| 3 | 0.0065 |
| 4 | 0.0073 |
| 5 | 0.0102 |
| 6 | 0.0076 |
| 7 | -0.0011 |
| 8 | -0.0183 |
| 9 | 0.0161 |

### level 1
- mean 0.0047 · median 0.0024 · min -0.0422 · max 0.0834 · n 9 · folds Δ<0: **4/9**

| target | Δ audit_ucl |
|---:|---:|
| 1 | 0.0834 |
| 2 | 0.0024 |
| 3 | 0.0039 |
| 4 | -0.0011 |
| 5 | -0.0422 |
| 6 | 0.0046 |
| 7 | -0.0018 |
| 8 | -0.0127 |
| 9 | 0.0060 |

## k2 — target metrics: Δ = OACI − ERM  (bAcc ↑ · NLL ↓ · ECE ↓ better)
### level 0
- ΔbAcc (↑): mean -0.0085 · median -0.0208 · min -0.0833 · max 0.0608 · improved **4/9**
- ΔNLL (↓): mean -0.0681 · median -0.0342 · min -0.5490 · max 0.1705 · improved **6/9**
- ΔECE (↓): mean -0.0230 · median -0.0134 · min -0.1655 · max 0.0738 · improved **8/9**

### level 1
- ΔbAcc (↑): mean -0.0164 · median -0.0156 · min -0.0885 · max 0.0747 · improved **2/9**
- ΔNLL (↓): mean 0.0081 · median -0.0637 · min -0.2207 · max 0.3856 · improved **5/9**
- ΔECE (↓): mean -0.0127 · median 0.0028 · min -0.1304 · max 0.0762 · improved **4/9**
