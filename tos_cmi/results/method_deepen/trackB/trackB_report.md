# Track B source-OOD benefit gate --- post-hoc audit (hardened)

Gate is SOURCE-ONLY; target used only here. Gate config `tos_cmi/eeg/configs/trackB_gate_fixed.yaml` sha256:`ac4ba42b961a` (safety task-drop UCB<=0.02; benefit LCB>+0.01; domain-gain=diagnostic-only; target=audit-only).

## Fold coverage (this is a **pilot / sampled** run, not full LOSO)
| dataset | backbone | analyzed outer folds/seed | expected (full LOSO) | coverage |
|---|---|---|---|---|
| Cho2017 | EEGNet | 15 | 52 | SAMPLED (first 15 subjects) |
| Cho2017 | TSMNet | 15 | 52 | SAMPLED (first 15 subjects) |
| Lee2019_MI | EEGNet | 15 | 54 | SAMPLED (first 15 subjects) |
| Lee2019_MI | TSMNet | 15 | 54 | SAMPLED (first 15 subjects) |

## Gate decisions vs actual target
| dataset | bb | method | gate action | src task-drop UCB | src benefit LCB | domain-gain | **actual target ΔbAcc [CI]** | class | correct? |
|---|---|---|---|---|---|---|---|---|---|
| Cho2017 | EEGNet | INLP | **REJECT** | +0.270 | -0.267 | +0.297 | -0.150 [-0.184,-0.117] | HARMFUL | yes |
| Cho2017 | EEGNet | LEACE | **REJECT** | +0.270 | -0.267 | +0.297 | -0.150 [-0.185,-0.118] | HARMFUL | yes |
| Cho2017 | EEGNet | RLACE | **REJECT** | +0.270 | -0.267 | +0.297 | -0.150 [-0.185,-0.118] | HARMFUL | yes |
| Cho2017 | EEGNet | TOS_VD | **ABSTAIN** | +0.001 | -0.001 | +0.239 | -0.001 [-0.002,-0.000] | neutral | yes |
| Cho2017 | EEGNet | random_k | **REJECT** | +0.030 | -0.033 | +0.124 | -0.007 [-0.010,-0.004] | neutral | yes |
| Cho2017 | TSMNet | INLP | **REJECT** | +0.435 | -0.439 | +0.967 | -0.105 [-0.127,-0.084] | HARMFUL | yes |
| Cho2017 | TSMNet | LEACE | **REJECT** | +0.022 | -0.008 | +0.976 | -0.001 [-0.003,+0.000] | neutral | yes |
| Cho2017 | TSMNet | RLACE | **REJECT** | +0.114 | -0.122 | +0.404 | -0.000 [-0.004,+0.004] | neutral | yes |
| Cho2017 | TSMNet | TOS_VD | **ABSTAIN** | +0.000 | -0.000 | +0.012 | -0.000 [-0.001,+0.001] | neutral | yes |
| Cho2017 | TSMNet | random_k | **REJECT** | +0.046 | -0.052 | +0.000 | -0.000 [-0.002,+0.002] | neutral | yes |
| Lee2019_MI | EEGNet | INLP | **REJECT** | +0.295 | -0.279 | +0.246 | -0.185 [-0.220,-0.152] | HARMFUL | yes |
| Lee2019_MI | EEGNet | LEACE | **REJECT** | +0.295 | -0.279 | +0.246 | -0.185 [-0.220,-0.152] | HARMFUL | yes |
| Lee2019_MI | EEGNet | RLACE | **REJECT** | +0.295 | -0.279 | +0.246 | -0.185 [-0.219,-0.150] | HARMFUL | yes |
| Lee2019_MI | EEGNet | TOS_VD | **ABSTAIN** | +0.000 | -0.001 | +0.199 | -0.001 [-0.002,+0.001] | neutral | yes |
| Lee2019_MI | EEGNet | random_k | **REJECT** | +0.039 | -0.040 | +0.089 | -0.007 [-0.010,-0.005] | neutral | yes |
| Lee2019_MI | TSMNet | INLP | **REJECT** | +0.456 | -0.443 | +0.974 | -0.154 [-0.181,-0.127] | HARMFUL | yes |
| Lee2019_MI | TSMNet | LEACE | **REJECT** | +0.025 | -0.012 | +0.976 | -0.002 [-0.003,+0.000] | neutral | yes |
| Lee2019_MI | TSMNet | RLACE | **REJECT** | +0.085 | -0.089 | +0.481 | -0.003 [-0.006,-0.000] | neutral | yes |
| Lee2019_MI | TSMNet | TOS_VD | **ABSTAIN** | +0.000 | -0.001 | +0.023 | -0.000 [-0.001,+0.001] | neutral | yes |
| Lee2019_MI | TSMNet | random_k | **REJECT** | +0.038 | -0.043 | +0.001 | +0.000 [-0.001,+0.001] | neutral | yes |

## Naive-controller baselines (same source signals; shows what leakage/safety-only would accept)
| controller | accepts | false-accepts (non-beneficial) | harm-accepts (harmful) |
|---|---|---|---|
| domain-gain-only (accept if subj removed) | 16 | 16 | 8 |
| safety-only (accept if source task safe) | 4 | 4 | 0 |
| always-erase-if-any-domain-gain | 20 | 20 | 8 |
| OUR GATE (benefit+safety, domain=diagnostic) | 0 | 0 | 0 |

## Summary
- cells: 20  (ACCEPT 0 / REJECT 16 / ABSTAIN 4)
- **false-accepts (ACCEPT a non-beneficial): 0**
- observed-positive interventions on real EEG: 0
- missed observed positives: 0 by vacuity (no observed positive to miss)
- **acceptance power: UNTESTED on real EEG** (no real positive exists; tested on V2 semi-synthetic, Phase 3)
- **harm-prevented (REJECT a harmful): 8** of 8 harmful cells
- correct decisions: 20/20
- target-use: audit-only (gate never saw target) --- PASS
