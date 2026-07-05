# Track B source-OOD benefit gate --- post-hoc target audit

Gate is SOURCE-ONLY; target used only here to score it. Actual = target ΔbAcc [subject-cluster CI].

| dataset | bb | method | gate action | src task-drop UCB | src benefit LCB | **actual target ΔbAcc [CI]** | class | correct? |
|---|---|---|---|---|---|---|---|---|
| Cho2017 | EEGNet | INLP | **REJECT** | +0.270 | -0.267 | -0.150 [-0.184,-0.117] | HARMFUL | yes |
| Cho2017 | EEGNet | LEACE | **REJECT** | +0.270 | -0.267 | -0.150 [-0.185,-0.118] | HARMFUL | yes |
| Cho2017 | EEGNet | RLACE | **REJECT** | +0.270 | -0.267 | -0.150 [-0.185,-0.118] | HARMFUL | yes |
| Cho2017 | EEGNet | TOS_VD | **ABSTAIN** | +0.001 | -0.001 | -0.001 [-0.002,-0.000] | neutral | yes |
| Cho2017 | EEGNet | random_k | **REJECT** | +0.030 | -0.033 | -0.007 [-0.010,-0.004] | neutral | yes |
| Cho2017 | TSMNet | INLP | **REJECT** | +0.435 | -0.439 | -0.105 [-0.127,-0.084] | HARMFUL | yes |
| Cho2017 | TSMNet | LEACE | **REJECT** | +0.022 | -0.008 | -0.001 [-0.003,+0.000] | neutral | yes |
| Cho2017 | TSMNet | RLACE | **REJECT** | +0.114 | -0.122 | -0.000 [-0.004,+0.004] | neutral | yes |
| Cho2017 | TSMNet | TOS_VD | **ABSTAIN** | +0.000 | -0.000 | -0.000 [-0.001,+0.001] | neutral | yes |
| Cho2017 | TSMNet | random_k | **REJECT** | +0.046 | -0.052 | -0.000 [-0.002,+0.002] | neutral | yes |
| Lee2019_MI | EEGNet | INLP | **REJECT** | +0.295 | -0.279 | -0.185 [-0.220,-0.152] | HARMFUL | yes |
| Lee2019_MI | EEGNet | LEACE | **REJECT** | +0.295 | -0.279 | -0.185 [-0.220,-0.152] | HARMFUL | yes |
| Lee2019_MI | EEGNet | RLACE | **REJECT** | +0.295 | -0.279 | -0.185 [-0.219,-0.150] | HARMFUL | yes |
| Lee2019_MI | EEGNet | TOS_VD | **ABSTAIN** | +0.000 | -0.001 | -0.001 [-0.002,+0.001] | neutral | yes |
| Lee2019_MI | EEGNet | random_k | **REJECT** | +0.039 | -0.040 | -0.007 [-0.010,-0.005] | neutral | yes |
| Lee2019_MI | TSMNet | INLP | **REJECT** | +0.456 | -0.443 | -0.154 [-0.181,-0.127] | HARMFUL | yes |
| Lee2019_MI | TSMNet | LEACE | **REJECT** | +0.025 | -0.012 | -0.002 [-0.003,+0.000] | neutral | yes |
| Lee2019_MI | TSMNet | RLACE | **REJECT** | +0.085 | -0.089 | -0.003 [-0.006,-0.000] | neutral | yes |
| Lee2019_MI | TSMNet | TOS_VD | **ABSTAIN** | +0.000 | -0.001 | -0.000 [-0.001,+0.001] | neutral | yes |
| Lee2019_MI | TSMNet | random_k | **REJECT** | +0.038 | -0.043 | +0.000 [-0.001,+0.001] | neutral | yes |

## Summary
- cells: 20  (ACCEPT 0 / REJECT 16 / ABSTAIN 4)
- **false-accept (ACCEPT a non-beneficial): 0**
- false-reject (REJECT a beneficial): 0
- missed-benefit (ABSTAIN a beneficial): 0
- **harm-prevented (REJECT a harmful): 8**
- correct decisions: 20/20
- harmful cells actually present: 8 (all should be REJECT/ABSTAIN)
- beneficial cells actually present: 0
