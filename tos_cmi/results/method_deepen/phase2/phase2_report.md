# Phase 2 dry-run --- task-preserving / conditional erasure (Lee2019 & Cho2017, EEGNet, seed0)

Config `phase2_task_preserving_fixed.yaml` sha256:`69fed2a3051a` (thresholds frozen = Track B: safety task-drop UCB<=0.02, benefit LCB>+0.01, domain-gain diagnostic-only, target audit-only). Gate action is SOURCE-ONLY; target ΔbAcc/ΔNLL below are a POST-HOC audit.

> **Caveat (disclosed):** `cc_leace_predicted_route_deployable`'s exact +0.000 target ΔbAcc is partly a STRUCTURAL TAUTOLOGY -- routing target features by a task-predictor and then re-probing the task reproduces the router's boundary (verified: identical argmax on all target points). It is therefore NOT a clean test of conditional erasure; **`tp_leace` is the clean task-preserving result.** Both give zero deployable target benefit, so the conclusion is unchanged.

## Cho2017 EEGNet (15 folds, chance bAcc 0.500)
| eraser | src task after (was) | task-drop UCB | subj dec (full->eras) | domain-gain | src-LOSO benefit LCB | gate action | **target ΔbAcc [CI]** | **target ΔNLL [CI]** | verdict |
|---|---|---|---|---|---|---|---|---|---|
| cc_leace_oracle_route_diagnostic | 1.000 (0.769) | -0.229 | 0.31->0.02 | +0.292 | n/a | **DIAGNOSTIC** | +0.335 [+0.274,+0.390] | -0.613 [-0.710,-0.514] | DIAGNOSTIC (oracle upper bound; not deployable) |
| cc_leace_predicted_route_deployable | 0.769 (0.769) | +0.000 | 0.31->0.02 | +0.292 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | +0.012 [-0.035,+0.052] | no benefit |
| leace_baseline | 0.500 (0.769) | +0.271 | 0.31->0.02 | +0.292 | -0.273 | **REJECT** | -0.165 [-0.226,-0.110] | +0.043 [-0.053,+0.142] | HARMFUL (target degraded) |
| random_k | 0.749 (0.769) | +0.025 | 0.31->0.19 | +0.123 | -0.027 | **REJECT** | +0.009 [-0.004,+0.024] | -0.036 [-0.059,-0.016] | MIXED (point>0, CI includes 0) |
| tp_leace_task_carrier_preserving | 0.769 (0.769) | +0.000 | 0.31->0.03 | +0.288 | -0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | -0.000 [-0.001,+0.000] | no benefit |

**Vs the original LEACE collapse on Cho2017:**
- plain LEACE: source task 0.769->0.500 (drop UCB +0.271), target ΔbAcc -0.165 [-0.226,-0.110]
- TP-LEACE: preserves task? YES (drop UCB +0.000) | erases subject? YES (dom-gain +0.288) | improves target? no (ΔbAcc +0.000 [+0.000,+0.000])
- cc-LEACE (predicted): preserves task? YES (drop UCB +0.000) | erases subject? YES (dom-gain +0.292) | improves target? no (ΔbAcc +0.000 [+0.000,+0.000])
- oracle cc-LEACE (perfect routing, upper bound): src task 0.769->1.000, target ΔbAcc +0.335 [+0.274,+0.390] (uses TRUE target labels -> NOT deployable)

## Lee2019_MI EEGNet (15 folds, chance bAcc 0.500)
| eraser | src task after (was) | task-drop UCB | subj dec (full->eras) | domain-gain | src-LOSO benefit LCB | gate action | **target ΔbAcc [CI]** | **target ΔNLL [CI]** | verdict |
|---|---|---|---|---|---|---|---|---|---|
| cc_leace_oracle_route_diagnostic | 1.000 (0.797) | -0.200 | 0.27->0.02 | +0.256 | n/a | **DIAGNOSTIC** | +0.319 [+0.265,+0.374] | -0.605 [-0.703,-0.505] | DIAGNOSTIC (oracle upper bound; not deployable) |
| cc_leace_predicted_route_deployable | 0.797 (0.797) | +0.000 | 0.27->0.02 | +0.254 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | +0.030 [-0.010,+0.068] | no benefit |
| leace_baseline | 0.500 (0.797) | +0.300 | 0.27->0.02 | +0.256 | -0.311 | **REJECT** | -0.181 [-0.235,-0.126] | +0.063 [-0.035,+0.164] | HARMFUL (target degraded) |
| random_k | 0.755 (0.797) | +0.051 | 0.27->0.18 | +0.089 | -0.062 | **REJECT** | -0.011 [-0.025,+0.001] | +0.001 [-0.042,+0.038] | no benefit |
| tp_leace_task_carrier_preserving | 0.797 (0.797) | +0.000 | 0.27->0.03 | +0.247 | -0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | +0.000 [-0.000,+0.000] | no benefit |

**Vs the original LEACE collapse on Lee2019_MI:**
- plain LEACE: source task 0.797->0.500 (drop UCB +0.300), target ΔbAcc -0.181 [-0.235,-0.126]
- TP-LEACE: preserves task? YES (drop UCB +0.000) | erases subject? YES (dom-gain +0.247) | improves target? no (ΔbAcc +0.000 [+0.000,+0.000])
- cc-LEACE (predicted): preserves task? YES (drop UCB +0.000) | erases subject? YES (dom-gain +0.254) | improves target? no (ΔbAcc +0.000 [+0.000,+0.000])
- oracle cc-LEACE (perfect routing, upper bound): src task 0.797->1.000, target ΔbAcc +0.319 [+0.265,+0.374] (uses TRUE target labels -> NOT deployable)

## Decision
- No STOP triggered: no deployable eraser cleared (target ΔbAcc LCB>+0.01 & safe & random doesn't reproduce).
- If any eraser PRESERVES task (drop UCB<=0.02) but target still does NOT improve -> the high-value 'task safe, transfer flat' result -> proceed to V2 with the new erasers in the set.
- If no eraser both preserves task AND erases subject -> subject<->task strongly entangled in the compact binary EEGNet latent; refusal is the correct action.
