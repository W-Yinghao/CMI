# V2 --- source-only acceptance CEILING / non-identifiability (smoke)

Config sha256:`169a6b2507f2`; thresholds FROZEN (safety UCB<=0.020, benefit LCB>0.010, domain diagnostic-only, target audit-only). **No world expects ACCEPT.** World-gen: variantA=? f_align/phi=0.15 beta=1.00 m=4. **Semi-synthetic (real latents + injected nuisance); a limit result, not a main-paper claim.**

## World A --- target_beneficial_but_source_uncertifiable (expect REJECT/ABSTAIN (not ACCEPT))
| intervention | n_src | alpha | task-drop UCB | src-LOSO benefit LCB | domain-gain | gate | target ΔbAcc [CI] | safe | tgt-benef |
|---|---|---|---|---|---|---|---|---|---|
| alpha_leace | all | 0.25 | +0.002 | -0.001 | -0.000 | **ABSTAIN** | +0.002 [+0.000,+0.006] | Y | n |
| alpha_leace | all | 0.25 | +0.000 | -0.001 | +0.000 | **ABSTAIN** | +0.001 [+0.000,+0.003] | Y | n |
| alpha_leace | all | 0.50 | +0.000 | -0.001 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 0.50 | +0.001 | -0.001 | +0.000 | **ABSTAIN** | +0.000 [-0.003,+0.003] | Y | n |
| alpha_leace | all | 1.00 | +0.000 | -0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 2.00 | +0.001 | -0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| fair_conditional_leace_disjoint_router | all | 0.25 | +0.005 | -0.014 | +0.358 | **ABSTAIN** | +0.027 [+0.016,+0.042] | Y | Y |
| fair_conditional_leace_disjoint_router | all | 0.25 | +0.002 | -0.014 | +0.373 | **ABSTAIN** | +0.025 [+0.011,+0.038] | Y | Y |
| fair_conditional_leace_disjoint_router | all | 0.50 | +0.002 | -0.015 | +0.364 | **ABSTAIN** | +0.018 [+0.010,+0.025] | Y | Y |
| fair_conditional_leace_disjoint_router | all | 0.50 | +0.002 | -0.011 | +0.372 | **ABSTAIN** | +0.019 [+0.001,+0.040] | Y | n |
| fair_conditional_leace_disjoint_router | all | 1.00 | +0.001 | -0.011 | +0.365 | **ABSTAIN** | +0.012 [+0.007,+0.017] | Y | n |
| fair_conditional_leace_disjoint_router | all | 1.00 | +0.002 | -0.011 | +0.369 | **ABSTAIN** | +0.014 [+0.004,+0.027] | Y | n |
| fair_conditional_leace_disjoint_router | all | 2.00 | +0.001 | -0.010 | +0.367 | **ABSTAIN** | +0.009 [+0.006,+0.012] | Y | n |
| fair_conditional_leace_disjoint_router | all | 2.00 | +0.001 | -0.012 | +0.369 | **ABSTAIN** | +0.011 [+0.001,+0.028] | Y | n |
| identity | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| inlp | all | 0.25 | +0.165 | -0.175 | +0.472 | **REJECT** | -0.064 [-0.167,+0.024] | n | n |
| inlp | all | 0.25 | +0.170 | -0.184 | +0.477 | **REJECT** | -0.009 [-0.039,+0.023] | n | n |
| inlp | all | 0.50 | +0.159 | -0.175 | +0.478 | **REJECT** | -0.072 [-0.171,+0.017] | n | n |
| inlp | all | 0.50 | +0.185 | -0.190 | +0.486 | **REJECT** | -0.002 [-0.034,+0.035] | n | n |
| inlp | all | 1.00 | +0.172 | -0.171 | +0.483 | **REJECT** | -0.041 [-0.094,+0.015] | n | n |
| inlp | all | 1.00 | +0.173 | -0.188 | +0.478 | **REJECT** | +0.016 [-0.015,+0.047] | n | n |
| inlp | all | 2.00 | +0.145 | -0.156 | +0.479 | **REJECT** | -0.026 [-0.106,+0.035] | n | n |
| inlp | all | 2.00 | +0.149 | -0.158 | +0.481 | **REJECT** | +0.023 [-0.030,+0.063] | n | n |
| leace_baseline | all | 0.25 | +0.022 | -0.044 | +0.491 | **REJECT** | +0.079 [+0.050,+0.109] | n | Y |
| leace_baseline | all | 0.25 | +0.016 | -0.032 | +0.490 | **ABSTAIN** | +0.081 [+0.049,+0.111] | Y | Y |
| leace_baseline | all | 0.50 | +0.021 | -0.048 | +0.498 | **REJECT** | +0.076 [+0.042,+0.111] | n | Y |
| leace_baseline | all | 0.50 | +0.017 | -0.035 | +0.495 | **ABSTAIN** | +0.086 [+0.051,+0.121] | Y | Y |
| leace_baseline | all | 1.00 | +0.021 | -0.048 | +0.498 | **REJECT** | +0.075 [+0.041,+0.113] | n | Y |
| leace_baseline | all | 1.00 | +0.018 | -0.035 | +0.495 | **ABSTAIN** | +0.086 [+0.050,+0.122] | Y | Y |
| leace_baseline | all | 2.00 | +0.020 | -0.048 | +0.498 | **REJECT** | +0.075 [+0.040,+0.116] | n | Y |
| leace_baseline | all | 2.00 | +0.019 | -0.037 | +0.495 | **ABSTAIN** | +0.085 [+0.050,+0.120] | Y | Y |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.25 | +0.007 | -0.019 | +0.459 | **DIAGNOSTIC** | +0.040 [+0.026,+0.062] | Y | Y |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.25 | +0.004 | -0.011 | +0.464 | **DIAGNOSTIC** | +0.041 [+0.016,+0.069] | Y | Y |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.50 | +0.007 | -0.024 | +0.466 | **DIAGNOSTIC** | +0.039 [+0.023,+0.062] | Y | Y |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.50 | +0.005 | -0.012 | +0.470 | **DIAGNOSTIC** | +0.047 [+0.018,+0.081] | Y | Y |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 1.00 | +0.006 | -0.025 | +0.466 | **DIAGNOSTIC** | +0.038 [+0.022,+0.061] | Y | Y |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 1.00 | +0.005 | -0.013 | +0.470 | **DIAGNOSTIC** | +0.049 [+0.020,+0.082] | Y | Y |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 2.00 | +0.006 | -0.025 | +0.466 | **DIAGNOSTIC** | +0.038 [+0.022,+0.061] | Y | Y |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 2.00 | +0.005 | -0.014 | +0.470 | **DIAGNOSTIC** | +0.049 [+0.020,+0.082] | Y | Y |
| random_k | all | 0.25 | +0.001 | -0.000 | +0.002 | **ABSTAIN** | +0.000 [-0.003,+0.003] | Y | n |
| random_k | all | 0.25 | +0.001 | -0.004 | +0.002 | **ABSTAIN** | -0.001 [-0.006,+0.005] | Y | n |
| random_k | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 0.50 | +0.001 | -0.001 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| rlace | all | 0.25 | +0.013 | -0.031 | +0.427 | **ABSTAIN** | +0.053 [+0.035,+0.080] | Y | Y |
| rlace | all | 0.25 | +0.011 | -0.028 | +0.428 | **ABSTAIN** | +0.075 [+0.041,+0.110] | Y | Y |
| rlace | all | 0.50 | +0.015 | -0.036 | +0.396 | **ABSTAIN** | +0.060 [+0.038,+0.088] | Y | Y |
| rlace | all | 0.50 | +0.018 | -0.034 | +0.420 | **ABSTAIN** | +0.089 [+0.049,+0.135] | Y | Y |
| rlace | all | 1.00 | +0.011 | -0.031 | +0.245 | **ABSTAIN** | +0.046 [+0.027,+0.071] | Y | Y |
| rlace | all | 1.00 | +0.021 | -0.040 | +0.264 | **REJECT** | +0.090 [+0.047,+0.132] | n | Y |
| rlace | all | 2.00 | +0.003 | -0.010 | +0.047 | **ABSTAIN** | +0.012 [+0.001,+0.024] | Y | n |
| rlace | all | 2.00 | +0.011 | -0.020 | +0.057 | **ABSTAIN** | +0.046 [+0.025,+0.067] | Y | Y |
| tos_vd | all | 0.25 | +0.002 | -0.003 | +0.396 | **ABSTAIN** | +0.005 [+0.001,+0.009] | Y | n |
| tos_vd | all | 0.25 | +0.002 | -0.002 | +0.388 | **ABSTAIN** | +0.000 [-0.003,+0.003] | Y | n |
| tos_vd | all | 0.50 | -0.000 | -0.001 | +0.232 | **ABSTAIN** | +0.001 [+0.000,+0.003] | Y | n |
| tos_vd | all | 0.50 | +0.000 | -0.001 | +0.235 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.25 | +0.000 | +0.000 | +0.035 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.25 | +0.001 | +0.000 | +0.101 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.50 | +0.000 | -0.000 | +0.088 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.50 | +0.000 | -0.000 | +0.200 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 1.00 | +0.000 | -0.001 | +0.104 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 1.00 | +0.000 | -0.000 | +0.226 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 2.00 | +0.000 | -0.000 | +0.109 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 2.00 | +0.000 | -0.001 | +0.226 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |

**Ceiling:** 21 SAFE cell(s) with a real target gain (target dbAcc LCB>+0.01) that the gate does NOT accept (principled ACCEPTs=0). oracle target dbAcc=+0.049 vs random_k=+0.000 (random reproduces oracle? no). -> PASS.

## World B --- task_entangled_unsafe (expect REJECT)
| intervention | n_src | alpha | task-drop UCB | src-LOSO benefit LCB | domain-gain | gate | target ΔbAcc [CI] | safe | tgt-benef |
|---|---|---|---|---|---|---|---|---|---|
| alpha_leace | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| fair_conditional_leace_disjoint_router | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| fair_conditional_leace_disjoint_router | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| fair_conditional_leace_disjoint_router | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| fair_conditional_leace_disjoint_router | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| fair_conditional_leace_disjoint_router | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| fair_conditional_leace_disjoint_router | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| fair_conditional_leace_disjoint_router | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| fair_conditional_leace_disjoint_router | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| inlp | all | 0.25 | +0.491 | -0.509 | +0.485 | **REJECT** | -0.446 [-0.517,-0.355] | n | n |
| inlp | all | 0.25 | +0.492 | -0.495 | +0.484 | **REJECT** | -0.456 [-0.508,-0.404] | n | n |
| inlp | all | 0.50 | +0.494 | -0.508 | +0.489 | **REJECT** | -0.447 [-0.515,-0.358] | n | n |
| inlp | all | 0.50 | +0.486 | -0.480 | +0.481 | **REJECT** | -0.461 [-0.494,-0.413] | n | n |
| inlp | all | 1.00 | +0.494 | -0.508 | +0.491 | **REJECT** | -0.492 [-0.525,-0.456] | n | n |
| inlp | all | 1.00 | +0.489 | -0.507 | +0.481 | **REJECT** | -0.454 [-0.484,-0.424] | n | n |
| inlp | all | 2.00 | +0.498 | -0.502 | +0.492 | **REJECT** | -0.476 [-0.532,-0.426] | n | n |
| inlp | all | 2.00 | +0.480 | -0.503 | +0.476 | **REJECT** | -0.445 [-0.500,-0.377] | n | n |
| leace_baseline | all | 0.25 | +0.500 | -0.500 | +0.504 | **REJECT** | -0.500 [-0.500,-0.500] | n | n |
| leace_baseline | all | 0.25 | +0.500 | -0.500 | +0.504 | **REJECT** | -0.500 [-0.500,-0.500] | n | n |
| leace_baseline | all | 0.50 | +0.500 | -0.500 | +0.504 | **REJECT** | -0.500 [-0.500,-0.500] | n | n |
| leace_baseline | all | 0.50 | +0.500 | -0.500 | +0.504 | **REJECT** | -0.500 [-0.500,-0.500] | n | n |
| leace_baseline | all | 1.00 | +0.500 | -0.500 | +0.504 | **REJECT** | -0.500 [-0.500,-0.500] | n | n |
| leace_baseline | all | 1.00 | +0.500 | -0.500 | +0.504 | **REJECT** | -0.500 [-0.500,-0.500] | n | n |
| leace_baseline | all | 2.00 | +0.500 | -0.500 | +0.504 | **REJECT** | -0.500 [-0.500,-0.500] | n | n |
| leace_baseline | all | 2.00 | +0.500 | -0.500 | +0.504 | **REJECT** | -0.500 [-0.500,-0.500] | n | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.25 | +0.235 | -0.256 | +0.230 | **DIAGNOSTIC** | -0.313 [-0.422,-0.192] | n | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.25 | +0.207 | -0.219 | +0.204 | **DIAGNOSTIC** | -0.248 [-0.347,-0.151] | n | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.50 | +0.235 | -0.256 | +0.230 | **DIAGNOSTIC** | -0.313 [-0.422,-0.192] | n | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.50 | +0.207 | -0.219 | +0.204 | **DIAGNOSTIC** | -0.248 [-0.347,-0.151] | n | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 1.00 | +0.235 | -0.256 | +0.230 | **DIAGNOSTIC** | -0.313 [-0.422,-0.192] | n | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 1.00 | +0.207 | -0.219 | +0.204 | **DIAGNOSTIC** | -0.248 [-0.347,-0.151] | n | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 2.00 | +0.235 | -0.256 | +0.230 | **DIAGNOSTIC** | -0.313 [-0.422,-0.192] | n | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 2.00 | +0.207 | -0.219 | +0.204 | **DIAGNOSTIC** | -0.248 [-0.347,-0.151] | n | n |
| random_k | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| rlace | all | 0.25 | +0.237 | -0.231 | +0.190 | **REJECT** | -0.185 [-0.224,-0.146] | n | n |
| rlace | all | 0.25 | +0.165 | -0.156 | +0.135 | **REJECT** | -0.117 [-0.150,-0.094] | n | n |
| rlace | all | 0.50 | +0.215 | -0.205 | +0.160 | **REJECT** | -0.164 [-0.212,-0.120] | n | n |
| rlace | all | 0.50 | +0.126 | -0.117 | +0.098 | **REJECT** | -0.077 [-0.103,-0.059] | n | n |
| rlace | all | 1.00 | +0.179 | -0.159 | +0.114 | **REJECT** | -0.096 [-0.146,-0.057] | n | n |
| rlace | all | 1.00 | +0.084 | -0.077 | +0.069 | **REJECT** | -0.055 [-0.070,-0.040] | n | n |
| rlace | all | 2.00 | +0.027 | -0.026 | +0.016 | **REJECT** | -0.013 [-0.023,-0.003] | n | n |
| rlace | all | 2.00 | +0.009 | -0.006 | +0.007 | **ABSTAIN** | -0.012 [-0.017,-0.008] | Y | n |
| tos_vd | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |

**Unsafe-accept:** 0/56 principled cells ACCEPTED (want 0) -> PASS.

## World C --- removable_but_useless_identity (expect REJECT/ABSTAIN)
| intervention | n_src | alpha | task-drop UCB | src-LOSO benefit LCB | domain-gain | gate | target ΔbAcc [CI] | safe | tgt-benef |
|---|---|---|---|---|---|---|---|---|---|
| alpha_leace | all | 0.25 | +0.000 | -0.000 | -0.000 | **ABSTAIN** | +0.001 [+0.000,+0.003] | Y | n |
| alpha_leace | all | 0.25 | +0.000 | -0.001 | -0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 0.50 | +0.001 | -0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 0.50 | +0.001 | -0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 1.00 | +0.000 | -0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| alpha_leace | all | 2.00 | +0.000 | -0.000 | +0.000 | **ABSTAIN** | -0.001 [-0.003,+0.000] | Y | n |
| alpha_leace | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| fair_conditional_leace_disjoint_router | all | 0.25 | +0.004 | -0.001 | +0.467 | **ABSTAIN** | +0.000 [-0.006,+0.006] | Y | n |
| fair_conditional_leace_disjoint_router | all | 0.25 | +0.001 | +0.000 | +0.463 | **ABSTAIN** | +0.002 [-0.002,+0.006] | Y | n |
| fair_conditional_leace_disjoint_router | all | 0.50 | +0.004 | -0.000 | +0.473 | **ABSTAIN** | +0.001 [-0.003,+0.006] | Y | n |
| fair_conditional_leace_disjoint_router | all | 0.50 | +0.001 | +0.001 | +0.469 | **ABSTAIN** | +0.003 [+0.000,+0.009] | Y | n |
| fair_conditional_leace_disjoint_router | all | 1.00 | +0.004 | -0.000 | +0.473 | **ABSTAIN** | +0.001 [-0.003,+0.006] | Y | n |
| fair_conditional_leace_disjoint_router | all | 1.00 | +0.001 | -0.000 | +0.469 | **ABSTAIN** | +0.003 [+0.000,+0.009] | Y | n |
| fair_conditional_leace_disjoint_router | all | 2.00 | +0.004 | -0.000 | +0.473 | **ABSTAIN** | +0.000 [-0.003,+0.003] | Y | n |
| fair_conditional_leace_disjoint_router | all | 2.00 | +0.001 | +0.001 | +0.469 | **ABSTAIN** | +0.002 [+0.000,+0.006] | Y | n |
| identity | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 0.25 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| identity | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| inlp | all | 0.25 | +0.000 | -0.001 | +0.496 | **ABSTAIN** | -0.002 [-0.005,+0.002] | Y | n |
| inlp | all | 0.25 | +0.001 | +0.000 | +0.497 | **ABSTAIN** | -0.000 [-0.006,+0.006] | Y | n |
| inlp | all | 0.50 | +0.000 | -0.001 | +0.501 | **ABSTAIN** | -0.002 [-0.005,+0.002] | Y | n |
| inlp | all | 0.50 | +0.000 | -0.001 | +0.503 | **ABSTAIN** | +0.001 [-0.005,+0.007] | Y | n |
| inlp | all | 1.00 | +0.001 | -0.001 | +0.501 | **ABSTAIN** | -0.003 [-0.007,+0.002] | Y | n |
| inlp | all | 1.00 | +0.000 | +0.000 | +0.506 | **ABSTAIN** | +0.002 [-0.002,+0.007] | Y | n |
| inlp | all | 2.00 | +0.001 | -0.001 | +0.504 | **ABSTAIN** | -0.004 [-0.007,-0.001] | Y | n |
| inlp | all | 2.00 | +0.000 | +0.001 | +0.510 | **ABSTAIN** | +0.001 [-0.005,+0.007] | Y | n |
| leace_baseline | all | 0.25 | +0.000 | +0.000 | +0.485 | **ABSTAIN** | +0.001 [-0.005,+0.006] | Y | n |
| leace_baseline | all | 0.25 | +0.001 | -0.000 | +0.485 | **ABSTAIN** | -0.001 [-0.007,+0.005] | Y | n |
| leace_baseline | all | 0.50 | +0.000 | +0.000 | +0.490 | **ABSTAIN** | +0.001 [-0.005,+0.006] | Y | n |
| leace_baseline | all | 0.50 | +0.001 | -0.001 | +0.492 | **ABSTAIN** | -0.000 [-0.004,+0.005] | Y | n |
| leace_baseline | all | 1.00 | +0.001 | +0.000 | +0.490 | **ABSTAIN** | +0.002 [-0.002,+0.006] | Y | n |
| leace_baseline | all | 1.00 | +0.001 | -0.002 | +0.492 | **ABSTAIN** | -0.000 [-0.004,+0.005] | Y | n |
| leace_baseline | all | 2.00 | +0.000 | +0.000 | +0.490 | **ABSTAIN** | +0.001 [-0.002,+0.004] | Y | n |
| leace_baseline | all | 2.00 | +0.000 | -0.001 | +0.492 | **ABSTAIN** | -0.000 [-0.004,+0.005] | Y | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.25 | +0.000 | -0.000 | +0.497 | **DIAGNOSTIC** | -0.003 [-0.007,+0.000] | Y | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.25 | +0.001 | -0.002 | +0.496 | **DIAGNOSTIC** | -0.000 [-0.008,+0.008] | Y | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.50 | +0.000 | -0.000 | +0.502 | **DIAGNOSTIC** | -0.003 [-0.007,+0.000] | Y | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 0.50 | +0.001 | -0.002 | +0.502 | **DIAGNOSTIC** | +0.001 [-0.008,+0.010] | Y | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 1.00 | +0.001 | -0.001 | +0.502 | **DIAGNOSTIC** | -0.003 [-0.007,+0.000] | Y | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 1.00 | +0.000 | -0.003 | +0.502 | **DIAGNOSTIC** | +0.001 [-0.008,+0.010] | Y | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 2.00 | +0.001 | -0.001 | +0.502 | **DIAGNOSTIC** | -0.004 [-0.007,-0.001] | Y | n |
| oracle_nuisance_eraser_DIAGNOSTIC_ONLY | all | 2.00 | +0.000 | -0.002 | +0.502 | **DIAGNOSTIC** | +0.001 [-0.008,+0.010] | Y | n |
| random_k | all | 0.25 | +0.001 | -0.002 | +0.000 | **ABSTAIN** | -0.001 [-0.006,+0.003] | Y | n |
| random_k | all | 0.25 | -0.001 | -0.002 | +0.000 | **ABSTAIN** | -0.000 [-0.008,+0.006] | Y | n |
| random_k | all | 0.50 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 0.50 | +0.000 | -0.001 | +0.000 | **ABSTAIN** | +0.001 [+0.000,+0.003] | Y | n |
| random_k | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| random_k | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| rlace | all | 0.25 | +0.001 | -0.001 | +0.434 | **ABSTAIN** | -0.002 [-0.006,+0.002] | Y | n |
| rlace | all | 0.25 | +0.003 | -0.007 | +0.454 | **ABSTAIN** | -0.004 [-0.011,+0.004] | Y | n |
| rlace | all | 0.50 | +0.001 | +0.000 | +0.401 | **ABSTAIN** | -0.001 [-0.003,+0.000] | Y | n |
| rlace | all | 0.50 | +0.004 | -0.008 | +0.430 | **ABSTAIN** | -0.009 [-0.020,+0.002] | Y | n |
| rlace | all | 1.00 | +0.001 | -0.001 | +0.254 | **ABSTAIN** | -0.001 [-0.006,+0.003] | Y | n |
| rlace | all | 1.00 | +0.005 | -0.011 | +0.284 | **ABSTAIN** | -0.007 [-0.016,+0.001] | Y | n |
| rlace | all | 2.00 | +0.001 | -0.001 | +0.046 | **ABSTAIN** | +0.000 [-0.003,+0.003] | Y | n |
| rlace | all | 2.00 | +0.001 | -0.004 | +0.063 | **ABSTAIN** | +0.005 [+0.001,+0.009] | Y | n |
| tos_vd | all | 0.25 | +0.000 | -0.001 | +0.358 | **ABSTAIN** | +0.001 [+0.000,+0.003] | Y | n |
| tos_vd | all | 0.25 | +0.000 | -0.000 | +0.468 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 0.50 | +0.000 | +0.000 | +0.082 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 0.50 | +0.000 | +0.000 | +0.177 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 1.00 | +0.000 | +0.000 | +0.082 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 1.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tos_vd | all | 2.00 | +0.000 | +0.000 | +0.000 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.25 | +0.000 | -0.000 | +0.482 | **ABSTAIN** | +0.001 [+0.000,+0.003] | Y | n |
| tp_leace | all | 0.25 | +0.000 | -0.000 | +0.450 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.50 | +0.000 | +0.000 | +0.492 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 0.50 | +0.000 | +0.000 | +0.453 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 1.00 | +0.000 | -0.000 | +0.492 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 1.00 | +0.000 | -0.000 | +0.454 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |
| tp_leace | all | 2.00 | +0.000 | -0.000 | +0.491 | **ABSTAIN** | -0.001 [-0.003,+0.000] | Y | n |
| tp_leace | all | 2.00 | +0.000 | +0.000 | +0.453 | **ABSTAIN** | +0.000 [+0.000,+0.000] | Y | n |

**Useless-accept:** 0/56 principled cells ACCEPTED (want 0); high-domain-gain-but-useless cell present=True -> domain-gain != benefit -> PASS.

## Naive controllers (all deployable cells; a GOOD accept = actually target-beneficial)
| controller | accepts | false-accepts (non-beneficial) | true-accepts (beneficial) |
|---|---|---|---|
| domain-gain-only (accept if subj/z removed) | 108 | 90 | 18 |
| safety-only (accept if source task safe) | 180 | 167 | 13 |
| always-erase-if-any-domain-gain | 118 | 100 | 18 |
| OUR GATE (benefit+safety, source-only) | 0 | 0 | 0 |
| ORACLE target-informed selector [DIAGNOSTIC, uses target labels] | 18 | 0 | 18 |

Scatter (source-LOSO benefit LCB vs actual target ΔbAcc LCB, colored by gate action, o=safe x=unsafe): `tos_cmi/results/method_deepen/v2/v2_smoke_scatter.png`

## Ceiling smoke verdict
- World A (target-beneficial but source-uncertifiable, NO accept): PASS
- World B (no unsafe accept): PASS
- World C (no useless accept; domain-gain != benefit): PASS
- **overall: PASS**

**Reading:** naive source-only controllers (domain-gain / safety) FALSE-ACCEPT; OUR gate accepts ~nothing (conservative -- correct under the ceiling); only the ORACLE target-informed selector (diagnostic, uses target labels) picks the beneficial cells -> crossing the ceiling needs target info.
