# C29 — Representation-Head Origin of Target Class-Conditioned Confidence (frozen C19 `664007686afb520f`)

> The head is LINEAR: logit = W·z + b, so the offset-relevant representation projection W·z = (logit − b) is available READ-ONLY (no target-z re-persistence). C29 tests whether the C27 carrier originates from the parameter head-bias b or the representation projection / its target-specific shift. DIAGNOSTIC-ONLY.

- **PRIMARY: `R2_effective_logit_bias_from_target_representation`** — the offset-carrying effective class bias is an EFFECTIVE logit bias induced by the target representation projection mean(W.z), NOT the parameter head-bias b.
- established: **R2_effective_logit_bias_from_target_representation, R3_target_representation_shift_drives_offset, R6_source_representation_tracks_source_error_only**
- carrier identity-entangled (id acc +0.665) — carried through from C26/C27.

## Q1 — head-bias vs representation-projection decomposition (DECISIVE)

- full carrier gap **+0.524** (survives True); remove parameter b → **+0.510** (destroys? no); remove effective mean (C27) → **-0.313**.

| 4-vec gauge | gap closed | survives |
|---|---:|:--:|
| effective class bias (mean logit) | -0.055 | False |
| parameter head-bias b | -0.943 | False |
| representation-projection mean W·z | -0.085 | False |

- parameter-bias drives: **False**; representation-projection drives: **True**. removing the parameter head-bias b (-> W.z) PRESERVES the carrier (0.510 vs full 0.524) while removing the per-class EFFECTIVE mean DESTROYS it -> the offset-carrying effective class bias is the representation-projection mean mean(W.z), NOT the parameter head-bias b (R2). The carrier is a NONLINEAR softmax confidence, so the LINEAR b/projection-mean 4-vec gauges do not isolate it (b-gauge -0.943, projmean-gauge -0.085) -- evidence is the counterfactual on the actual carrier, not the linear gauges.

## Q4 — head/representation logit counterfactuals

| intervention | gap closed | survives | destroys |
|---|---:|:--:|:--:|
| raw | +0.524 | True | False |
| parameter_bias_removed | +0.510 | True | False |
| effective_mean_removed | -0.313 | False | True |
| projection_only | +0.510 | True | False |
| weight_norm_normalized | +0.353 | False | False |
| global_scale_removed | -0.149 | False | True |
| source_mean_centered_projection | -0.297 | False | True |

- baseline +0.524; destroyers: **effective_mean_removed, global_scale_removed, source_mean_centered_projection**.

## Q2/Q3 — representation projection geometry + source↔target residual

- logit = W.z + b, so the projection W.z = (logit - b) is the COMPLETE offset-relevant representation summary; z-components orthogonal to W are offset-irrelevant (a full 800-d z re-persistence would add only offset-orthogonal descriptive geometry).
- target projection-mean gap **-0.085**; source-explained **+0.348**; TARGET RESIDUAL **-0.116** → residual carries: **False**. neither the source-explained projection nor its residual cleanly carries the offset

## Boundary of the claim

> DIAGNOSTIC-ONLY. Head params were a CPU read of frozen parameters (no training/inference). W·z=(logit−b) captures ALL offset-relevant representation information; a full 800-d z re-persistence would add only offset-orthogonal descriptive geometry. NOT a selector, NOT deployable; the carrier remains identity-entangled (C26/C27). Target labels were not used in any factor construction.