# C27 — Confidence-Occupancy Logit Geometry Counterfactual Audit (frozen C19 `664007686afb520f`)

> C26: predicted-class mix is a stable decision-occupancy pattern that IS the target fingerprint; the score-offset recovery is a confidence-mix SYNERGY interaction (P5). C27 dissects that interaction in LOGIT space (read-only over the C26 per-sample logits; no re-inference/tuning/feature-selection). DIAGNOSTIC-ONLY.

- **PRIMARY: `L1_class_conditioned_confidence_carries_interaction`** — the interaction is CLASS-CONDITIONED CONFIDENCE (how confidently the model occupies each predicted class): it is a SINGLE sufficient factor (recovers alone, survives permutation), which REVISES C26's 'irreducible synergy' -- the synergy was an artifact of class-agnostic global confidence being too coarse; global confidence SCALE is NOT necessary (temperature/logit-norm survive) while the class-specific structure + per-target occupancy-confidence coupling ARE. Still IDENTITY-ENTANGLED (the per-class confidence profile is also the target fingerprint) and only PARTIALLY error-geometry-coupled.
- established: **L1_class_conditioned_confidence_carries_interaction, L6_error_geometry_coupled, L5_identity_fingerprint_only**

## Baseline — full-R3 recovery reproduced from raw logits (consistency gate)

- full-R3 (occupancy + global confidence) gap closed **+0.491** (survives permutation: True); target-id acc **+0.670** (entangled: True) — should reproduce the C24/C26 +0.491.

## C27-A — class-conditioned confidence decomposition

- **KEY**: class-conditioned confidence ALONE gap **+0.524** (survives True, target-id +0.665) → a SINGLE sufficient factor. This REVISES C26's 'irreducible synergy': the synergy was an artifact of class-agnostic GLOBAL confidence being too coarse. Still IDENTITY-ENTANGLED (per-class confidence profile is also the fingerprint).
- occupancy + class-conditioned confidence gap **+0.660** (survives True); occ×conf interaction-only **+0.081** → class-conditioned confidence explains: **True**. class-conditioned confidence (occupancy + conf_k) reproduces >=50%% of the full-R3 recovery and survives permutation -> the interaction is class-conditioned confidence (how confidently the model occupies each class)

## C27-B — logit counterfactuals (which transform DESTROYS recovery?)

| intervention | gap closed | survives | destroys recovery |
|---|---:|:--:|:--:|
| raw | +0.491 | True | False |
| temperature | +0.512 | True | False |
| class_bias_center | -0.585 | False | True |
| logit_norm_normalize | +0.384 | False | False |
| class_uniformize | +0.073 | False | True |
| confidence_shuffle | -0.396 | False | True |
| class_shuffle | -0.935 | False | True |

- baseline gap +0.491; destroyers: **class_bias_center, class_uniformize, confidence_shuffle, class_shuffle**. interventions that destroy the offset recovery: class_bias_center, class_uniformize, confidence_shuffle, class_shuffle -> the recovery depends on the factor(s) they remove

## C27-C — sufficiency / necessity (offset recovery vs identity fingerprint, jointly)

| combo | gap closed | survives | target-id acc |
|---|---:|:--:|---:|
| occupancy | +0.003 | False | +0.698 |
| global_confidence | -0.561 | False | +0.426 |
| class_conditioned_confidence | +0.524 | True | +0.665 |
| class_bias | -0.055 | False | +0.794 |
| occ_x_conf_interaction | +0.081 | False | +0.697 |
| occupancy+global_confidence | +0.491 | True | +0.670 |
| occupancy+class_conditioned_confidence | +0.660 | True | +0.789 |
| occupancy+class_conditioned_margin | +0.432 | False | +0.779 |

## C27-D — label alignment under interventions (QUARANTINED labels, post-hoc)

- raw predmix↔per-class-recall corr **+0.881**; alignment destroyers: **class_bias_center**; offset & error-geometry coupled: **True** (class_bias_center). class_bias_center destroy(s) BOTH offset recovery and error-geometry alignment (coupled); BUT class_uniformize destroy(s) offset recovery while PRESERVING alignment -> coupling is PARTIAL, not clean (occupancy magnitude carries offset separably from error geometry)

## Boundary of the claim

> DIAGNOSTIC-ONLY logit-space mechanism audit. Factor families FROZEN (no feature selection). The recovery is identity-ENTANGLED (disclosed); NOT identity-free, NOT a selector, NOT deployable calibration. Target labels entered ONLY the quarantined post-hoc alignment, never the factor path.