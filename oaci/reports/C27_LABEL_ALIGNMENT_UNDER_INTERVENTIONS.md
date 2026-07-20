# C27-D — label alignment under interventions (LABEL-DIAGNOSTIC-ONLY, quarantined)

> class_bias_center destroy(s) BOTH offset recovery and error-geometry alignment (coupled); BUT class_uniformize destroy(s) offset recovery while PRESERVING alignment -> coupling is PARTIAL, not clean (occupancy magnitude carries offset separably from error geometry)

- raw predmix↔per-class-recall corr +0.881; offset & alignment coupled: True (class_bias_center)

| intervention | predmix↔recall corr | alignment destroyed |
|---|---:|:--:|
| raw | +0.881 | False |
| temperature | +0.881 | False |
| class_bias_center | +0.402 | True |
| logit_norm_normalize | +0.881 | False |
| class_uniformize | +0.858 | False |