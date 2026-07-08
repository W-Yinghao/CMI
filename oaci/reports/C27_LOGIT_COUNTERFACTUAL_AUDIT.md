# C27-B — logit counterfactual audit

> interventions that destroy the offset recovery: class_bias_center, class_uniformize, confidence_shuffle, class_shuffle -> the recovery depends on the factor(s) they remove

baseline (raw full-R3) gap +0.491

| intervention | gap closed | perm p | survives | destroys recovery |
|---|---:|---:|:--:|:--:|
| raw | +0.491 | +0.024 | True | False |
| temperature | +0.512 | +0.014 | True | False |
| class_bias_center | -0.585 | +0.802 | False | True |
| logit_norm_normalize | +0.384 | +0.066 | False | False |
| class_uniformize | +0.073 | +0.423 | False | True |
| confidence_shuffle | -0.396 | +0.824 | False | True |
| class_shuffle | -0.935 | +0.944 | False | True |