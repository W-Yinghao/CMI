# C29 — counterfactual geometry audit

> destroyers: effective_mean_removed, global_scale_removed, source_mean_centered_projection; baseline +0.524

| intervention | gap closed | survives | destroys recovery |
|---|---:|:--:|:--:|
| raw | +0.524 | True | False |
| parameter_bias_removed | +0.510 | True | False |
| effective_mean_removed | -0.313 | False | True |
| projection_only | +0.510 | True | False |
| weight_norm_normalized | +0.353 | False | False |
| global_scale_removed | -0.149 | False | True |
| source_mean_centered_projection | -0.297 | False | True |