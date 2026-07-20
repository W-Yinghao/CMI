# C28 — source-factor offset prediction audit

> the source-only class-conditioned confidence gauge does NOT recover the target offset -> source-unobservability confirmed at the logit-factor level

- target-carrier reference gap +0.524; source predicts offset: False

| source gauge | gap closed | perm p | survives |
|---|---:|---:|:--:|
| target_carrier_reference | +0.524 | +0.032 | True |
| source_carrier__source_guard | -0.378 | +0.982 | False |
| source_occupancy__source_guard | +0.384 | +0.134 | False |
| source_global_confidence__source_guard | -0.410 | +0.974 | False |
| source_carrier__source_audit | -0.797 | +0.956 | False |
| source_occupancy__source_audit | -0.499 | +0.898 | False |
| source_global_confidence__source_audit | -0.626 | +0.984 | False |