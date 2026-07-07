# C24-A — source-only non-identifiability witnesses

> CROSS-TARGET source distance does NOT predict offset distance (Mantel 0.212, p 0.020) — all-pairs 0.343 was a within-target block artifact; 9 near-source/divergent-offset collisions -> source is non-identifying for the offset

- units (target×regime): 27; pairs: 351 (324 cross-target)
- Mantel corr(source-dist, offset-dist): all-pairs **+0.343** (p +0.005) → **CROSS-TARGET +0.212** (p +0.020)
- within-target block confound detected: **True** (all-pairs correlation inflated by same-target pairs); source predicts offset (cross-target ≥0.3): **False**
- near-distance threshold (bottom 15%): +5.228; far-offset threshold (top 15%): +0.327
- near-source/divergent-offset collisions: **9**; source non-identifying: **True**

| unit A | unit B | source dist | offset diff | strength |
|---|---|---:|---:|---:|
| t4:S0_full_support | t3:S2_rare_cells | 3.7594 | 0.5316 | 0.1414 |
| t3:S0_full_support | t4:S0_full_support | 3.8281 | 0.5098 | 0.1332 |
| t4:S0_full_support | t3:S3_nonestimable_cells | 4.0301 | 0.5213 | 0.1294 |
| t3:S2_rare_cells | t4:S2_rare_cells | 4.3879 | 0.5287 | 0.1205 |
| t3:S3_nonestimable_cells | t4:S3_nonestimable_cells | 4.4479 | 0.5235 | 0.1177 |
| t3:S2_rare_cells | t4:S3_nonestimable_cells | 4.5468 | 0.5338 | 0.1174 |
| t4:S2_rare_cells | t3:S3_nonestimable_cells | 4.4733 | 0.5184 | 0.1159 |
| t3:S0_full_support | t4:S2_rare_cells | 4.7812 | 0.5069 | 0.106 |
| t3:S0_full_support | t4:S3_nonestimable_cells | 5.1277 | 0.512 | 0.0998 |