# W1 Repaired H2CMI Legacy Compare

Legacy rows are quarantined diagnostic-only because P6.1 found Cho2017 single-class evaluation under the old split. This file compares magnitudes only; it does not rehabilitate the legacy split.

| metric | legacy subject-weighted | repaired subject-weighted | legacy dataset-macro | repaired dataset-macro |
|---|---:|---:|---:|---:|
| identity_uniform | 0.671176 | 0.673622 | 0.675636 | 0.677485 |
| joint_geometry_uniform | 0.731595 | 0.681449 | 0.723613 | 0.686697 |
| G | 0.060419 | 0.007828 | 0.047978 | 0.009212 |
| P | -0.006521 | -0.009660 | -0.007081 | -0.009421 |
| I_int | 0.004333 | 0.005238 | 0.004952 | 0.005667 |
| full_joint_delta | 0.058232 | 0.003406 | 0.045848 | 0.005458 |

## Red Team Review

- Legacy rows remain diagnostic-only.
- Repaired rows use `class_stratified_half` and have no single-class evaluation rows.
- This comparison is not a claim that old and repaired splits are directly interchangeable.
