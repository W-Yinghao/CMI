# Step 16 — harm-control policy frontier

Scope: harm-control policy frontier (R2); not SOTA. policy frontier under R2 label budgets / iid sampling contract; NOT R1 target-gain identifiability; oracle excluded from the deployable frontier.

- static cells: **160** · sequential cells: **96** · oracle excluded: **True**
- any policy meets harm<=0.05: **False** · <=0.10: **False** · <=0.20: **False**

| harm_threshold | min_coverage | best_policy | source | best_adapt_cov | best_labels |
|---:|---:|---|---|---:|---:|
| 0.05 | 0.01 | None | None | None | None |
| 0.05 | 0.05 | None | None | None | None |
| 0.05 | 0.1 | None | None | None | None |
| 0.1 | 0.01 | None | None | None | None |
| 0.1 | 0.05 | None | None | None | None |
| 0.1 | 0.1 | None | None | None | None |
| 0.2 | 0.01 | None | None | None | None |
| 0.2 | 0.05 | None | None | None | None |
| 0.2 | 0.1 | None | None | None | None |
| 0.5 | 0.01 | plugin_sign | static | 0.1692 | 256 |
| 0.5 | 0.05 | plugin_sign | static | 0.1692 | 256 |
| 0.5 | 0.1 | plugin_sign | static | 0.1692 | 256 |

> policy frontier under R2 label budgets / iid sampling contract; NOT R1 target-gain identifiability; oracle excluded from the deployable frontier.
