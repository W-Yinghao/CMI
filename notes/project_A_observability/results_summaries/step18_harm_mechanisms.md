# Step 18 — TTA harm-mechanism decomposition

Scope: TTA harm-mechanism decomposition (oracle/evaluation-only); not SOTA. harm-mechanism decomposition is oracle/evaluation-only; it identifies no target functional under R0/R1 and makes no adaptation or SOTA claim.

- runs: **54** · mean lost-correct **0.121148** · mean gained-correct **0.078856** · mean net gain **-0.042293**
- runs with mixed class effects: **52** (**0.963**) · prior-dependent-possible fraction **0.963**

| dataset | most-common worst class | worst-class histogram |
|---|---|---|
| BNCI2014_001 | 1 | {2: 3, 1: 9, 0: 6, 3: 9} |
| BNCI2014_004 | 0 | {0: 22, 1: 5} |

Dominant identity->adapt wrong transitions (true-class -> predicted):

- `0->1` ×35
- `1->0` ×33
- `1->2` ×18
- `3->2` ×17
- `0->2` ×15
- `2->0` ×8
- `2->1` ×7
- `3->0` ×7

> harm-mechanism decomposition is oracle/evaluation-only; it identifies no target functional under R0/R1 and makes no adaptation or SOTA claim.
