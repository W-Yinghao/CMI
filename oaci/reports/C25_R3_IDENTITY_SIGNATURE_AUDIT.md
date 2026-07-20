# C25 Q2 — R3 target-identity signature audit

> R3 features are target-id separable (expected for per-target moments), BUT the recovery SURVIVES the LOTO offset-permutation null -> it is target-marginal geometry, not a pure identity fingerprint (which cannot help a held-out unseen target).

- R3-feature target-id accuracy: **+0.671** (chance +0.111, source ref +0.541); identity-separable: True
- recovery survives LOTO offset-permutation: **True**; identity signature dominates: **False**
- recovering family **pred_class_prop** vs most-identity family **pred_class_prop** → dissociated: **False**

| family | target-id acc | gap closed | survives perm |
|---|---:|---:|:--:|
| confidence_entropy | +0.375 | -0.580 | False |
| margin_logitnorm | +0.429 | -0.265 | False |
| pred_class_prop | +0.706 | +0.003 | False |