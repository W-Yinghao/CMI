# Step 14 — harm-prediction power / sensitivity

Scope: harm-prediction power/sensitivity; not SOTA.

- runs: **54** · harmed **46** · non-harmed **8** · minority fraction **0.1481** · groups **18**
- observed R1 bAcc **0.6522** · permutation-null p95 **0.6821** · min detectable bAcc ≈ **0.7121**
- robust signal: **False** · underpowered: **True**

> Underpowered for stable harm-prediction claims: minority n=8 (fraction 0.1481), high permutation-null p95=0.6821. A balanced-acc below ~0.7121 cannot be distinguished from the overfitting-inflated null; the observed R1 signal is NOT robust (marginal).

> Permutation-null significance is empirical retrospective evidence; it does NOT make target gain identifiable under R1. Duplicating targets is NOT evidence and is not performed.
