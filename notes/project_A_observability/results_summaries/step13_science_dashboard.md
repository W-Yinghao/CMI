# Step 13 — Science Dashboard (rich R1 diagnostics + real minimal-label curves)

Scope: rich R1 diagnostics + real minimal-label curves; not SOTA.

## Key metrics

- real runs: **54** · harm-rate **0.8519** · R1 diagnostics available **1.0**
- harm-predictor bAcc — R0 **0.3342** · R1 **0.6522** · R1 beats baseline **True** · improves over Step 12 **True**
- real minimal-label best k — ≥0.8 **None** · ≥0.9 **None**
- claim boundary ok **True** · target labels used in R1 diagnostics **False**

## What changed from Step 12

1. Richer R1 diagnostics only MARGINALLY clear a high permutation null (bAcc vs perm-p95 0.6413; margin < 0.03) -> within Monte-Carlo/overfitting noise, NOT a robust predictor; R0 source-only stays below its own null (TOS-1 ceiling).
2. Real minimal-label curves: harm-sign reaches 0.8 at k=None, 0.9 at k=None (labeled slice under an iid sampling contract).
3. R1 diagnostics remain label-free; oracle per-trial labels used only for R2 curves / evaluation.

> R1 diagnostics are label-free; R0/R1 harm prediction is retrospective, not identifiability; real k>0 curves are labeled slices under an iid sampling contract. No SOTA claim.
