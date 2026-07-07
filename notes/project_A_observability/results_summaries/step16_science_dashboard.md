# Step 16 — Science Dashboard (benefit anatomy + sequential frontier)

Scope: benefit anatomy + sequential label-acquisition frontier; not SOTA.

## Key metrics

- real runs: **54** · benefit-rate **0.0926** · target sign-stability **0.6111**
- best static policy **None** · best sequential policy **None** (budget **None**, mean-labels **None**, adapt-cov **None**, harm **None**)
- safe adaptation requires full/near-full labels **True**
- frontier meets harm<=0.05 **False** · <=0.10 **False** · <=0.20 **False**
- claim boundary ok **True** · oracle selected deployable **False**

## What we learned

1. Beneficial cells are rare (benefit-rate 0.0926) and their sign is 0.6111 consistent across seeds per target; beneficial gains are small (q90 bAcc 0.015281) -> the Step-15 false positives are explained by rare, small, unstable benefit.
2. Sequential label acquisition does NOT rescue Step 15: no sequential policy meets harm<=0.05 with coverage>=0.05 at any budget -> minimal labels do not enable safe adaptation selection; identity/default remains safest.
3. Frontier: any deployable policy meets harm<=0.05 False, <=0.10 False, <=0.20 False -- shows whether the 0.05 constraint was simply too strict.
4. Benefit anatomy is oracle-only; sequential policies are R2 labeled slices under an iid sampling contract; the oracle policy is an evaluation-only upper bound, never deployable.

## What remains unknown

1. Whether benefit rarity/instability holds on clinical / non-motor-imagery EEG.
2. Whether a label-free coverage proxy could pre-screen which targets to label.
3. Whether active (non-iid) acquisition beats iid sampling at fixed harm.

> Benefit anatomy is oracle/evaluation-only; sequential policies are R2 labeled slices under an iid sampling contract, NOT R1 identifiability; oracle policy not deployable. No SOTA.
