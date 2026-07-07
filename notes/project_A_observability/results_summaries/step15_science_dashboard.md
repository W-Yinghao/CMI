# Step 15 — Science Dashboard (coverage-aware harm-control policies)

Scope: coverage-aware harm-control policies under minimal labels; not SOTA.

## Key metrics

- real runs: **54** · always-adapt harm-rate **0.8519**
- best deployable policy: **None** (k **None**, tau **None**)
- adaptation coverage **None** · decision coverage **None** · harm-among-adapt **None**
- prevented harm vs always-adapt **None** · missed benefit **None**
- coverage/control tradeoff observed **False** · oracle selected deployable **False** · claim boundary ok **True**

## What we learned

1. NO deployable minimal-label policy adapts while keeping harm<=0.05: the best a label-based policy achieves is adapt-coverage 0.2554 at harm-among-adapt 0.7807 (plugin_sign, k=16). Confident/positive slices do NOT select beneficial cells -- with a high harm base-rate, adapt-positive events are dominated by false positives on harmful cells.
2. The oracle full-label upper bound adapts 0.1481 of cells at harm 0.0 (prevented 1.0, missed 0.0): safe adaptation IS possible, but only with (near-)full target labels -- the measurement->control gap is NOT closed by R2 minimal-label CI policies on this grid.
3. Decisions use k>0 target labels (R2 labeled slice under an iid sampling contract); k=0 stays R1 non-identifiable; the oracle policy is an evaluation-only upper bound.

## What remains unknown

1. Whether the same policy transfers to clinical / non-motor-imagery EEG.
2. Whether cheaper-than-iid label acquisition raises coverage at fixed harm.
3. Whether a label-free coverage proxy could pre-screen which targets to label.

> R2 minimal-label policy evaluation under an iid sampling contract; NOT R1 target-gain identifiability; oracle policy is not deployable. No SOTA.
