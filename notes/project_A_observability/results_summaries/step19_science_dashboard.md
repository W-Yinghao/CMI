# Step 19 — Science Dashboard (prior-uncertainty robustness frontier)

Scope: prior-uncertainty robustness frontier + prior-robust policy (C15); not SOTA.

## Key metrics

- real runs: **54** · median L1 flip-radius from uniform **0.165151** (q25 **0.073094** / q75 **0.304194**) · unflippable **2**
- flip within L1 ≤0.10 **0.2778** · ≤0.20 **0.6296** · ≤0.50 **0.8704**
- at ρ=0.10: robust-harm **0.6852** · ambiguous **0.2778** · robust-benefit **0.037**
- prior-robust safe adaptation exists @ρ0.10 (sign) **True** · with harm margin **False** · best policy ρ **None** τ **None**
- prior-uncertainty contract required **C15** · deployment prior identified under R1 **False** · claim boundary ok **True**

## What we learned

1. The gain sign is FRAGILE: median L1 flip-radius from uniform is 0.165151 (q25 0.073094 / q75 0.304194); 0.2778 of runs flip within L1≤0.10 and 0.6296 within ≤0.20. Only 2 runs cannot flip over the whole simplex.
2. Safe adaptation CANNOT be certified under bounded prior uncertainty: no (rho, tau) with a harm margin tau>=0.05 yields any robustly-beneficial run (best policy = none). Even at the zero-margin sign level only 0.037 of runs are robustly beneficial at rho=0.10, collapsing to 0 by rho=0.20. Robust-benefit is not attainable under declared uncertainty.
3. Identity/block is robustly justified for a meaningful fraction: robust-harm 0.6852 at rho=0.10, ambiguity 0.2778 — under bounded prior uncertainty most decisions become abstain, and robust adaptation is never certifiable here.
4. Robust bounds are over DECLARED L1 prior-uncertainty sets (C15); class deltas are oracle/evaluation-only; this is not a deployable selector and does not identify the actual target prior. No SOTA.

## What remains unknown

1. Whether the true operating prior lies within a small L1 ball of uniform (needs TU-1-grade evidence).
2. Whether class-specific harm channels can be avoided by a utility-aware acquisition.
3. Whether the sign fragility persists on clinical / non-motor-imagery EEG.

> Robust gain bounds are over DECLARED prior-uncertainty sets (C15); class deltas are oracle/evaluation-only; the actual target prior is NOT identified (Prior-Decoupled boundary). No SOTA.
