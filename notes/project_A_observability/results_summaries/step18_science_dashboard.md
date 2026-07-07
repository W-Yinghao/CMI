# Step 18 — Science Dashboard (harm mechanisms + prior stress)

Scope: TTA harm mechanisms + deployment-prior stress; not SOTA.

## Key metrics

- real runs: **54** · mean lost-correct **0.121148** · mean gained-correct **0.078856** · mean net gain **-0.042293**
- mixed class effects **0.963** · prior-dependent-sign **0.963** · harmful-under-all-priors **0.037** · beneficial-under-all-priors **0.0**
- uniform-harm-but-some-prior-benefit **0.8148** · uniform-benefit-but-some-prior-harm **0.1481** · mean prior-sign-width **0.442315**
- worst classes by dataset **{'BNCI2014_001': 1, 'BNCI2014_004': 0}**
- prior contract required **C14** · deployment prior identified under R1 **False** · claim boundary ok **True**

## What we learned

1. TTA harm is driven by lost-correct > gained-correct trials (mean 0.121148 vs 0.078856); per-class it is MIXED in 0.963 of runs.
2. Harm is CLASS/PRIOR-DEPENDENT: only 0.037 of runs are harmful under all priors while 0.963 are prior-dependent (a declared deployment prior flips the gain sign). The benchmark-uniform bAcc hides this; deployment utility/prior matters (contract C14). This is the Prior-Decoupled boundary: without a declared prior the gain sign is under-determined — NOT that adaptation is safe (class deltas are oracle, the true prior is unidentified under R0/R1).
3. Uniform-bAcc evaluation can MASK niche-class benefit (0.8148 of runs are uniform-harm-but-some-prior-benefit) AND can MASK deployment harm (0.1481 uniform-benefit-but-some-prior-harm) — a bAcc-positive adaptation is not prior-robust.
4. Harm-channel decomposition and prior stress are oracle/evaluation-only; the deployment prior is DECLARED (C14), never identified from R0/R1; no adaptation or SOTA claim.

## What remains unknown

1. Whether the true deployment prior can be bounded cheaply (would need TU-1-grade contracts).
2. Whether class-specific harm persists on clinical / non-motor-imagery EEG.
3. Whether a utility-aware (C14-declared) selector could avoid the worst-class harm channel.

> Harm-channel and prior-stress analyses are oracle/evaluation-only; priors are DECLARED (C14), not identified; this does NOT revive a source-only target-prior claim (Prior-Decoupled boundary). No SOTA.
