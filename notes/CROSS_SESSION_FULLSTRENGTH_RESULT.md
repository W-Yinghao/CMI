# Cross-Session FULL-STRENGTH oracle-ceiling — FS-C (full-strength DG-null, verified) [Track A]

**Branch** `agent/cmi-trace-cross-session-fullstrength` (fleet @ post-a-fix). **Manuscript FROZEN.** Only the owner
stops/redirects a line. This is the PM-directed bounded diagnostic to retire the ONE open counterpoint to the
deployable negative — "the objective was rolled back to ~40% λ by source-val early-stopping, so it was never really
tested at full strength."

## What was run (three changes ONLY vs the deployable 5-arm run)
Same 5 arms (A=ERM-continue, B=CS-RW-MCC, C=weight-perm B, D=direct cross-session risk, E=weight-perm D) from the SAME
hash-verified ERM warm-up, matched dropout, identical optimizer/LR/sampler — but **λ=1.0 from epoch 0 (no ramp),
exactly 20 epochs, NO source-val checkpoint rollback** (the final epoch-20 model is dumped). Per-epoch metrics logged
at {0,1,2,5,10,15,20} (target bAcc = trajectory diagnostic; target labels never enter training/weights). 63 cells, 315
arms. PRIMARY = epoch-20 decisive dU_B-C^(20)/dU_D-E^(20); max_{e≥5} = a NON-DEPLOYABLE target-epoch oracle upper bound.

## Result = FS-C_FULL_STRENGTH_CROSS_SESSION_OBJECTIVE_DG_NULL
| dataset | dU_B-A^20 | dU_D-A^20 | dU_B-C^20 (decisive) | dU_D-E^20 (decisive) |
|---|---|---|---|---|
| BNCI2014 (n=9)  | −0.0001 p=.53 | −0.0007 p=.71 | **−0.0010** [lcb −0.0071] | **−0.0024** [lcb −0.0078] |
| BNCI2015 (n=12) | +0.0012 p=.35 | −0.0006 p=.58 | **−0.0037** [lcb −0.0100] | **−0.0060** [lcb −0.0138] |

**At the deployable fixed endpoint (epoch 20), the decisive contrasts are NEGATIVE on both datasets and both contrasts
(informed ≤ its permuted control).** Full strength, applied from epoch 0 with no rollback, does NOT resurrect any
cross-session signal → **the deployable negative is NOT a ramp / early-stopping artifact.** The λ-inert / rolled-back
counterpoint is retired.

## The oracle-UB is a max-selection ARTIFACT (adversarially verified, high confidence)
The raw max_{e≥5} oracle upper bound looked positive (B-C +0.0059/+0.0055, D-E +0.0069/+0.0070, all LCB>0) — but a
5-way adversarial recomputation (workflow wffz1285v) proved it is **entirely max-of-4-noisy-draws selection inflation**:
- **Unbiased LEVEL** (mean over e≥5, not max): B-C −0.0008/−0.0032, D-E −0.0011/−0.0028 — **null-to-negative, all LCB<0**.
- **Matched null**: the identical max operator on a control-vs-control contrast (C−E, which carries no informed signal)
  manufactures an EQUAL-OR-LARGER LCB>0 positive (+0.008/+0.011, p=.002/.009).
- **NET vs matched null** (informed maxE − C−E maxE, paired): B-C −0.0024/−0.0050, D-E −0.0014/−0.0035 — **negative
  everywhere**; informed never exceeds the matched null.
- Parametric check: measured σ≈0.007–0.008, epoch autocorr ≈0.68 → E[max−mean] ≈ +0.008–0.009, matching the observed
  bias. Correlation does not rescue it.
The aggregator now routes on the LEVEL + NET (a real epoch effect must have a positive UNBIASED level AND beat the
matched max-selection null); it labels the raw maxE "SELECTION-INFLATED" and routes **FS-C**. There is no real (even
non-deployable) epoch-specific ceiling.

## Bottom line for the PM
The full-strength diagnostic **confirms and strengthens the deployable negative**: cross-session geometry consistency
(B) and direct late-session risk (D), applied at full λ with no source-val rollback, are DG-null and anti-specific at
the deployable epoch, and the apparent oracle-epoch ceiling is a statistical artifact of max-over-epochs selection.
The mechanism-consistency / cross-session line is exhausted as a source-side DG lever (5th consecutive
measurement→control gap). This does NOT touch the frozen manuscript. The information-regime ladder (Track B) carries
the next question: how much TARGET information is needed to identify any useful action.
