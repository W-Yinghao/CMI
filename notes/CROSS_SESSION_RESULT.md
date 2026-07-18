# Cross-Session 5-arm result — CROSS_SESSION_PROXY_INVALID / GENERIC_EXTRA_TRAINING (NEGATIVE, verified)

**Branch** `agent/cmi-trace-cross-session-objective-audit` (fleet @ `aff9a34d`). **Manuscript FROZEN.** Only the
project owner may stop or redirect this line.

## What was run
Direct full real-EEG 5-arm continuation training (the 4th-discipline re-scope: no cheap proxy first). From ONE
hash-verified ERM warm-up per bundle, identical epochs/optimizer/LR/checkpoint-selection, **matched dropout across
arms** (global torch RNG reseeded per arm), the ONLY difference the aux loss:
- **A** ERM-continue (extra-training-budget control, λ=0)
- **B** CS-RW-MCC — mechanism-consistency cosine geometry weighted by cross-session source instability `W`
- **C** weight-permuted B (per-pair subject permutation of `W`)
- **D** direct cross-session risk — weighted late-session predictive CE, no geometry
- **E** weight-permuted D

`W` = source-only per (subject d, class-pair): fit `h^early_{-d}` on the EARLIEST session of the OTHER source
subjects, `r^sess = [l_late − l_early]_+`, winsor-p90 / mean-norm / clip-4. λ=1.0, 5-epoch ramp, 20 continuation
epochs, LR 1e-4; best checkpoint on **SOURCE-val** bAcc (no target labels). 63 bundles (BNCI2014_001 sub1-9 +
BNCI2015_001 sub1-12 × seed0-2) = 315 arms. Target X/Y eval-only; the exact-gradient target alignment is an
audit-only CO-DIAGNOSTIC (not a gate).

## Result (inference unit = target subject, 3 seeds→subject first, subject-cluster bootstrap + exact sign-flip)
| dataset | dU_B−A | dU_D−A | **dU_B−C** (decisive) | **dU_D−E** (decisive) | dU_D−B |
|---|---|---|---|---|---|
| BNCI2014_001 (n=9)  | −0.0003 p=.58 | −0.0032 p=.90 | +0.0019 [lcb −0.0018] p=.19 | −0.0040 p=.92 | −0.0029 p=.90 |
| BNCI2015_001 (n=12) | +0.0012 p=.19 | +0.0007 p=.40 | −0.0006 p=.68 | −0.0007 p=.63 | −0.0005 p=.60 |

**Every endpoint LCB on both datasets is ≤0.** No cross-session objective — MCC-mediated (B) or direct late-risk
(D) — beats extra-training (A) or its permuted control decisively. **damaged=False** both datasets (worst source
drop +0.010/+0.015 < 0.02 tol), **0 noop bundles** → the null is NOT masking damage or empty weights. Arms are
healthy and measurably diverge (within-cell target-bAcc spread median 0.012, max 0.049) — a real, controllable
SOURCE mechanism that is **DG-inert**. Verdict: **CROSS_SESSION_PROXY_INVALID_or_GENERIC_EXTRA_TRAINING**. This is
the **4th consecutive phase** landing on the same measurement→control gap (M1-P / MCC-λ1 / RW-MCC / this).

## Co-diagnostic (non-gating, target labels audit-only)
The DIRECT cross-session risk gradient is **positively** aligned with the target future-session gradient
(cs_risk +0.073 / +0.141, exceeds plain source-task +0.018/+0.045 and a random baseline) while the MCC geometry is
**negatively** aligned (cs_rw −0.078 / −0.031). Yet neither converts to trained DG: dU_D−A and its permuted twin
dU_D−E are both ≤0 on both datasets. First-order source→target alignment does NOT realize DG utility — a clean
measurement→control gap. (At the subject inference unit the aggregate cs_risk alignment is not significantly ≠0,
z≈1.1–1.2, and ~40% of subjects' source direction points AWAY from their own future drift.)

## Adversarial verification (5-agent workflow, all 4 skeptics HOLDS, judge high-confidence)
- **Aggregation correct**: independent recompute from the 63 raw manifests reproduces every mean to printed
  precision; target_bacc verified against raw logits; genuine held-out LOSO; correct signs; null ≠ sign-cancellation.
- **No hidden positive**: the strongest mined subgroup (BNCI2014 top-half-by-cs_risk-alignment → dU_B−C +0.0057,
  p=0.031, 5/5) disintegrates — p=0.031 is the exact-test FLOOR at n=5, fails leave-one-out (→0.062), REVERSES on
  BNCI2015 (−0.0008, frac>0=0.33), and is 1 of ~16 subgroup tests (E[hits]≈0.8). Nothing replicates across datasets.

## Disclosed caveat (SCOPE, not a bug) — the partial-λ / low-selected-epoch regime
Source-val best-checkpoint selection rolls arms back to **median epoch 1–2 of 20** (frac≤2: A .62, B .81, C .89,
D .86, E .83); the λ ramp completes only at epoch 5, so aux arms deploy at ~40% average λ and ~75% of cells roll
every aux arm to λ<1. This is **not a plumbing bug** — it is a faithful property of the DEPLOYABLE pipeline, and it
happens for a mechanistically interpretable reason: the objective mildly depresses the ONLY label-free selection
signal (mean source-val A 0.8138 vs aux 0.8115–0.8125), so the source-only selector actively rejects it. **That is
itself the measurement→control gap** (the hidden-positive skeptic: this STRENGTHENS the negative, not excuses it).
- **Fully closed (deployable) claim**: no SOURCE-ONLY-SELECTABLE cross-session objective beats extra-training with a
  permuted-control-beating, undamaged margin. Robust, independently reproduced.
- **Bounded (not closed)**: "the FULL-STRENGTH cross-session direction is fundamentally DG-inert" — rests on an
  underpowered full-λ subset (n≤9; strict both-arms-epoch≥5 n≈1), a flat dose-response (corr(selected_epoch,dU)~0–0.05,
  corr(mean_aux,dU) −0.04/−0.18), and matched-epoch isolation (B−C −0.0003, D−E −0.0003 ≈0). In the small full-λ
  subset the decisive contrasts stay negative-leaning. `.npz` stores only the selected checkpoint, so per-epoch
  target curves for the rolled-back 75% do not exist.

## Routing / owner options (report-then-wait; only the owner redirects the line)
1. **ACCEPT** the deployable negative and pivot to the **information-regime ladder** (source-only → target-X →
   few-shot target labels; minimal-info sample-complexity of `TARGET_HINDSIGHT_ONLY`) — as the verdict already routes.
2. **Bounded DIAGNOSTIC** to retire ONLY the forced-full-λ counterpoint (NOT a DG claim, NOT deployable): on the
   existing warm-ups, fix λ=1.0 with NO ramp and NO source-val early-stop, train full 20 epochs, eval target bAcc at
   FINAL and each ep≥5 for D-vs-E-vs-A and B-vs-C, reported explicitly as the **oracle-selection ceiling**. It CANNOT
   upgrade the FROZEN verdict unless the decisive permuted-control contrast turns robustly positive on BOTH datasets.
   No new losses/backbones/samplers. (Any protocol that "rescues" a signal by selecting a late/full-λ checkpoint
   abandons source-only selection → needs target/non-source info → does not falsify the negative; it re-confirms the gap.)
3. **Do nothing further** — line exhausted per its own routing; manuscript stays FROZEN.

No CLOSED without a 3rd-dataset confirm; only external check available is BNCI2014 vs BNCI2015 and every candidate
positive reverses sign between them.
