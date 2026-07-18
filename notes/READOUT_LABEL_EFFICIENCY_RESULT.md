# Target Readout Calibration Ladder — R-D (verified, high confidence)

**Branch** `agent/cmi-trace-readout-label-efficiency` (worktree `CMI_AAAI_readout`, base `de170ede`). **Manuscript
FROZEN.** Only the owner stops/redirects a line. 252 cells, 4 datasets (dev: BNCI2014_001 9 subj, BNCI2015_001 12
subj; external confirmatory: Lee2019_MI 54 subj, BNCI2014_004/BCI-IV-2b 9 subj) × seeds 0/1/2, frozen ERM EEGNet
features (no re-inference). Inference unit = target subject (draw→seed→subject), subject-cluster bootstrap + exact
sign-flip, Holm across 7 budgets.

## Verdict = R-D_READOUT_SHIFT_GENERIC_SUBSPACE_NOT_CAUSAL
Adversarial verification (workflow w9jw0bkri, high confidence, endpoints reproduced) **downgraded my first-pass R-A**:
the aggregator's no-harm gate only tested harm-vs-*fresh* (a strawman — a from-scratch head is trivially bad at 1–2
labels/class, so it never fired) and missed harm-vs-*frozen*. Fixed → **R-D**.

### 1. Anchoring (source-anchored MAP beats a from-scratch head) — REAL but a SOFT claim
`dU_MAP_fresh` k*_anchor = **1 on all four datasets**, LCB>0 few-shot: BNCI2014 k1 +0.095[+0.056]; BNCI2015 +0.085
[+0.055]; **Lee2019 +0.127[+0.107]**; 2b +0.117[+0.066] (falls to ~0/negative at Full). BUT: a fresh logistic on 1–2
labels/class is near-chance, so this is the *expected* component; and it is partly a warm-start artifact — a FIXED
α=1.0 reproduces the anchoring win (the α selection is not the source), and at α=0 the win is already +0.04–0.06
(early-stopping an overfit unregularized fresh head, not the L2-to-source penalty). The from-scratch baseline is soft.

### 2. Utility (does ADAPTING beat NOT adapting? MAP vs frozen) — the DISTINCTIVE deployable claim — does NOT externally replicate
`dU_MAP_frozen` k*_util: BNCI2014 = 2, BNCI2015 = 4 (dev, robust), but on the load-bearing external data it fails:
- **Lee2019**: sub-1% near-ceiling (k1 +0.0002 null, k4 +0.0040[+0.0008], Full +0.0067) — statistically LCB>0 only
  because n=54; scientifically marginal.
- **BNCI2014_004 (2b)**: **NEGATIVE few-shot** (k1 mean −0.011, k2 −0.008, k4 −0.005 — adaptation HURTS vs the frozen
  head), positive only at Full budget. This is the CMI-Trace line's historical sign-reversal failure mode.
So few-shot MAP does not robustly beat NOT-adapting externally; the R-A "label-efficient deployable direction" claim
rests on dev-only wins plus a marginal Lee effect and over-claims.

### 3. Subspace — CONFIRMED NON-CAUSAL (generic readout refit)
`dGh_specific` is never LCB>0 on any dataset (significantly NEGATIVE on BNCI2014, null elsewhere): deleting the
informed B_cond subspace is no better than a matched-random deletion. **Any readout benefit is a generic refit of
P(Y|Z), not the subject/CMI subspace.** No causal-subspace claim survives. (Power caveat: the source-retention filter
leaves <10 matched-random reps on many 2b/Lee cells, but the dGh means are null/negative even in well-powered cells.)

## What this establishes
- The DG bottleneck IS the target-specific **readout** (confirms IL-C), and a source-anchored *constrained* update
  regularizes few-shot heads better than from-scratch — a real but modest regularization effect.
- It is NOT a robust deployable "adapt beats the frozen head" gain externally (Lee near-ceiling, 2b few-shot harm).
- The subject **subspace is confirmed non-causal** — closing the erasure/selection line's causal hypothesis on the
  readout axis too.
- Firewall (source-only α), aggregation, subject-cluster bootstrap, and Holm are all sound; the only defect was the
  routing/labeling, now fixed.

## Owner options (report-then-wait; nothing run unilaterally; manuscript FROZEN; no CLOSED without a hardened re-test)
1. **Re-frame the claim to R-D**: "source-anchored MAP robustly beats a from-scratch few-shot readout (anchoring),
   but few-shot adaptation over the frozen head is dataset-dependent and can HARM externally (2b)" — drop
   "label-efficient utility" as a headline.
2. **Harden the from-scratch baseline** (ridge / warm-start-matched fresh head) and re-test whether the anchoring win
   survives; if it collapses, anchoring reduces to a regularization/warm-start effect.
3. **Add a budget-gated deploy rule** (adapt only when k ≥ threshold, else keep the frozen head) + report the 2b
   negative few-shot regime honestly.
4. **Investigate WHY external target sessions have no readout headroom** (Lee near-ceiling; 2b harm) vs dev — the
   frozen ERM head may already be near-optimal externally; this may be the real (negative/scoping) finding.
5. **Drop/heavily caveat any subspace-causal claim** (dGh_specific non-positive everywhere).
6. If a positive is still desired, **pre-register the anchoring claim proper** (few-shot head regularization) against
   a hardened baseline on a held-out external cohort before unfreezing the manuscript.
