# W0.2 + W0.5 — proposed manuscript inserts (STAGED, NOT APPLIED)

Based on `W0.2_RESULTS.md` and `W0.5_RESULTS.md` (90-unit corrected V2P; V2P/FRSC switch significant; sleep
lens no switch). **Do not edit the manuscript source until sign-off.**

**Wording guards (enforced):**
- **Forbidden:** "ordinary accuracy wants prevalence."
- **Required form:** "ordinary accuracy can benefit from prevalence-aware decisions when the deployment
  prevalence is the target objective and the score model supports that decision rule."
- Sleep negative is stated as "not automatically decision-useful under real source/adapter misspecification"
  — **not** a proven causal calibration diagnosis (no independent specification diagnostic in W0.5).

## Section 7.2 — Fixed-reservoir prevalence intervention (REPLACES the old 3-point displacement text)

```latex
In the corrected q-grid fixed-reservoir intervention, FRSC remained clearly prevalence-sensitive. Evaluation-embedding displacement followed the hierarchy
pooled < FRSC < fixed-prior iterative / joint << oracle labels
(approximately 0.08 < 0.42 < 0.80 << 2.0). Thus fixed-reference soft conditioning attenuates prevalence leakage but does not remove it. The utility curves show why displacement alone is insufficient: prevalence-induced movement can be visible in latent geometry without yielding balanced-accuracy utility.
```

Then connect to W0.5 (short paragraph / subsection at end of 7.2):

```latex
The same q-grid also separates metric use from geometry movement. Under balanced accuracy, uniform decoding remains the metric-correct branch. Under ordinary accuracy with an oracle-matched deployment prevalence, prevalence-aware decoding can help, but only for operators whose score geometry supports the switch.
```

## Metric-switch subsection (new; W0.5 result)

```latex
The metric switch is real but not automatic. In the controlled two-class prevalence intervention, using the oracle q decision prior significantly improves ordinary accuracy for FRSC at both prevalence extremes (q=0.1: +0.0225, CI [+0.014,+0.032]; q=0.9: +0.0333, CI [+0.022,+0.046]), while the same prevalence-aware decision rule is harmful for balanced accuracy. This confirms the metric rule: BA uses a uniform decision prior, whereas ordinary accuracy may use a prevalence-weighted prior when the deployment prevalence is the target objective.

However, the switch is specification-dependent. In the natural sleep setting, prevalence-aware decoding does not improve ordinary accuracy; uniform decoding dominates both metrics. Thus estimated or oracle prevalence is not automatically decision-useful under real source/adapter misspecification. The operational rule is therefore: use uniform for balanced accuracy; use prevalence-aware decisions only when the deployment objective is prevalence-weighted and the score model supports that use.
```

## Sleep / W2 section — one sentence AFTER the W0.3 staged insert (do NOT clutter the W0.3 table)

```latex
The sleep metric-switch audit shows the complementary boundary condition: in this misspecified natural setting, prevalence-aware decoding does not improve ordinary accuracy, so prevalence is not automatically decision-useful even when the evaluation metric changes.
```

## Placement summary

- **7.2** = fixed-reservoir displacement hierarchy + q-grid utility + "FRSC not invariant" + "displacement ≠
  utility" + the metric-switch-under-controlled-prevalence paragraph.
- **Sleep/W2** = W0.3 mechanism table (main) + the single W0.5 boundary-condition sentence. W0.3 stays the
  main sleep mechanism; W0.5 is the metric-dependent deployment rule, kept out of the W0.3 table.

## How W0.2/W0.5 pair with W0.3 (for the Discussion)

- W0.3: even oracle sleep-stage prevalence is the wrong decision prior for **balanced accuracy**.
- W0.5: prevalence-aware decisions can help **ordinary accuracy** in a controlled, well-specified prevalence
  intervention, but not automatically under natural sleep misspecification.
- Together they defeat both strawmen ("prevalence always bad" / "use prevalence when it shifts") and give
  the paper's cleanest operational rule: the correct decision prior is **metric-dependent and
  specification-dependent**.

## Supersession

- OLD Section 7.2 fixed-reservoir 3-point (q=0.25/0.50/0.75) displacement numbers → **REPLACED** by the
  corrected 9-point q-grid hierarchy + utility curves.
- Any statement implying prevalence estimation is uniformly useless, or uniformly useful under shift, is
  **superseded** by the metric-and-specification-dependent rule.
