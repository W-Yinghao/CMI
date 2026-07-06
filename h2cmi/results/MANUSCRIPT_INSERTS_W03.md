# W0.3 — proposed manuscript inserts (STAGED, NOT APPLIED)

These are the proposed replacements for the W2 mechanism, based on `W0.3_RESULTS.md` (aggregate, 75
subjects, residual 0, main-consistency 0, `P_cross = −0.1439` matching W0.1). **Do not edit the manuscript
source until sign-off.** No h2cmi LaTeX manuscript exists yet; when it does, apply these there.

Wording guards enforced: (a) no "flatter-than-truth bias" claim (we have no entropy/majorization
diagnostic) — we say only "π_J's deviation from ρ_A partially offsets the harm"; (b) `ρ_E` is explicitly an
**oracle diagnostic**, not a deployable quantity.

## Abstract replacement

```latex
In sleep staging, the large joint-adaptation failure is not primarily a cross-night prevalence-transfer effect or a poor prior-estimation effect. A deterministic same-session control shows that even the oracle evaluation prevalence is a poor decision prior for balanced accuracy; the dominant mechanism is metric-prior mismatch, with estimated-prior deviations partially offsetting rather than causing the harm.
```

## Results / W2 replacement paragraph

```latex
Sleep staging. In W2, the conventional negative ``joint'' number decomposes sharply. The geometry branch is close to neutral:
\(G=-0.020\) with CI \([-0.041,+0.001]\). The fitted-prior decision branch is large and harmful:
\(P=-0.144\) with CI \([-0.160,-0.128]\). A deterministic W0.1 rerun preserved the terminal decomposition to four decimals and saved admissible branch-level predictions and confusions. The per-stage recalls localized the harm to minority and boundary stages under prevalence-weighted decisions.

A same-session prevalence-matched control then isolated the mechanism. With adaptation and evaluation prevalence matched, \(P\) remained large:
\(-0.134\) with CI \([-0.150,-0.118]\). Decomposing \(P\) showed that the dominant term was the balanced-accuracy metric-prior mismatch:
\(B_E(\rho_E)-B_E(\mathrm{Unif})=-0.161\) in same-session and \(-0.162\) in cross-night. The adapt/evaluation prevalence-transfer term was negligible in cross-night
\((-0.0055\), CI crossing zero) and exactly zero in same-session by construction. The fitted-prior deviation term was positive
(\(+0.024\) cross-night, \(+0.028\) same-session), meaning that \(\pi_J\)'s deviation from the adaptation prevalence partially offset the oracle-prevalence harm. Thus the dominant W2 failure is not night-to-night prevalence transfer or poor prior estimation; it is using prevalence as a decision prior for a balanced-accuracy objective.
```

## Table caption (main-text mechanistic table; per-stage table → appendix if page-limited)

```latex
Table X: Mechanistic decomposition of the W2 decision-prior branch.
For \(B_E(\pi)\), identity geometry is decoded on the evaluation set \(E\) using decision prior \(\pi\). The fitted-prior branch decomposes exactly as
\(B_E(\pi_J)-B_E(\mathrm{Unif})
=
[B_E(\rho_E)-B_E(\mathrm{Unif})]
+
[B_E(\rho_A)-B_E(\rho_E)]
+
[B_E(\pi_J)-B_E(\rho_A)]\).
The dominant term is the balanced-accuracy metric-prior mismatch: even the oracle evaluation prevalence is a harmful decision prior for balanced accuracy. Here \(\rho_E\) is an oracle diagnostic used only to localize the mechanism.
```

## Discussion replacement

```latex
The sleep result is stronger than a cross-night prior-transfer failure. The same-session prevalence-matched control shows that the large decision-prior term persists when adaptation and evaluation prevalence are equal. The exact decomposition attributes the effect to metric-prior mismatch: balanced accuracy asks for a uniform decision prior, whereas sleep-stage prevalence is highly non-uniform. Oracle prevalence therefore improves majority-stage recall at the expense of minority and boundary stages. The estimated prior \(\pi_J\) is not the primary culprit in aggregate; its deviation from the adaptation prevalence partially offsets the oracle-prevalence harm. This reinforces the operational rule: estimate prevalence if it is scientifically useful, but do not use it as \(\pi_{\mathrm{dec}}\) unless the deployment metric asks for prevalence-weighted decisions.
```

## Limitations replacement (removes the stale "W2 confusion not used" sentence)

```latex
An earlier W2 reproducibility limitation was addressed by a deterministic rerun that saved branch-level predictions and passed exact self-replay. The remaining limits are external-cohort coverage, possible source-model misspecification, and the restriction to the frozen latent diagonal family. The oracle-prevalence decomposition is a diagnostic analysis, not a deployable unlabeled procedure, because \(\rho_E\) uses evaluation labels only to localize the mechanism.
```

## Final approved one-paragraph statement (for wherever the summary claim lives)

> Sleep-stage prevalence is a poor balanced-accuracy decision prior even when known exactly and even when
> adaptation and evaluation prevalence are matched. The dominant W2 failure is metric-prior mismatch, not
> cross-night transfer or poor π_J estimation. In aggregate, π_J's deviation from the adaptation prevalence
> partially offsets the oracle-prevalence harm rather than causing it.

## Supersession bookkeeping

- OLD (W2): "night-1 estimated prevalence is a poor night-2 balanced-accuracy decision prior." →
  **SUPERSEDED** by the stronger metric-prior-mismatch statement above (still true, but not the dominant
  mechanism).
- OLD (Limitations): "W2 per-stage confusion not saved / should be saved in future." → **REMOVED** (closed
  by W0.1; see `WAVE0_MANUSCRIPT_NOTES.md`).
