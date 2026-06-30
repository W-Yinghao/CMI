# CSC Route B — B-P1 first finding (few-label core: works on the target side, source reference is the bottleneck)

DEVELOPMENT prototype, simulator-only. New namespace `csc/mininfo/`. Does NOT touch the frozen A tag
`csc-confirmatory-v1`. No freeze, no confirmatory, no real EEG.

## What was built

`csc/mininfo/few_label_test.py` — the B2 core: a **few-label conditional-risk test**. Fit a
source-calibrated posterior `p_S(y|z)`; on a few queried target subjects, test whether the labels are
more surprising than `p_S` expects of itself (log-score residual `r_i = -log p_S(Y_i|Z_i) - H(p_S(·|Z_i))`,
subject-condition-vote aggregation, conditional Monte-Carlo null `Y*~p_S(·|Z)`, one-sided p; optional
prior-correction to the audit-estimated target prior as a label-shift guard).

## Central claim — SUPPORTED

Few target labels crack the Z-only-**unidentifiable** `pure_conditional` shift (P(Z) identical to clean,
P(Y|Z) relabeled): rejection **8/8** source seeds at m = 5/10/20, with and without prior-correction. This
is the whole point of Route B — adding minimal target information moves `UNIDENTIFIABLE` → detectable,
exactly as the theory predicts.

## Type-I problem — ATTRIBUTED (the real bottleneck)

The naive test over-rejects the **controls** (which have NO concept shift). Diagnostic (m=20, 8 seeds,
prior-corrected):

| source | clean | covariate | pure_conditional |
|---|---|---|---|
| concept_domains = 0 (clean reference) | **1/8 (~control)** | 3/8 | 8/8 |
| concept_domains = 3 (heterogeneous, realistic) | 5/8 | 6/8 | 8/8 |

- **Dominant cause — source-reference contamination.** With a clean source, `clean` type-I is controlled
  (1/8); adding 3 concept domains to the source inflates it to 5/8. The pooled `p_S` averages in the
  source's own per-domain boundary movement, so it mis-predicts a genuinely-clean target near the
  boundary → false "concept" rejections. Estimating a **concept-robust reference posterior from a
  heterogeneous source is the SAME hard problem Direction A faced** (its atlas/h0 machinery existed for
  exactly this) — the few labels do not help the source side.
- **Secondary cause — covariate-shift extrapolation.** Even with a clean source, `covariate` rejects 3/8:
  `p_S` is evaluated on target Z shifted along the nuisance axis (and standardized with source moments),
  where the logistic extrapolates and is miscalibrated, even though true Y|Z is unchanged.

## Design implication (steer needed before the full build)

The few-label idea works, but a **source-reference** B2 inherits A's source-heterogeneity + covariate
problems. The more robust path is **B3 paired (target-internal reference)**: a within-subject ON/OFF
contrast tests whether Y|Z **changed between the target's own conditions**, using the target's other
condition as the reference — **no source posterior needed**, so it sidesteps BOTH the contamination and
the extrapolation confounds (and cancels the subject random effect). This strengthens the reviewer's B3
emphasis: pairing is not just a power booster, it removes the type-I confound that B2-against-source
suffers. B2 few-labels remains valuable for the unpaired case but needs a concept-robust reference +
covariate-robustness to control type-I.

**Open fork for the next B-P1 iteration:**
1. invest in a concept-robust source-reference posterior (h0-style shared boundary + domain-robust
   consensus) + covariate-robust evaluation, OR
2. prioritise B3 paired (target-internal conditional-change test; no source reference) as the primary
   confirmable mechanism, with B2 few-labels as the unpaired fallback.

No code beyond `few_label_test.py` is committed for the certifier/runner yet — deliberately, to avoid
building the wrong scaffold before this fork is decided.
