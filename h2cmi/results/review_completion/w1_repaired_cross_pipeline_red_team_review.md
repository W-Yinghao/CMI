# P10 Independent Red-Team Review

## Verdict

No blocker. The P10 comparability audit and final evidence freeze are
technically and scientifically suitable for commit.

## Independent Checks

- Verified the P7 source is byte-identical to result commit
  `bc61ee11d21e023966fa9be637e960fdaf77a9c1`, SHA-256
  `6d5106a78dad9ce852c8e01ca292ef5b4a37bbeaaaac810a177dccb8b6b9089c`.
- Verified the P9 source is byte-identical to result commit
  `8972de878a93e00a5b6cf6b8118bc32adc05eb48`, SHA-256
  `95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3`.
- Independently reconstructed the 345-row repaired manifest semantic hash
  `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`.
  All row-level split hashes and P9 adaptation/evaluation index hashes match.
- Verified exact coverage: 115 targets x source seeds 0/1/2. P7 contains 345
  target-seed units; P9 contains 345 target-seed units x four methods.
- Verified the six requested H2CMI mappings and four requested SPDIM mappings.
- Independently recomputed all 50 standardized result rows. The maximum
  numerical difference was exactly zero.
- Independently recomputed all 120 harm rows. The maximum numerical
  difference was exactly zero.
- Verified seeds are averaged before aggregation and the bootstrap uses
  10,000 dataset-stratified target-subject cluster replicates with seed
  `20260710`, preserving paired methods within sampled subjects.
- Verified every newly standardized interval is labeled
  `posthoc_cross_pipeline_comparability_audit`.
- Recomputed the four H2CMI decomposition headlines and all five official
  SPDIM paired contrasts and confidence intervals.

## Adversarial Findings and Resolution

1. The initial generator emitted several comparability/freeze verdicts as
   literal booleans. They are now derived from result coverage, source summary
   gates, pipeline-property comparisons, source-seed coverage, and accepted CI
   predicates.
2. Dataset-macro harm has no meaningful integer numerator because it averages
   three rates with unequal subject denominators. The artifact therefore uses
   `harm_count=NA`, denominator `3` datasets, and the mean of the three dataset
   rates; raw subject counts are retained only as context.
3. The same numeric source seeds do not imply the same initialization or
   training trajectory across different backbones. No cross-pipeline paired
   adapter contrast is reported.

## Claim Boundary

- Same target subjects, trial IDs, source-seed labels, metric, and standardized
  bootstrap cluster: true.
- Same backbone, source objective, source-only baseline, feature space, and
  adaptation action family: false.
- Same-split full-pipeline comparison: supported.
- Controlled adapter-only H2CMI-versus-SPDIM comparison: not supported.
- Equivalence and noninferiority claims: not supported.

## Residual Risk

P7 result rows bind trial IDs through the complete manifest split hash rather
than storing trial IDs directly. The accepted P7 runner consumes those IDs
explicitly, so the evidence is cryptographically strong but indirect at the
result-row level. Downstream readers must preserve dataset-macro
`harm_count=NA`; coercing it to zero would change the estimand.
