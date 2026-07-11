# Review Completion Summary

> **CURRENT STATUS:** P10 is the canonical repaired-W1 evidence freeze.
> Contiguous-split W1 results are quarantined and may be used only as legacy
> diagnostics. `MANUSCRIPT_NUMBERS_READY.md` is the canonical all-numbers
> writer digest.

This package is an additive review-completion result/support package. It does
not imply that the unresolved orthogonal-score or montage-remapping concerns
have been solved.

## Canonical Repaired-W1 Evidence

- P7 H2CMI repaired W1 is confirmatory: 115 target subjects, source seeds
  0/1/2, balanced adaptation/evaluation blocks, and no split overlap.
- P9 is the complete official SPDIM repaired-split three-source-seed same-split
  baseline: 1,380 accepted rows covering source-only TSMNet, RCT, SPDIM
  geodesic, and SPDIM bias.
- P10 freezes the standardized same-split full-pipeline comparison in
  `FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md/json` and
  `w1_repaired_cross_pipeline_results.*`.
- H2CMI versus SPDIM is not an adapter-only controlled comparison: backbone,
  source objective, baseline, feature space, and adaptation action family
  differ.

Canonical H2CMI repaired-split headlines:

- `G=+0.0078277 [+0.0026504,+0.0132705]`
- `P=-0.0096602 [-0.0136878,-0.0057326]`
- `I_int=+0.0052383 [+0.0012029,+0.0093253]`
- full joint delta `+0.0034058 [-0.0021352,+0.0091774]`

Canonical official SPDIM subject-weighted headlines:

- source-only TSMNet `0.5419807 [0.5334460,0.5508986]`
- RCT `0.6471643 [0.6304244,0.6637672]`
- SPDIM geodesic `0.6444235 [0.6277308,0.6610388]`
- SPDIM bias `0.6431530 [0.6264633,0.6599215]`
- RCT minus source-only `+0.1051836 [+0.0918437,+0.1189907]`
- SPDIM geodesic minus RCT `-0.0027407 [-0.0046506,-0.0008260]`
- SPDIM bias minus RCT `-0.0040113 [-0.0072577,-0.0007150]`

RCT improves over source-only. SPDIM geodesic and bias do not improve over
RCT. These results do not support equivalence or noninferiority.

## Supporting Current Evidence

- Sleep deterministic replay, confusion matrices, per-stage recall, and the
  Sleep rows of the four-branch analysis remain current.
- Corrected V2P unit-key reanalysis remains current; displacement is not
  utility, and the oracle-label diagnostic is not deployable.
- Existing geometry stress for null/reref/gain/dropout remains current.
- Off-diagonal rotation/mixing/strong-reref/block-mixing stress is accepted as
  exploratory/supplemental bounded operator-family evidence.
- Encoder/backbone implementation details remain current, with P9 supplying
  the official TSMNet/SPDIM runtime configuration.

## Superseded Evidence

- Original REVIEW_P0 contiguous-split W1 results, including the old MI
  four-branch and heterogeneity tables, are legacy diagnostics only.
- P6 contiguous-split SPDIM seed-0 artifacts are superseded by the repaired
  split and must not be used as a current baseline.
- P8 repaired-split seed-0 SPDIM artifacts remain valid execution evidence but
  are superseded as final-baseline evidence by P9's three-seed packet.

## Current Blockers

1. Orthogonal-score estimator/evaluation: score and Fisher interfaces and a
   frozen evaluation artifact remain unavailable.
2. Montage-layout or cross-montage remapping stress: not evaluated by the
   bounded geometry panels.

No additional GPU work is required for the canonical repaired-W1 freeze.

## Current Claim Boundary

- Cite the repaired W1 values only.
- Describe H2CMI and SPDIM as same-split full-pipeline results, not a controlled
  adapter-only comparison.
- Do not claim equivalence, noninferiority, orthogonal-score evaluation, or
  universal montage robustness.
- Do not label internal `Latent-IM-Diag` as official SPDIM.

## Canonical Files

- `REVIEW_COMPLETION_CURRENT_STATUS.md/json`
- `CANONICAL_EVIDENCE_INDEX.md/json`
- `MANUSCRIPT_NUMBERS_READY.md`
- `FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md/json`
- `w1_repaired_h2cmi_results.csv`
- `spdim_w1_repaired_three_seed_results.csv`
- `w1_repaired_cross_pipeline_results.*`
- `w1_repaired_cross_pipeline_harm.*`
- `STALE_CLAIM_AUDIT.md/json`
