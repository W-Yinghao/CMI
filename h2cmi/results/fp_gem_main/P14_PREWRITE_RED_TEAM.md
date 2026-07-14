# P14 Pre-Write Red-Team Gate

Status: `PASS`.

This gate was completed before authoring the P14 writer-facing freeze. It read only committed P12/P13 artifacts and raw validation caches; it launched no training or adaptation run and wrote no P12/P13 source artifact.

## Independent Checks

- P12 result SHA-256 matches `f3e4ca699b81e4fa2cab404109aa2dfe7aa1fbe58f25e2779d3d11651e40d48d`.
- P12 rebuilt 189 target-seed units and 63 seed-averaged subject clusters; all 31 completion, provenance, metric, and fixed-bootstrap gates passed.
- P12's five subject-weighted FP-GEM contrasts reproduce exactly. FP-GEM improves over source-only, is lower than RCT and both SPDIM variants, and is inconclusive against Joint-GEM.
- P13 result SHA-256 matches `cf9e403eb8be1c0548a95f9007eb7089ee3f93d8bee2401af22587903bffdb2f`.
- P13 independently reconstructed all 162 evaluation blocks, recomputed all 2,916 prediction rows and 10,000 bootstrap replicates, and found zero metric or interval mismatches.
- P13's FP-GEM-minus-Joint-GEM sensitivity interval crosses zero. The only supported external sensitivity contrast is FP-GEM minus RCT, while FP-GEM endpoint performance is lower than RCT and both SPDIM variants.

## Adversarial Claim Gate

- Do not claim FP-GEM outperforms RCT or SPDIM.
- Do not claim FP-GEM is more prevalence-robust than Joint-GEM or SPDIM.
- Do not claim universal prevalence invariance, state of the art, equivalence, or noninferiority.
- Keep P12 as a two-dataset same-backbone comparison, not a broad benchmark.
- Keep P13 in the appendix as a limitation and boundary test.

The P14 authoring gate is open only under these boundaries.
