# Blockers

## Active

1. Orthogonal-score estimator/evaluation: score and Fisher interfaces and a
   frozen evaluation artifact remain unavailable. Header-only result files are
   blocker placeholders, not negative findings.
2. Montage-layout or cross-montage remapping stress: the bounded geometry
   panels do not test channel-layout remapping or a learned montage map.

## Resolved

- Official SPDIM baseline: resolved by P9 commit
  `8972de878a93e00a5b6cf6b8118bc32adc05eb48`. Canonical result:
  `spdim_w1_repaired_three_seed_results.csv`.
- Repaired W1 class composition: resolved by the P7 repaired manifest and
  confirmatory H2CMI rerun; legacy contiguous-split rows remain quarantined.
- Off-diagonal bounded geometry stress: rotation, cross-channel mixing,
  stronger rereferencing, and block mixing were executed and audited as
  exploratory/supplemental evidence.

No additional GPU work is required for the P10 canonical W1 evidence freeze.
