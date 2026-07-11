status: RESOLVED_BY_P9
resolved_commit: 8972de878a93e00a5b6cf6b8118bc32adc05eb48
canonical_result: spdim_w1_repaired_three_seed_results.csv

# Official SPDIM Baseline Blocker: Historical Record

This file preserves the pre-P4/P9 blocker history. It is not an active blocker
or a current result pointer. Current interpretation is frozen in
`FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md`.

## Historical Blocker

At revision `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`, the external official
SPDIM repository imported successfully, but its demo protocol and pretrained
weights targeted `BNCI2015_001` with a 13-channel model. Those weights were not
compatible with the H2CMI repaired-W1 datasets and could not establish a fair
baseline. The required work was exact-split TSMNet source training followed by
unlabeled official SPDIM adaptation and held-out evaluation.

## Resolution

P4 through P9 implemented and audited that path without vendoring third-party
code or using official pretrained weights. P9 completed all 115 targets for
source seeds 0/1/2 and four methods, producing 1,380 accepted rows.

The resolved baseline is:

- result: `spdim_w1_repaired_three_seed_results.csv`
- summary: `spdim_w1_repaired_three_seed_summary.json`
- result SHA-256:
  `95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3`

Internal `Latent-IM-Diag` remains an H2CMI comparator and must not be relabeled
as official SPDIM.
