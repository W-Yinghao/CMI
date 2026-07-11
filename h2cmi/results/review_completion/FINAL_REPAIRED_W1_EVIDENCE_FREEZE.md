# Final Repaired-W1 Evidence Freeze

## Frozen Status

- Legacy W1 split: quarantined; no legacy row enters this freeze.
- H2CMI repaired W1: confirmatory accepted packet.
- SPDIM: official repaired-split three-source-seed same-split baseline.
- Additional GPU work required: `false`.
- H2CMI source SHA-256: `6d5106a78dad9ce852c8e01ca292ef5b4a37bbeaaaac810a177dccb8b6b9089c`
- SPDIM source SHA-256: `95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3`
- repaired manifest hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`

## H2CMI Repaired-Split Headlines

- G: `+0.0078277`
- P: `-0.0096602`
- I_int: `+0.0052383`
- full joint delta: `+0.0034058`

## Official SPDIM Repaired-Split Headlines

| method | subject-weighted bAcc [95% CI] |
|---|---:|
| source-only TSMNet | 0.5420 [0.5334, 0.5509] |
| RCT | 0.6472 [0.6304, 0.6638] |
| SPDIM geodesic | 0.6444 [0.6277, 0.6610] |
| SPDIM bias | 0.6432 [0.6265, 0.6599] |

| paired contrast | subject-weighted estimate [95% CI] |
|---|---:|
| rct_minus_source_only_tsmnet | +0.1052 [+0.0918, +0.1190] |
| spdim_geodesic_minus_source_only_tsmnet | +0.1024 [+0.0895, +0.1161] |
| spdim_bias_minus_source_only_tsmnet | +0.1012 [+0.0883, +0.1149] |
| spdim_geodesic_minus_rct | -0.0027 [-0.0047, -0.0008] |
| spdim_bias_minus_rct | -0.0040 [-0.0073, -0.0007] |

## Final Claim Gate

RCT improves over source-only. Neither SPDIM geodesic nor SPDIM bias improves over RCT. Equivalence and noninferiority are not supported.

H2CMI and SPDIM can be shown in a same-split full-pipeline table, but their absolute difference cannot be attributed solely to adaptation because backbone, source objective, baseline, feature space, and action family differ.
