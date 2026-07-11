# Stale Claim Audit

- status: `pass`
- scanned Markdown/JSON files: `122`
- active_stale_error_count: `0`
- excluded self-referential outputs: `STALE_CLAIM_AUDIT.md/json` only

## Pattern Coverage

| pattern | query | hits |
|---|---|---:|
| `legacy_G_0.0604` | `G=+0.0604 / G = +0.0604` | 1 |
| `cho2017_drives` | `Cho2017 drives` | 0 |
| `no_same_split_official` | `no same-split H2CMI official SPDIM` | 0 |
| `no_official_spdim` | `no official SPDIM` | 0 |
| `do_not_claim_official` | `Do not claim an official SPDIM comparison` | 0 |
| `seed_zero_only` | `seed-0 only` | 0 |
| `legacy_w1_confirmatory_false` | `current_w1_results_can_be_used_as_confirmatory=false` | 3 |
| `legacy_p6_spdim_false` | `current_spdim_p6_can_be_used_as_seed0_baseline=false` | 3 |

## Classified Hits

| path:line | section | pattern | classification | statement |
|---|---|---|---|---|
| `h2cmi/results/review_completion/MANUSCRIPT_NUMBERS_READY.md:73` | Superseded Legacy Contiguous-Split Diagnostic | `legacy_G_0.0604` | `correctly_labeled_legacy_history` | The old W1 analysis reported `G = +0.0604` and Cho2017 `G = +0.1227`. These |
| `h2cmi/results/review_completion/w1_legacy_split_quarantine.json:42` | json_document | `legacy_p6_spdim_false` | `correctly_labeled_legacy_history` | "current_spdim_p6_can_be_used_as_seed0_baseline": false, |
| `h2cmi/results/review_completion/w1_legacy_split_quarantine.json:43` | json_document | `legacy_w1_confirmatory_false` | `correctly_labeled_legacy_history` | "current_w1_results_can_be_used_as_confirmatory": false, |
| `h2cmi/results/review_completion/w1_split_metric_impact_verdict.json:12` | json_document | `legacy_p6_spdim_false` | `correctly_labeled_legacy_history` | "current_spdim_p6_can_be_used_as_seed0_baseline": false, |
| `h2cmi/results/review_completion/w1_split_metric_impact_verdict.json:13` | json_document | `legacy_w1_confirmatory_false` | `correctly_labeled_legacy_history` | "current_w1_results_can_be_used_as_confirmatory": false, |
| `h2cmi/results/review_completion/w1_split_metric_impact_verdict.md:12` | W1 Split/Metric Impact Verdict | `legacy_w1_confirmatory_false` | `correctly_labeled_legacy_history` | - current_w1_results_can_be_used_as_confirmatory: `False` |
| `h2cmi/results/review_completion/w1_split_metric_impact_verdict.md:13` | W1 Split/Metric Impact Verdict | `legacy_p6_spdim_false` | `correctly_labeled_legacy_history` | - current_spdim_p6_can_be_used_as_seed0_baseline: `False` |

Historical statements are retained only where the document or section explicitly marks them as legacy, superseded, quarantined, or resolved. No active writer-facing stale claim remains.
