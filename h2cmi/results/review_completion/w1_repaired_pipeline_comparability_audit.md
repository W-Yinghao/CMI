# Repaired-W1 H2CMI/SPDIM Pipeline Comparability Audit

- status: `pass`
- H2CMI result commit: `bc61ee11d21e023966fa9be637e960fdaf77a9c1`
- SPDIM result commit: `8972de878a93e00a5b6cf6b8118bc32adc05eb48`
- H2CMI result SHA-256: `6d5106a78dad9ce852c8e01ca292ef5b4a37bbeaaaac810a177dccb8b6b9089c`
- SPDIM result SHA-256: `95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3`
- repaired manifest hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`
- standardized interval label: `posthoc_cross_pipeline_comparability_audit`

## Gate

| field | value | evidence/interpretation |
|---|---|---|
| `same_target_subjects` | `true` | Both packets cover the same 115 repaired-W1 targets. |
| `same_adaptation_trial_ids` | `true` | Both consume the frozen manifest; all result split hashes match it. |
| `same_evaluation_trial_ids` | `true` | Both consume the frozen manifest; SPDIM index hashes also recompute exactly. |
| `same_source_seeds` | `true` | Both use source seeds 0, 1, and 2. |
| `same_metric` | `true` | Both are summarized with balanced accuracy on the repaired evaluation blocks. |
| `same_bootstrap_cluster` | `true` | P10 standardizes both to dataset x target_subject clusters. |
| `same_backbone` | `false` | H2Encoder/HybridHead and official TSMNet are different models. |
| `same_source_training_objective` | `false` | The H2 hybrid objective differs from TSMNet cross-entropy training. |
| `same_source_only_baseline` | `false` | Identity-H2 and source-only TSMNet are different trained baselines. |
| `same_feature_space` | `false` | H2 latent z_c and the TSMNet SPD representation are different. |
| `same_adaptation_action_family` | `false` | The pipelines expose different adaptation operators. |
| `adapter_only_head_to_head_valid` | `false` | Backbone, source model, feature space, baseline, and actions are not controlled. |
| `full_pipeline_same_split_comparison_valid` | `true` | Targets, split, seeds, and metric are harmonized. |

## Valid Interpretation

This is a same-split full-pipeline comparison. It is not a controlled adapter-only comparison.

The original P7 and P9 interval configurations were not identical. P10 recomputes both under the explicitly post-hoc 10,000-replicate, seed-20260710, dataset-stratified subject-cluster policy.

## Prohibited Claims

- H2CMI outperforms SPDIM as an adaptation algorithm.
- SPDIM outperforms H2CMI as an adaptation algorithm.
- Any absolute bAcc difference is attributable solely to the adapter.
