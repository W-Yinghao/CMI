# RQ2 claim hardening — sensitivity of the erasure→target result

Primary claim (**READY**, independent of subsets): *subject signal is erasable, but erasure strength does not certify target benefit* — `benefit_claimable = 0/40`.

Supporting result (**NOT_ROBUST_DO_NOT_HEADLINE**): *stronger subject removal is negatively associated with target bAcc.* Sensitivity:

| subset | rho | 95% CI | n | sign | excl 0 |
|---|---|---|---|---|---|
| 1_all_cells | -0.4204 | [-0.6824,-0.1133] | 40 | neg | True |
| 2_clean_cells_no_collapse_no_harm | -0.4348 | [-0.7311,-0.0574] | 30 | neg | True |
| 3_excl_random_k | -0.2617 | [-0.5867,0.1251] | 32 | neg | False |
| 4_excl_INLP | -0.3086 | [-0.6448,0.0617] | 32 | neg | False |
| 5_excl_INLP_and_random_k | -0.1453 | [-0.5848,0.2969] | 24 | neg | False |
| 6_LEACE_RLACE_only | 0.5394 | [0.0671,0.8617] | 16 | pos | True |
| 7_within_dataset_backbone_rank_resid | -0.7014 | [-0.8576,-0.4966] | 40 | neg | True |

Per-dataset / per-backbone sign (all negative required for a broad claim):

| group_kind | group | rho | sign | excl 0 |
|---|---|---|---|---|
| 8_per_dataset | BNCI2014_001 | -0.3818 | neg | False |
| 8_per_dataset | Cho2017 | -0.3497 | neg | False |
| 8_per_dataset | Lee2019_MI | -0.4658 | neg | False |
| 8_per_dataset | Schirrmeister2017 | -0.6364 | neg | True |
| 9_per_backbone | EEGNet | -0.3105 | neg | False |
| 9_per_backbone | TSMNet | -0.5789 | neg | True |
| cell_dataset_backbone | BNCI2014_001|EEGNet | -0.8 | neg | True |
| cell_dataset_backbone | BNCI2014_001|TSMNet | 0.0 | zero | False |
| cell_dataset_backbone | Cho2017|EEGNet | -0.7826 | neg | True |
| cell_dataset_backbone | Cho2017|TSMNet | -0.8 | neg | True |
| cell_dataset_backbone | Lee2019_MI|EEGNet | -0.875 | neg | True |
| cell_dataset_backbone | Lee2019_MI|TSMNet | -0.7 | neg | False |
| cell_dataset_backbone | Schirrmeister2017|EEGNet | -0.9 | neg | True |
| cell_dataset_backbone | Schirrmeister2017|TSMNet | -0.8 | neg | True |

**Verdict:** the negative association is NOT robust: it FLIPS to positive (rho=0.5394) on the principled-eraser subset (LEACE/RLACE only) and is ns when INLP/random_k are dropped -> driven by over-erasure (INLP) + the random-k anchor, not a real 'more removal, worse target' law -> supporting result status = **NOT_ROBUST_DO_NOT_HEADLINE**.

The headline stays the **READY** primary claim (*erasure does not certify target benefit; benefit_claimable=0/40*). The negative correlation is **not** reported as a finding.

No target labels used for fit (target y is EVAL_ONLY in the underlying deploy summaries). CPU-only.
