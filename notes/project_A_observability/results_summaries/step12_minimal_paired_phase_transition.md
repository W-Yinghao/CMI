# Step 12 — minimal-paired phase transition (simulator)

Scope: minimal-paired R1->R2 phase transition (simulator); not SOTA. k=0 is the R1 non-identifiability boundary; k>0 is an R2 labeled slice under an iid sampling contract, not full-target-risk identification.

- repeats: **50** · k grid: **[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]**
- phase transition observed (harm-sign acc ≥ 0.9): **True** · best k overall: **256**
- k=0 status: **not_identified_R1**

| shift | k | true_gain | harm_sign_acc | abstention | risk_ci_width | status |
|---|---:|---:|---:|---:|---:|---|
| prior_shift_only | 0 | 0.07 | 0.5 | 1.0 | None | not_identified_R1 |
| prior_shift_only | 1 | 0.07 | 0.28 | 0.5 | 0.0 | labeled_slice_under_iid_sampling_contract |
| prior_shift_only | 2 | 0.07 | 0.14 | 0.8 | 1.1424 | labeled_slice_under_iid_sampling_contract |
| prior_shift_only | 4 | 0.07 | 0.16 | 0.84 | 1.2068 | labeled_slice_under_iid_sampling_contract |
| prior_shift_only | 8 | 0.07 | 0.08 | 0.92 | 0.8891 | labeled_slice_under_iid_sampling_contract |
| prior_shift_only | 16 | 0.07 | 0.04 | 0.96 | 0.6624 | labeled_slice_under_iid_sampling_contract |
| prior_shift_only | 32 | 0.07 | 0.06 | 0.92 | 0.4735 | labeled_slice_under_iid_sampling_contract |
| prior_shift_only | 64 | 0.07 | 0.14 | 0.86 | 0.3378 | labeled_slice_under_iid_sampling_contract |
| prior_shift_only | 128 | 0.07 | 0.2 | 0.8 | 0.2396 | labeled_slice_under_iid_sampling_contract |
| prior_shift_only | 256 | 0.07 | 0.5 | 0.5 | 0.17 | labeled_slice_under_iid_sampling_contract |
| prior_shift_only | 512 | 0.07 | 0.7 | 0.3 | 0.1202 | labeled_slice_under_iid_sampling_contract |
| concept_shift | 0 | -0.1 | 0.5 | 1.0 | None | not_identified_R1 |
| concept_shift | 1 | -0.1 | 0.36 | 0.42 | 0.0 | labeled_slice_under_iid_sampling_contract |
| concept_shift | 2 | -0.1 | 0.14 | 0.82 | 1.1289 | labeled_slice_under_iid_sampling_contract |
| concept_shift | 4 | -0.1 | 0.14 | 0.84 | 1.2212 | labeled_slice_under_iid_sampling_contract |
| concept_shift | 8 | -0.1 | 0.1 | 0.88 | 0.9099 | labeled_slice_under_iid_sampling_contract |
| concept_shift | 16 | -0.1 | 0.04 | 0.96 | 0.6705 | labeled_slice_under_iid_sampling_contract |
| concept_shift | 32 | -0.1 | 0.14 | 0.86 | 0.4792 | labeled_slice_under_iid_sampling_contract |
| concept_shift | 64 | -0.1 | 0.18 | 0.82 | 0.3425 | labeled_slice_under_iid_sampling_contract |
| concept_shift | 128 | -0.1 | 0.34 | 0.66 | 0.2428 | labeled_slice_under_iid_sampling_contract |
| concept_shift | 256 | -0.1 | 0.6 | 0.4 | 0.1721 | labeled_slice_under_iid_sampling_contract |
| concept_shift | 512 | -0.1 | 0.8 | 0.2 | 0.1218 | labeled_slice_under_iid_sampling_contract |
| support_failure | 0 | -0.16 | 0.5 | 1.0 | None | not_identified_R1 |
| support_failure | 1 | -0.16 | 0.32 | 0.58 | 0.0 | labeled_slice_under_iid_sampling_contract |
| support_failure | 2 | -0.16 | 0.14 | 0.84 | 1.0687 | labeled_slice_under_iid_sampling_contract |
| support_failure | 4 | -0.16 | 0.18 | 0.82 | 1.1646 | labeled_slice_under_iid_sampling_contract |
| support_failure | 8 | -0.16 | 0.22 | 0.78 | 0.8904 | labeled_slice_under_iid_sampling_contract |
| support_failure | 16 | -0.16 | 0.18 | 0.82 | 0.6597 | labeled_slice_under_iid_sampling_contract |
| support_failure | 32 | -0.16 | 0.28 | 0.7 | 0.474 | labeled_slice_under_iid_sampling_contract |
| support_failure | 64 | -0.16 | 0.58 | 0.42 | 0.3382 | labeled_slice_under_iid_sampling_contract |
| support_failure | 128 | -0.16 | 0.72 | 0.28 | 0.241 | labeled_slice_under_iid_sampling_contract |
| support_failure | 256 | -0.16 | 0.96 | 0.04 | 0.1708 | labeled_slice_under_iid_sampling_contract |
| support_failure | 512 | -0.16 | 1.0 | 0.0 | 0.121 | labeled_slice_under_iid_sampling_contract |
| montage_transport_shift | 0 | 0.03 | 0.5 | 1.0 | None | not_identified_R1 |
| montage_transport_shift | 1 | 0.03 | 0.18 | 0.62 | 0.0 | labeled_slice_under_iid_sampling_contract |
| montage_transport_shift | 2 | 0.03 | 0.06 | 0.9 | 1.3087 | labeled_slice_under_iid_sampling_contract |
| montage_transport_shift | 4 | 0.03 | 0.12 | 0.84 | 1.1586 | labeled_slice_under_iid_sampling_contract |
| montage_transport_shift | 8 | 0.03 | 0.02 | 0.96 | 0.9175 | labeled_slice_under_iid_sampling_contract |
| montage_transport_shift | 16 | 0.03 | 0.08 | 0.9 | 0.6681 | labeled_slice_under_iid_sampling_contract |
| montage_transport_shift | 32 | 0.03 | 0.08 | 0.92 | 0.4759 | labeled_slice_under_iid_sampling_contract |
| montage_transport_shift | 64 | 0.03 | 0.08 | 0.92 | 0.3412 | labeled_slice_under_iid_sampling_contract |
| montage_transport_shift | 128 | 0.03 | 0.08 | 0.9 | 0.2425 | labeled_slice_under_iid_sampling_contract |
| montage_transport_shift | 256 | 0.03 | 0.1 | 0.9 | 0.1718 | labeled_slice_under_iid_sampling_contract |
| montage_transport_shift | 512 | 0.03 | 0.22 | 0.78 | 0.1215 | labeled_slice_under_iid_sampling_contract |

> k>0: an iid k-label slice estimates target risk under an iid sampling contract with a finite-sample CI; NOT full target risk without that contract.
