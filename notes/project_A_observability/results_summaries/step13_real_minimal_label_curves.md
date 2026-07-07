# Step 13 — real minimal-label curves

Scope: real minimal-label curves (R2 labeled slice); not SOTA. k=0 = R1 non-identifiable; k>0 = R2 labeled slice under an iid sampling contract compared to the oracle full-target sign; not full-target identification. Oracle labels are used only here (R2/evaluation).

- runs with per-trial oracle predictions: **54** · repeats **200**
- best k for harm-sign acc ≥ 0.8: **None** · ≥ 0.9: **None**
- k=0 status: **not_identified_R1**

| k | harm_sign_acc | decisive_rate | abstention | mean_ci_width | status |
|---:|---:|---:|---:|---:|---|
| 0 | 0.5 | 0.0 | 1.0 | None | not_identified_R1 |
| 1 | 0.1237 | 0.1992 | 0.8008 | 0.0 | r2_labeled_slice_under_iid_sampling_contract |
| 2 | 0.0184 | 0.0262 | 0.9738 | 0.676 | r2_labeled_slice_under_iid_sampling_contract |
| 4 | 0.0091 | 0.0127 | 0.9873 | 0.6322 | r2_labeled_slice_under_iid_sampling_contract |
| 8 | 0.0473 | 0.0582 | 0.9418 | 0.5186 | r2_labeled_slice_under_iid_sampling_contract |
| 16 | 0.056 | 0.0622 | 0.9378 | 0.399 | r2_labeled_slice_under_iid_sampling_contract |
| 32 | 0.1204 | 0.1285 | 0.8715 | 0.2915 | r2_labeled_slice_under_iid_sampling_contract |
| 64 | 0.1572 | 0.1631 | 0.8369 | 0.2094 | r2_labeled_slice_under_iid_sampling_contract |
| 128 | 0.2308 | 0.2341 | 0.7659 | 0.149 | r2_labeled_slice_under_iid_sampling_contract |
| 256 | 0.3175 | 0.318 | 0.682 | 0.1055 | r2_labeled_slice_under_iid_sampling_contract |

> k>0 estimates a labeled-slice gain under an iid sampling contract and compares it to the oracle full-target sign; NOT full-target identification without that contract.
