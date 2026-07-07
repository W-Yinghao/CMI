# Step 13/14 — real minimal-label curves (coverage-decomposed)

Scope: real minimal-label curves (R2 labeled slice, coverage-decomposed); not SOTA. k=0 = R1 non-identifiable (no estimator, accuracy NULL); k>0 = R2 labeled slice under an iid sampling contract vs the oracle full-target sign; not full-target identification. Oracle labels used only here (R2/evaluation).

- runs with per-trial oracle predictions: **54** · repeats **200**
- best k (unconditional ≥0.8): **None** · (conditional ≥0.8): **8**
- oracle sign distribution: **{'harm': 46, 'benefit_or_no_harm': 8}** · always-predict-harm baseline **0.8519** (oracle, not R1)
- k=0 status: **not_identified_R1** (no estimator; accuracy NULL, not 0.5)

| k | coverage (decisive) | uncond correct | conditional acc (decisive) | abstention | ci_width |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | None | None | 1.0 | None |
| 1 | 0.1992 | 0.1237 | 0.6211 | 0.8008 | 0.0 |
| 2 | 0.0262 | 0.0184 | 0.7032 | 0.9738 | 0.676 |
| 4 | 0.0127 | 0.0091 | 0.7153 | 0.9873 | 0.6322 |
| 8 | 0.0582 | 0.0473 | 0.8124 | 0.9418 | 0.5186 |
| 16 | 0.0622 | 0.056 | 0.9003 | 0.9378 | 0.399 |
| 32 | 0.1285 | 0.1204 | 0.9366 | 0.8715 | 0.2915 |
| 64 | 0.1631 | 0.1572 | 0.9637 | 0.8369 | 0.2094 |
| 128 | 0.2341 | 0.2308 | 0.9862 | 0.7659 | 0.149 |
| 256 | 0.318 | 0.3175 | 0.9985 | 0.682 | 0.1055 |

> k>0 estimates a labeled-slice gain under an iid sampling contract and compares it to the oracle full-target sign; NOT full-target identification without that contract.
> Coverage–confidence tradeoff: when the slice is DECISIVE it may be accurate (conditional accuracy), but COVERAGE (decisive rate) stays low; the burden is coverage.
