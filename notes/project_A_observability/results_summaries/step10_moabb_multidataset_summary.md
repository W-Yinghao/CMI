# Step 10 — multi-dataset audited summary (chance-normalized)

Scope: multi-dataset audited expansion; not SOTA. Target metrics are oracle/evaluation-only.

- datasets: **BNCI2014_001, BNCI2014_004, BNCI2015_001**  ·  n_datasets: **3**
- mixed n_classes: **True**  ·  raw-bAcc overall suppressed: **True**
- all datasets valid: **True**  ·  any missing cells: **False**  ·  any forbidden violations: **False**
- all target metrics identifiable=null: **True**

## Overall (chance-normalized — the cross-dataset numbers)

- mean strict-DG excess-norm: **0.2246**
- mean offline-TTA gain-norm: **-0.0654**
- offline-TTA harm-rate: **0.8333**  (over **54** ok runs)

## Per dataset

| dataset | K | n_ok | n_skip | valid | id=null | raw bAcc (within) | strict excess-norm | offline gain-norm | harm-rate | missing |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| BNCI2014_001 | 4 | 27 | 0 | True | True | 0.395 | 0.193 | -0.067 | 0.815 | 0 |
| BNCI2014_004 | 2 | 27 | 0 | True | True | 0.628 | 0.256 | -0.064 | 0.852 | 0 |
| BNCI2015_001 | None | 0 | 36 | True | False | — | — | — | — | 0 |

> Cross-dataset aggregates use chance-normalized excess ((bAcc-1/K)/(1-1/K)); raw bAcc is never pooled across datasets with different class counts. Target metrics remain oracle/evaluation-only. Not a SOTA claim.
