# Step 10 BNCI2014_004 audited mini-grid summary

Scope: interface + audited-report validation — **not a SOTA claim**; target metrics are oracle/evaluation-only.

- runs: **27**  ·  ok: **27**  ·  skipped: **0**
- all forbidden-violations empty: **True**
- all target metrics oracle-only: **True**
- all target metrics identifiable=null: **True**
- all prior claims compliant: **True**  ·  no unknown estimands: **True**
- mean strict-DG bAcc: **0.6282**  ·  mean offline-TTA gain: **-0.032**  ·  offline-TTA harm-rate: **0.8519**
- n_classes: **2**  ·  chance bAcc: **0.5**  ·  mean strict-DG excess-norm: **0.2563**  ·  mean offline-TTA gain-norm: **-0.0639**
- expected runs: **27**  ·  missing cells: **0**

| target | seed | status | strict bAcc | offline gain | online bAcc | claims | violations | prior claim | metrics id=null |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 0 | ✅ ok | 0.658 | -0.046 | 0.651 | 8 | 0 | rejected_conclusion_false | True |
| 1 | 1 | ✅ ok | 0.650 | -0.051 | 0.656 | 8 | 0 | rejected_conclusion_false | True |
| 1 | 2 | ✅ ok | 0.667 | -0.090 | 0.674 | 8 | 0 | rejected_conclusion_false | True |
| 2 | 0 | ✅ ok | 0.507 | -0.006 | 0.526 | 8 | 0 | rejected_conclusion_false | True |
| 2 | 1 | ✅ ok | 0.544 | -0.007 | 0.538 | 8 | 0 | rejected_conclusion_false | True |
| 2 | 2 | ✅ ok | 0.522 | 0.000 | 0.522 | 8 | 0 | rejected_conclusion_false | True |
| 3 | 0 | ✅ ok | 0.560 | 0.008 | 0.544 | 8 | 0 | rejected_conclusion_false | True |
| 3 | 1 | ✅ ok | 0.556 | -0.024 | 0.551 | 8 | 0 | rejected_conclusion_false | True |
| 3 | 2 | ✅ ok | 0.560 | -0.019 | 0.554 | 8 | 0 | rejected_conclusion_false | True |
| 4 | 0 | ✅ ok | 0.739 | -0.012 | 0.739 | 8 | 0 | rejected_conclusion_false | True |
| 4 | 1 | ✅ ok | 0.769 | 0.007 | 0.758 | 8 | 0 | rejected_conclusion_false | True |
| 4 | 2 | ✅ ok | 0.770 | 0.008 | 0.764 | 8 | 0 | rejected_conclusion_false | True |
| 5 | 0 | ✅ ok | 0.580 | -0.014 | 0.564 | 8 | 0 | rejected_conclusion_false | True |
| 5 | 1 | ✅ ok | 0.612 | -0.049 | 0.612 | 8 | 0 | rejected_conclusion_false | True |
| 5 | 2 | ✅ ok | 0.557 | -0.055 | 0.561 | 8 | 0 | rejected_conclusion_false | True |
| 6 | 0 | ✅ ok | 0.625 | -0.007 | 0.611 | 8 | 0 | rejected_conclusion_false | True |
| 6 | 1 | ✅ ok | 0.658 | -0.067 | 0.657 | 8 | 0 | rejected_conclusion_false | True |
| 6 | 2 | ✅ ok | 0.628 | -0.064 | 0.614 | 8 | 0 | rejected_conclusion_false | True |
| 7 | 0 | ✅ ok | 0.582 | -0.022 | 0.528 | 8 | 0 | rejected_conclusion_false | True |
| 7 | 1 | ✅ ok | 0.613 | -0.032 | 0.614 | 8 | 0 | rejected_conclusion_false | True |
| 7 | 2 | ✅ ok | 0.601 | -0.018 | 0.606 | 8 | 0 | rejected_conclusion_false | True |
| 8 | 0 | ✅ ok | 0.703 | -0.033 | 0.688 | 8 | 0 | rejected_conclusion_false | True |
| 8 | 1 | ✅ ok | 0.728 | -0.099 | 0.711 | 8 | 0 | rejected_conclusion_false | True |
| 8 | 2 | ✅ ok | 0.691 | -0.064 | 0.661 | 8 | 0 | rejected_conclusion_false | True |
| 9 | 0 | ✅ ok | 0.635 | -0.040 | 0.625 | 8 | 0 | rejected_conclusion_false | True |
| 9 | 1 | ✅ ok | 0.622 | -0.035 | 0.619 | 8 | 0 | rejected_conclusion_false | True |
| 9 | 2 | ✅ ok | 0.625 | -0.032 | 0.629 | 8 | 0 | rejected_conclusion_false | True |

## Per-target (mean ± std over seeds)

| target | n_ok | strict-DG bAcc | offline-TTA gain | harm-rate |
|---|---:|---|---|---:|
| 1 | 3 | 0.658 ± 0.007 [0.650,0.667] | -0.062 ± 0.020 | 1.000 |
| 2 | 3 | 0.524 ± 0.015 [0.507,0.544] | -0.004 ± 0.003 | 0.667 |
| 3 | 3 | 0.558 ± 0.002 [0.556,0.560] | -0.012 ± 0.014 | 0.667 |
| 4 | 3 | 0.759 ± 0.014 [0.739,0.770] | 0.001 ± 0.009 | 0.333 |
| 5 | 3 | 0.583 ± 0.023 [0.557,0.612] | -0.039 ± 0.018 | 1.000 |
| 6 | 3 | 0.637 ± 0.015 [0.625,0.658] | -0.046 ± 0.028 | 1.000 |
| 7 | 3 | 0.599 ± 0.013 [0.582,0.613] | -0.024 ± 0.006 | 1.000 |
| 8 | 3 | 0.707 ± 0.015 [0.691,0.728] | -0.065 ± 0.027 | 1.000 |
| 9 | 3 | 0.627 ± 0.005 [0.622,0.635] | -0.036 ± 0.004 | 1.000 |

> These are evaluation-only target metrics, not R0/R1 identifiable target risk or gain.
