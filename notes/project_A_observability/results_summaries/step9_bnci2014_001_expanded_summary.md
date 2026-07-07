# Step 9 BNCI2014_001 audited mini-grid summary

Scope: interface + audited-report validation — **not a SOTA claim**; target metrics are oracle/evaluation-only.

- runs: **27**  ·  ok: **27**  ·  skipped: **0**
- all forbidden-violations empty: **True**
- all target metrics oracle-only: **True**
- all target metrics identifiable=null: **True**
- all prior claims compliant: **True**  ·  no unknown estimands: **True**
- mean strict-DG bAcc: **0.3946**  ·  mean offline-TTA gain: **-0.0502**  ·  offline-TTA harm-rate: **0.8148**
- expected runs: **27**  ·  missing cells: **0**

| target | seed | status | strict bAcc | offline gain | online bAcc | claims | violations | prior claim | metrics id=null |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 0 | ✅ ok | 0.545 | -0.056 | 0.523 | 8 | 0 | rejected_conclusion_false | True |
| 1 | 1 | ✅ ok | 0.552 | -0.200 | 0.505 | 8 | 0 | rejected_conclusion_false | True |
| 1 | 2 | ✅ ok | 0.571 | -0.099 | 0.509 | 8 | 0 | rejected_conclusion_false | True |
| 2 | 0 | ✅ ok | 0.285 | -0.043 | 0.271 | 8 | 0 | rejected_conclusion_false | True |
| 2 | 1 | ✅ ok | 0.276 | -0.033 | 0.269 | 8 | 0 | rejected_conclusion_false | True |
| 2 | 2 | ✅ ok | 0.286 | -0.035 | 0.285 | 8 | 0 | rejected_conclusion_false | True |
| 3 | 0 | ✅ ok | 0.509 | -0.111 | 0.510 | 8 | 0 | rejected_conclusion_false | True |
| 3 | 1 | ✅ ok | 0.477 | -0.009 | 0.472 | 8 | 0 | rejected_conclusion_false | True |
| 3 | 2 | ✅ ok | 0.464 | -0.102 | 0.396 | 8 | 0 | rejected_conclusion_false | True |
| 4 | 0 | ✅ ok | 0.351 | -0.054 | 0.335 | 8 | 0 | rejected_conclusion_false | True |
| 4 | 1 | ✅ ok | 0.368 | -0.050 | 0.365 | 8 | 0 | rejected_conclusion_false | True |
| 4 | 2 | ✅ ok | 0.359 | -0.026 | 0.342 | 8 | 0 | rejected_conclusion_false | True |
| 5 | 0 | ✅ ok | 0.266 | -0.003 | 0.250 | 8 | 0 | rejected_conclusion_false | True |
| 5 | 1 | ✅ ok | 0.260 | 0.003 | 0.253 | 8 | 0 | rejected_conclusion_false | True |
| 5 | 2 | ✅ ok | 0.252 | 0.005 | 0.252 | 8 | 0 | rejected_conclusion_false | True |
| 6 | 0 | ✅ ok | 0.358 | 0.014 | 0.330 | 8 | 0 | rejected_conclusion_false | True |
| 6 | 1 | ✅ ok | 0.368 | 0.019 | 0.365 | 8 | 0 | rejected_conclusion_false | True |
| 6 | 2 | ✅ ok | 0.330 | -0.002 | 0.293 | 8 | 0 | rejected_conclusion_false | True |
| 7 | 0 | ✅ ok | 0.307 | -0.031 | 0.295 | 8 | 0 | rejected_conclusion_false | True |
| 7 | 1 | ✅ ok | 0.278 | -0.007 | 0.280 | 8 | 0 | rejected_conclusion_false | True |
| 7 | 2 | ✅ ok | 0.269 | -0.031 | 0.285 | 8 | 0 | rejected_conclusion_false | True |
| 8 | 0 | ✅ ok | 0.576 | -0.038 | 0.554 | 8 | 0 | rejected_conclusion_false | True |
| 8 | 1 | ✅ ok | 0.517 | -0.205 | 0.484 | 8 | 0 | rejected_conclusion_false | True |
| 8 | 2 | ✅ ok | 0.531 | -0.181 | 0.469 | 8 | 0 | rejected_conclusion_false | True |
| 9 | 0 | ✅ ok | 0.467 | -0.043 | 0.470 | 8 | 0 | rejected_conclusion_false | True |
| 9 | 1 | ✅ ok | 0.420 | -0.052 | 0.399 | 8 | 0 | rejected_conclusion_false | True |
| 9 | 2 | ✅ ok | 0.411 | 0.014 | 0.373 | 8 | 0 | rejected_conclusion_false | True |

## Per-target (mean ± std over seeds)

| target | n_ok | strict-DG bAcc | offline-TTA gain | harm-rate |
|---|---:|---|---|---:|
| 1 | 3 | 0.556 ± 0.011 [0.545,0.571] | -0.118 ± 0.060 | 1.000 |
| 2 | 3 | 0.282 ± 0.005 [0.276,0.286] | -0.037 ± 0.005 | 1.000 |
| 3 | 3 | 0.483 ± 0.019 [0.464,0.509] | -0.074 ± 0.046 | 1.000 |
| 4 | 3 | 0.359 ± 0.007 [0.351,0.368] | -0.043 ± 0.012 | 1.000 |
| 5 | 3 | 0.259 ± 0.006 [0.252,0.266] | 0.002 ± 0.004 | 0.333 |
| 6 | 3 | 0.352 ± 0.016 [0.330,0.368] | 0.010 ± 0.009 | 0.333 |
| 7 | 3 | 0.285 ± 0.016 [0.269,0.307] | -0.023 ± 0.011 | 1.000 |
| 8 | 3 | 0.542 ± 0.025 [0.517,0.576] | -0.141 ± 0.073 | 1.000 |
| 9 | 3 | 0.433 ± 0.024 [0.411,0.467] | -0.027 ± 0.029 | 0.667 |

> These are evaluation-only target metrics, not R0/R1 identifiable target risk or gain.
