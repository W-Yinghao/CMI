# W1 Split/Metric Audit

- status: PASS
- scope: CPU-only P6.1 split/metric impact audit
- split_function: `h2cmi.data.real_eeg.contiguous_split`
- main_h2cmi_w1_runner: `h2cmi/run_w1_p0.py`
- legacy_w1a_runner: `h2cmi/run_w1_mi.py`
- spdim_p6_runner: `h2cmi/run_spdim_w1_seed0.py`
- scorer: `sklearn.metrics.balanced_accuracy_score`
- no Slurm jobs launched; no GPU work.

## Aggregate Single-Class Evaluation

| dataset | subjects | single-class eval subjects | fraction |
|---|---:|---:|---:|
| BNCI2014_001 | 9 | 0 | 0.000000 |
| Cho2017 | 52 | 52 | 1.000000 |
| Lee2019_MI | 54 | 0 | 0.000000 |
| all W1 | 115 | 52 | 0.452174 |

## Affected Rows

| artifact | affected rows | note |
|---|---:|---|
| corrected REVIEW_P0 W1 raw | 1560 | Cho2017 rows including `__decomposition__` |
| corrected REVIEW_P0 W1 metric rows | 1404 | Cho2017 rows excluding `__decomposition__` |
| legacy W1-A raw | 312 | Cho2017 W1-A method rows |
| SPDIM P6 seed-0 | 208 | Cho2017 rows = 52 targets x 4 methods |

## Per-Target Split Composition

| dataset | target | adapt blocks | eval blocks | n adapt | class counts adapt | n eval | class counts eval | single-class eval | scorer-defined | acc==bAcc reason |
|---|---:|---|---|---:|---:|---:|---:|---|---|---|
| BNCI2014_001 | 1 | `session=0/run=0; session=0/run=1; session=0/run=2` | `session=0/run=3; session=0/run=4; session=0/run=5` | 72 | `[36, 36]` | 72 | `[36, 36]` | `False` | `True` | class_balance |
| BNCI2014_001 | 2 | `session=0/run=0; session=0/run=1; session=0/run=2` | `session=0/run=3; session=0/run=4; session=0/run=5` | 72 | `[36, 36]` | 72 | `[36, 36]` | `False` | `True` | class_balance |
| BNCI2014_001 | 3 | `session=0/run=0; session=0/run=1; session=0/run=2` | `session=0/run=3; session=0/run=4; session=0/run=5` | 72 | `[36, 36]` | 72 | `[36, 36]` | `False` | `True` | class_balance |
| BNCI2014_001 | 4 | `session=0/run=0; session=0/run=1; session=0/run=2` | `session=0/run=3; session=0/run=4; session=0/run=5` | 72 | `[36, 36]` | 72 | `[36, 36]` | `False` | `True` | class_balance |
| BNCI2014_001 | 5 | `session=0/run=0; session=0/run=1; session=0/run=2` | `session=0/run=3; session=0/run=4; session=0/run=5` | 72 | `[36, 36]` | 72 | `[36, 36]` | `False` | `True` | class_balance |
| BNCI2014_001 | 6 | `session=0/run=0; session=0/run=1; session=0/run=2` | `session=0/run=3; session=0/run=4; session=0/run=5` | 72 | `[36, 36]` | 72 | `[36, 36]` | `False` | `True` | class_balance |
| BNCI2014_001 | 7 | `session=0/run=0; session=0/run=1; session=0/run=2` | `session=0/run=3; session=0/run=4; session=0/run=5` | 72 | `[36, 36]` | 72 | `[36, 36]` | `False` | `True` | class_balance |
| BNCI2014_001 | 8 | `session=0/run=0; session=0/run=1; session=0/run=2` | `session=0/run=3; session=0/run=4; session=0/run=5` | 72 | `[36, 36]` | 72 | `[36, 36]` | `False` | `True` | class_balance |
| BNCI2014_001 | 9 | `session=0/run=0; session=0/run=1; session=0/run=2` | `session=0/run=3; session=0/run=4; session=0/run=5` | 72 | `[36, 36]` | 72 | `[36, 36]` | `False` | `True` | class_balance |
| Cho2017 | 1 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 2 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 3 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 4 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 5 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 6 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 7 | `session=0/run=0/trial_offset=0:120` | `session=0/run=0/trial_offset=120:240` | 120 | `[120, 0]` | 120 | `[0, 120]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 8 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 9 | `session=0/run=0/trial_offset=0:120` | `session=0/run=0/trial_offset=120:240` | 120 | `[120, 0]` | 120 | `[0, 120]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 10 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 11 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 12 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 13 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 14 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 15 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 16 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 17 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 18 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 19 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 20 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 21 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 22 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 23 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 24 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 25 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 26 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 27 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 28 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 29 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 30 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 31 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 32 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 33 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 34 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 35 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 36 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 37 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 38 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 39 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 40 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 41 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 42 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 43 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 44 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 45 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 46 | `session=0/run=0/trial_offset=0:120` | `session=0/run=0/trial_offset=120:240` | 120 | `[120, 0]` | 120 | `[0, 120]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 47 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 48 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 49 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 50 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 51 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Cho2017 | 52 | `session=0/run=0/trial_offset=0:100` | `session=0/run=0/trial_offset=100:200` | 100 | `[100, 0]` | 100 | `[0, 100]` | `True` | `True` | single_class_scorer_ignores_absent_class |
| Lee2019_MI | 1 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[26, 24]` | 50 | `[24, 26]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 2 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 3 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 4 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[26, 24]` | 50 | `[24, 26]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 5 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[21, 29]` | 50 | `[29, 21]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 6 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 7 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 8 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 9 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 10 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 11 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 12 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 13 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[26, 24]` | 50 | `[24, 26]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 14 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 15 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 16 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[27, 23]` | 50 | `[23, 27]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 17 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 18 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 19 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 20 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 21 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 22 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 23 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[26, 24]` | 50 | `[24, 26]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 24 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 25 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 26 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 27 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[28, 22]` | 50 | `[22, 28]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 28 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 29 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 30 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 31 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 32 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 33 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 34 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[20, 30]` | 50 | `[30, 20]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 35 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 36 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 37 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 38 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 39 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 40 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 41 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 42 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 43 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 44 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 45 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 46 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 47 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 48 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 49 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 50 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 51 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[29, 21]` | 50 | `[21, 29]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 52 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[21, 29]` | 50 | `[29, 21]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 53 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[24, 26]` | 50 | `[26, 24]` | `False` | `True` | not_by_construction |
| Lee2019_MI | 54 | `session=0/run=0/trial_offset=0:50` | `session=0/run=0/trial_offset=50:100` | 50 | `[21, 29]` | 50 | `[29, 21]` | `False` | `True` | not_by_construction |

## Interpretation

- BNCI2014_001 has balanced evaluation blocks for all targets, so ordinary accuracy equals balanced accuracy by class balance for any prediction table.
- Cho2017 has single-class evaluation blocks for all targets. Under the project scorer, balanced accuracy is numerically defined but equals ordinary accuracy because the absent class is ignored.
- Lee2019_MI has both classes in evaluation for all targets, but most target blocks are not exactly balanced, so acc and bAcc are not equal by construction.
- The SPDIM P6 dry-run split hashes/counts match the recomputed split counts.

## Red Team Review

- CPU-only audit: no Slurm submission and no GPU training.
- Both main H2CMI and SPDIM P6 use the same contiguous_split function.
- Cho2017 single-class evaluation is confirmed for 52/52 target subjects.
- The project scorer returns a numeric score on one-class y_true by ignoring absent classes.
- No seeds 1/2 or full SPDIM expansion are approved by this audit.
