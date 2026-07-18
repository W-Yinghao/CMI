# C86R2 Overall Report

## Result

C86R2 resolved the adult-only untouched multi-cohort blocker using only public
participant metadata, loader/BIDS identities, licenses, and catalog metadata.
It did not open EEG, target labels, candidate outputs, or any active-policy
result.

The final primary adult confirmation set is:

```text
Brandl2020_CANONICAL_ADULT_V1:
  16 targets

OpenNeuro_ds007221_HYBRID_ADULT_V1:
  37 targets, sub-37 through sub-73

total:
  2 cohorts / 53 target subjects
```

Yang remains age-mixed because no valid subject-level adult mapping exists.
Kumar remains age uncertain because its public age field is invalid and the
paper does not prove an all-adult cohort. Neither historical role was rewritten.

The public search replay covers the complete installed 53-row MOABB imagery
catalog, 824 EEGDash rows, and 760 NEMAR rows. The external task/name screen
produced 149 catalog candidates, 79 native datasets after mirror
deduplication, and 82 task-interface decisions after splitting `ds007221`.
Every passing interface was retained.

The V3 common field has 53 target subjects, 424 contexts, 34,344
candidate-context slices, 648 shared candidate units, and 1,296 unit-cohort
artifacts. These are metadata planning quantities, not execution results.

## Validation

```text
focused C86P/C86R/C86R2: 51 passed
C65 cumulative: 1,103 passed / 1 skipped / 12 deselected
C23 cumulative: 1,514 passed / 1 skipped / 12 deselected
full OACI: 2,489 passed / 1 skipped / 12 deselected
accepted stderr: 0 bytes for every suite
```

## Identities

```text
C86R2 protocol commit:
  0da4ec39f26ac4bc0d89035e9ad951f452217f05

C86R2 protocol SHA-256:
  2e88e2fef7500b12ca8b3c5b19e6aab06df5a7f388781855b73793a1fe75df92

C86R2 implementation/count-reconciliation commit:
  ccc1807479a2f3b3273991f6b3ca0ab5514184f1

accepted regression evidence commit:
  b6c331d79ec9fae38dbc6c9379c19ef9ecac72a4

effective program V3 SHA-256:
  c6b7e490e0f78f74f820428cee138782caff1dc0033422723593a7d8e3c5f77e
```

## Gate

```text
C86_ADULT_UNTOUCHED_MULTI_COHORT_ELIGIBILITY_RESOLVED_READY_FOR_C86LP_PROTOCOL_REVIEW
```

No C86LP, C86L, C86D, C86C/F, C86H, C87, real-data, active-acquisition, or
manuscript authorization is created.
