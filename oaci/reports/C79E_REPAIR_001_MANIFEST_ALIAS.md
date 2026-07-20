# C79E Repair 001 - Manifest Compatibility Alias

## Failure

Seed-4 target-4 OACI/ERM job `893354` failed before optimizer execution because
the locked historical runtime requested `c79p_tables/full_unit_manifest.csv`,
while C79P committed the identical registry as
`expected_seed4_field_manifest.csv`.

```text
source EEG rows loaded:          3,456
target rows loaded:                  0
target labels read:                  0
optimizer steps:                     0
checkpoints created:                 0
model-specific outcomes read:        0
dependent jobs run:                  0
```

Jobs `893355` and `893356` were cancelled without execution after their
dependency became unsatisfiable.

## Repair

The repair adds only the historical ABI filename as a byte-identical compact
manifest alias:

```text
source: oaci/reports/c79p_tables/expected_seed4_field_manifest.csv
alias:  oaci/reports/c79p_tables/full_unit_manifest.csv
SHA-256 for both:
dcf704cd83aed80ae5f6ef5d360da43c5d45b1ec3453c66f18118d0108a27378
rows including header: 1,459
```

No implementation, scientific registry, expected unit identity, estimand,
threshold, model, null, multiplicity family, authorization scope, or execution
lock changes. The failed job and cancelled dependencies remain in the external
and compact ledgers.
