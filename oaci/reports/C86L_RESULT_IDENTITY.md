# C86L — Result Identity (authorized build under 授权 C86L)

**Gate reached**

```text
C86L_C84_CONSTRUCTION_POOL_TRIAL_CONTRIBUTION_FIELD_FROZEN_C86D_PROTOCOL_REVIEW_REQUIRED
```

Executed under a direct `授权 C86L` (job 901965, cpu-high, build 10.5 s). The
output field lives in the external results area (not committed); this is its
durable identity. C86D / C86H / C87 / manuscript remain NOT authorized.

## Output (three separate physical roots)

```text
root: /projects/EEG-foundation-model/yinghao/oaci-c86l-development-field-v1
  acquisition_unlabeled_pool/    944 context .npz  (probabilities only, NO labels)
  query_contribution_store/      944 context .npz  (label-derived nll/correct/confidence/conf_bin/signed_calibration + true_label)
  acquisition_label_oracle/      labels.csv        (4773 sealed construction labels)
  C86L_RESULT_MANIFEST.json  sha256 fea856165aa36a9dfd846618e2a3fc929f2f62f3fc65940d3b08b83b8cb50ce3
  C86L_CONTEXT_INDEX.json    (944 contexts)
```

## Verified invariants (fail-closed; all passed)

```text
contexts                      944   = 118 subjects x 8 (panel x seed x level)
candidates / context           81   = 1 ERM + 40 OACI + 40 SRC (canonical: ERM:0, OACI:1..40, SRC:1..40)
construction trials         4,773
context-trials             38,184   = 4,773 x 8
contributions           3,092,904   = 38,184 x 81
binary probability scalars 6,185,808
construction ⟂ evaluation overlap = 0
```

## Bound real inputs

```text
predictions   : oaci-c84-full-field-target-replay-v2/lock_f0c369ee.../complete_target_unlabeled_v2 (1944 npz)
construction  : oaci-c84s-analysis-v3/stage_a_labels/target_construction_label_view/labels.csv
                sha256 fdf36052d36ad9546cda06cbc567f68cdcced7ad08fd1311ab949471218b3134 (4773 rows, 0/1)
join key      : target_trial_id
held C85U outcome : oaci-c85u-candidate-utility-v2 (identity-bound only, NOT opened)
```

Isolation is real directory separation of pool / oracle / contribution; the
client-visible pool contains no labels. Recomputed nll/correct match the stored
contribution rows. Endpoint carried forward: `target_near_opt_prob = P(target
8-context mean regret <= eps)`.
