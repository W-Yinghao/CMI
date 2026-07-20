# C79 Seed-4 Label-View Red Team

```text
primary targets:                    8 / 8
construction rows:                 2235
evaluation rows:                   2373
union rows:                        4608
construction/evaluation overlap:  0
locked split hashes replayed:      8 / 8
target-label fields in unlabeled:  0
target-4 label view created:       false
same-label oracle created:         false
oracle descriptor visible:        false
trial ID used as predictor:        false
row order used as predictor:       false
```

The construction/evaluation descriptors exactly replay the C78S trial split
hashes and partition all 576 target trials per primary target without overlap.
The primary route contains no target-4 label view and no same-label-oracle
descriptor.

Gate: `C79E_LABEL_VIEW_PROVISIONING_RED_TEAM_PASSED`.
