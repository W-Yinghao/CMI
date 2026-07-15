# C84FR1 Protocol Timing Audit

C84F job `896185` loaded all 118 registered target EEG arrays after the atomic
1,944-unit model-field freeze, then failed before constructing or writing the
target trial registry. The failure was a Python dictionary-ordering `TypeError`.

At the stop:

```text
target-y accesses:             0
construction/evaluation labels: 0
same-label oracle:             0
selector scores:               0
scientific statistics:         0
target registry rows frozen:   0
target artifacts frozen:       0
candidate-context slices:      0
```

This repair is therefore designed after engineering target-X access but before
any target label or scientific outcome. It is prospective to the replacement
registry serialization, target instrumentation, canary replay, and complete
field manifest.

The repair changes one implementation freedom only: raw-file dictionaries are
ordered by the explicit tuple `(path, bytes, sha256)`. Dictionary insertion
order is semantically irrelevant. The frozen model field, candidate IDs,
scientific protocols, numerical tolerances, target partitions, and final field
gate do not change.

Job `896185`, its consumed authorization, failed root, logs, manifests, and
failure evidence remain immutable. The new target-only runtime may reload
target X only after a new execution lock and fresh direct PI authorization. It
cannot import or invoke training and cannot access target labels or C84S.

