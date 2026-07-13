# C81E Authorization and Preflight

## Gate

```text
C81E_AUTHORIZATION_PROTOCOL_LOCK_AND_VIEW_PREFLIGHT_PASSED
```

The PI directly stated `现在我明确授权C81E` in the current execution
conversation. Under policy commit `3d9dd76`, this is sufficient and requires no
magic token or repeated hash recital. The server record binds it to the unique
current C81 protocol, analysis lock, and field/view manifest digest.

## Bound Objects

```text
base HEAD and origin/oaci: 89c6afb56bf0a386200d5ce4e54c0d14153bcde8
protocol commit:           16a0d2eba4715a1cec78da6a79a182fd416a6629
protocol SHA-256:          cbdb42f54956b685c27a1718c37d7c56c513084817a5c69fb29f06bfb67ad3ee
implementation commit:     d17ffa62a63b929d36d03f74e4ce79794cd9601b
analysis lock commit:      541651c2ee3343c12d374a7322c91181a860a2c9
analysis lock SHA-256:     b383707f58063c10f719194a995ab34094f6dcefe08c1e71837644db83dc94f1
manifest digest:           6180275dcef26bdda4ae4b291d1ef6dc83434462ecacee0350fa94ae9c6a7fef
```

Protocol, lock, method registry, both implementation blobs, all eight locked
registry tables, and all 11 field/view manifest objects replay exactly. No
operative blob changed after `541651c`.

## Registry and Scope

```text
method entries:                    34 / 34
zero-label primary representatives: 6 / 6
strict-source representative:      S1 source-validation balanced accuracy
selection methods in adapter:      19 / 19 unconditional
target4 primary rows before run:     0
same-label-oracle accesses:          0
real C81 baseline statistics:        0
evaluation-label reads:              0
training / forward / GPU:             0 / 0 / 0
```

The Q0 B=1 and FULL comparators are immutable C80 artifacts. B5 oracle-best is
a denominator only. Score direction, feature layer, temperature, prior,
threshold, tie rule, target-cluster inference, paired-seed treatment, max-T,
noninferiority, LOTO stability, and C81 taxonomy all match the lock.

## Regression Count Reconciliation

The reported deltas are fully explained:

```text
C65: 369 + 43 C81P tests = 412
full: 1704 + 43 C81P tests = 1747
C23: 776 + 4 restored C34S nodes + 43 C81P tests = 823
```

The old C80E C23 selector used `test_cNN_*.py`-style globs. That selector did
not match `test_c34s_artifact_hygiene.py` because the milestone suffix precedes
the underscore. The full suite already contained its four tests. C81P used a
numeric milestone file selector that included the file, so the C23 suite gained
those four existing nodes in addition to the 43 new C81P nodes. Their exact
pytest node IDs are recorded in `c81e_regression_nodeid_delta.csv`.

This is a suite-selector correction, not a scientific implementation change.

## Physical Execution Order

The locked implementation is unchanged. To satisfy the committed freeze
barrier, execution will call the already locked `_selection_stage` first. That
stage can read source and target-unlabeled objects but has no evaluation-label
descriptor. Its content-addressed manifest will be red-teamed, recorded, and
committed before the original locked `run-real` entrypoint is allowed to
continue. `run-real` must replay the same selection hash before opening the
evaluation view.

No external C81 result directory or selection payload existed at this preflight
boundary.
