# C85E Frozen-Field Decision-Theory Bridge Overall Report

## Final Disposition

```text
C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_COMPLETE_C86_PROTOCOL_REVIEW_REQUIRED
```

C85E consumed one fresh authorization and completed the read-only frozen-field
bridge. It published one atomic result bundle containing all 26 registered
tables and the restricted-policy/information-value synthesis. Every result row
is `POST_C84S_EXPLORATORY`.

C85E did not reopen labels, logits, EEG, or source arrays; rerun selectors, Q0,
or inference; change C84 gates or C85 theorem statuses; execute active
acquisition; authorize C86; or modify manuscript prose.

## Authoritative Identities

| Object | Identity |
|---|---|
| C85EP2 readiness HEAD | `39891d827985a164a867381246f8dcf6242c7801` |
| authorization/execution HEAD | `c65402db892fb8a58a0592ca5e81a0ee871a88ce` |
| C85E lock commit | `48e177c9914003202cc75cefb4a98832ea8250c3` |
| C85E lock SHA-256 | `a59062305b521973476e0d40236069eba7c9e149aeca9d3fe03c08a1ce106176` |
| authorization file SHA-256 | `a67a57266d6454ab9352b9c6db861cd2950f725c4a3c2cf49ffd64ea623a541c` |
| authorization binding SHA-256 | `80b8a8203f87f6186826eee7beb3f0343e0584607302bea0607d898d31996934` |
| consumed-receipt SHA-256 | `3ae9e9b518a8ef1acbd0dec213dc948700f4cbae8b46bacbdc8b6fe8ca7de03b` |
| result SHA-256 | `f2f09d123dd7767d41e6141ae5fe573bff8c15868aee04ae3be9a3d5c1b8d4db` |
| result-manifest SHA-256 | `75ffaa7212b01f6dfc4c69f86ddbe9d8f6c911ce7a107917b997a8c9fcf6c55e` |
| completion-receipt SHA-256 | `fe0ce0873734fcb709136f32834bd349cccda5710f9098e03297c01d2ab8433f` |
| lifecycle SHA-256 | `898596c9e33d6e1ec920ac49a052716613c8e5b8032e05e5a46836bce9df5e3c` |

External root:

```text
/projects/EEG-foundation-model/yinghao/oaci-c85e-frozen-field-v1/
  c85e-v1-a59062305b521973-e1f6b3976f5f4499
```

Attempt identities:

```text
authorization ID:
  e1f6b397-6f5f-4499-98d8-ad0f5cedd779

attempt ID:
  89bd3b598c8b46aaaeb924169c90e9de

input replay SHA-256:
  0e233f27a786c0269a53d9e73ce7f266a4bf7909975403023b23165e573b2d58
```

## Authorization And Execution

The direct statement `授权 C85E` was bound only to the current lock. The exact
authorization record was committed and pushed in `c65402db892fb8a58a0592ca5e81a0ee871a88ce`.
Preflight established a clean `oaci` branch, `HEAD == origin/oaci`, exact lock
and runtime identities, and absent output/consumption paths before the one-use
receipt was created.

Slurm job 900801 used the locked envelope:

```text
partition:  cpu-high
CPU:        32
RAM:        128 GiB
GPU:        0
wall limit: 2 hours
```

`scontrol` records `COMPLETED`, `ExitCode=0:0`, and 13 seconds elapsed. The
application stdout contains only the final gate and stderr is empty. The
result-bundle transaction lifecycle spans 0.272935295 seconds. A `sacct` query
was attempted, but its accounting database connection was unavailable; no
`sacct` evidence is claimed.

## Frozen Coverage

The coordinator replayed the accepted C85U field and immutable C84S action
objects before analysis. The final result covers:

```text
utility contexts:                  944
candidate utility rows:         76,464
historical method-context rows: 18,432
finite Q0 action records:    8,749,056
Q0 shards:                         944
registered tables:                  26
registered table rows:          21,607
manifest artifacts:                 29
bundle files / bytes:       32 / 7,115,589
```

The production validator replayed every manifest path, size, hash and CSV row
count; all rows have the required exploratory tag. It also replayed the
authorization/result/completion linkage, exact seven-event lifecycle, all
protected counters, and the immutable C84/C85 states. No staging root remains.

## Exploratory Findings

These findings are descriptive consequences of the already frozen C84 field.
They are not new confirmation, selectors, p-values, or mechanism claims.

### Realized Policy Use

Nontrivial exact action-map collapse is scope-specific:

| Policy / reference | Exact scopes |
|---|---|
| U11 / B1 | Cho full panel `160/160`; Lee level 0 `88/88`; Physionet level 1 `304/304` |
| U15 / S1 | Cho full panel `160/160`; Lee level 1 `88/88`; Physionet full panel `608/608` |

T3 is exactly applicable only in these exact scopes (and trivial self-reference
scopes). No nontrivial policy collapses to its reference over the global
three-dataset field. Near-collapse is not treated as exact collapse.

### Candidate Geometry

Target-equal mean best-second raw composite-utility gaps were:

| Dataset | Level 0 | Level 1 |
|---|---:|---:|
| Cho2017 | 0.028880 | 0.031719 |
| Lee2019_MI | 0.038187 | 0.040246 |
| PhysionetMI | 0.052577 | 0.050158 |

These are held-evaluation geometry diagnostics. They are not selection-time
observables and do not identify why a policy succeeded or failed.

### Mean And Tail Risk

Full-panel target-equal standardized regret summaries illustrate that mean and
tail behavior remain distinct:

| Dataset | Method | Mean | Worst target | CVaR 0.90 |
|---|---|---:|---:|---:|
| Lee | S1 | 0.425163 | 0.627396 | 0.605646 |
| Lee | COTT | 0.277125 | 0.428965 | 0.422066 |
| Lee | Q0 FULL | 0.286498 | 0.577011 | 0.556788 |
| Cho | S1 | 0.537782 | 0.675689 | 0.664089 |
| Cho | COTT | 0.356778 | 0.557096 | 0.524899 |
| Cho | Q0 FULL | 0.209206 | 0.354383 | 0.341429 |
| Physionet | S1 | 0.446184 | 0.710180 | 0.604388 |
| Physionet | COTT | 0.349544 | 0.640629 | 0.569962 |
| Physionet | Q0 FULL | 0.376618 | 0.712668 | 0.598048 |

COTT has lower mean standardized regret than S1 in all three datasets, while
the frozen C84 target-tail qualification remains unchanged. Q0 FULL also has a
lower mean than S1 in all three datasets, but this does not change the
registered `C84-L4` frontier. In Cho, U11 and B1 are action-identical and have
identical full-panel mean, worst-target, and CVaR risk; this is realized policy
collapse, not incremental unlabeled-information value.

### Theorem Applicability

```text
T1: ASSUMPTIONS_NOT_IDENTIFIED
T2: DESCRIPTIVE_ANALOGUE_ONLY
T3: EXACTLY_APPLICABLE only in exact-collapse scopes
T4: ASSUMPTIONS_NOT_IDENTIFIED
T5: OPEN_THEOREM
T6: DESCRIPTIVE_ANALOGUE_ONLY
T7: ASSUMPTIONS_NOT_IDENTIFIED
```

C85E identifies no empirical Blackwell order, unrestricted information value,
minimax optimum, Le Cam lower bound, or T7 score-error law.

## Regression Verification

| Suite | Job | Accepted result | Pytest time | stderr bytes |
|---|---:|---|---:|---:|
| focused | 900803 | 394 passed, 3 deselected | 9.56 s | 0 |
| C65 | 900804 | 1,103 passed, 1 skipped, 12 deselected | 92.77 s | 0 |
| C23 | 900805 | 1,514 passed, 1 skipped, 12 deselected | 189.18 s | 0 |
| full OACI | 900806 | 2,438 passed, 1 skipped, 12 deselected | 522.82 s | 0 |

The deselections are historical readiness assertions superseded by accepted
later authorizations. No lock-bound implementation or test was changed. All
accepted stderr files are empty, and the production result semantic replay is
PASS.

## Immutable Boundary

```text
C84 primary:        C84-D
C84 label frontier: C84-L4

T1/T3/T4/T7: PROVED
T2/T6:       COUNTEREXAMPLE
T5:          OPEN

direct label/logit/EEG access: 0 / 0 / 0
selector/Q0/inference calls:   0 / 0 / 0
theorem-status writes:         0
active acquisition / C86:      0 / 0
manuscript writes:             0
```

C85E stops at C86 protocol review. It does not authorize C86, active
acquisition, new data/model zoos, real-data expansion, or manuscript changes.
