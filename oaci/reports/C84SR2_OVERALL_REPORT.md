# C84SR2 Overall Report

## Final Disposition

```text
C84S_STAGE_B_FIELD_DESCRIPTOR_COMPATIBILITY_REPAIRED_AND_V4_LOCK_READY_FOR_FRESH_PI_AUTHORIZATION
```

C84SR2 completed the additive Stage-B field-descriptor compatibility repair and
created a fresh C84S V4 analysis execution lock. It did **not** execute the real
scientific comparison. A new direct `授权 C84S` statement is required against
the V4 lock.

## Chronology And Identities

```text
repair protocol commit:
  5fa4366f57087f07cf6e290a84f37abbb1ef97c8

repair protocol SHA-256:
  6d7853cd60a85c9f3516cb21fda1c75909f0963e96ad2ac0292647bdc93f1aef

implementation commit:
  5fddbb103600207f70e2cfff0d615ee81d5391f2

pre-lock attempt-ledger amendment / lock-bound implementation commit:
  a737e2b91578aa7e85aab948ca4b0e64929c3073

V4 execution-lock commit:
  8528bd142bbf6c4cca047bdcc558133eebf5e757

V4 execution-lock SHA-256:
  582e5074b4b17d62ff1e5fbfd992f037dd3082b7763b22d707630aa19db81c3d

final accepted regression commit:
  b08538d3f399c77bb188246d23472cb5fd39ded5
```

The protocol was committed before implementation. The original readiness
command supplied an incorrect expanded implementation hash and failed before
the lock was written. That non-scientific pre-lock failure is retained in the
failure and regression-attempt ledgers; the corrected command used the exact
current commit.

## Preserved V3 Attempt

Authorized C84S V3 job `897843` remains an immutable failed attempt:

```text
authorization consumed:        yes
construction label access:     1
evaluation label access:       0
selector contexts computed:    0
scientific result rows:        0
training / forward / GPU:      0 / 0 / 0
same-label oracle:             0
evaluation descriptor sealed: yes
```

Stage A completed, then Stage B failed during context enumeration on the first
historical sidecar lacking `level_intervention_id`. The failed V3 root, its
authorization consumption, lifecycle ledger, Stage-A subprocess evidence and
Stage-B traceback remain preserved. The V3 authorization is consumed and is
not reusable or transferable to V4.

## Root Cause And Repair

The complete field descriptor has an authoritative intervention identity for
all 1,944 candidate units. The training-sidecar schemas divide exactly as
follows:

```text
native sidecars with level_intervention_id: 1,701
historical sidecars without the field:        243
total:                                      1,944
```

All 243 omissions are exactly the reused C84C units:

```text
datasets: Lee2019_MI / Cho2017 / PhysionetMI
panel:    A
seed:     5
level:    0
provenance: C84C
expected intervention:
  C84_LEVEL0_FULL_SOURCE_PANEL_V1
```

The repaired resolver uses the frozen complete-field descriptor as the
authority. A present sidecar value must match both the descriptor and locked
level definition. An absent value is accepted only for the exact C84C scope
above. Every other omission, provenance change, panel/seed/level change or
identity mismatch fails closed. Level-1 remains exactly
`C84_LEVEL1_FIXED_PANEL_LEFT_HAND_CELL_DELETION_V1`.

No candidate ID, method, score formula, prior, threshold, budget, label split,
inference rule or taxonomy changed.

## Immutable Stage-A Reuse

C84SR2 replays the already provisioned Stage-A views by immutable identity. It
does not call a dataset label loader and does not create a new split.

```text
Stage-A complete SHA-256:
  29e77b600184f3f96b65858a40767f562e664e81eb848e2546e5486886ed35bd

construction handoff SHA-256:
  29b0848b8bbc40d5d346722d7cc479c69995406444ee8b9688f663f9c4256223

evaluation seal SHA-256:
  54e06dff60d80255631dc4faa20c8c7db651f2af8fc5415671dd9ab6681b5502

construction rows / manifest / table:
  4,773
  8b86722c6b795ceec7bdf211b53b01e720c5a07ae5ddefa2139ca1fcee91be40
  fdf36052d36ad9546cda06cbc567f68cdcced7ad08fd1311ab949471218b3134

evaluation rows / manifest / table:
  4,848
  6fad247629eb48340a4badf9ab1a0669652757a58216e46826e4dfd8bfd608bd
  ea76c34663edac1e6e7e844fee6af3f06058aaaf3846febda1dff94df343a371

construction/evaluation overlap: 0
new label-loader calls:           0
target label rows reloaded:       0
```

The V4 coordinator consumes a fresh authorization, executes an immutable
Stage-A replay subprocess, passes only the construction handoff to Stage B, and
releases the evaluation seal to Stage C only after the exact selection freeze.

## Validation

The real metadata-only descriptor audit enumerated:

```text
field units:             1,944 / 1,944
level 0 / level 1:         972 / 972
target contexts:           944 / 944
candidates/context:         81 / 81
historical omissions:      243 / 243
```

The readiness replay streamed and rehashed 7,776 frozen external objects,
totaling 48,072,941,176 bytes. No candidate array or target-label value was
used to choose the compatibility rule.

The new full-scale synthetic production path passed:

```text
contexts:                    944
Q0 chains:                 2,048
Q0 records:            9,110,448
method-context rows:      18,608
selection-freeze SHA-256:
  0012344acd674df4d28e78f08cf81dac67ec7b37fef3e851faf5892efe6e6982
synthetic result SHA-256:
  33d825a4d8fe0538b1f775b01cf98a1dff1655f934b16c103339de64cf17d990
synthetic summary SHA-256:
  7b88c30f0f623894a33cfcfa6aea56149500a8df7a1f5cb202d765e169384749
```

All C84-A/B/C/D/E and C84-L1/L2/L3/L4 synthetic branches matched their locked
truth tables. No precomputed method-context rows were injected. Real field
array access, real target-label access, real selector scores and real scientific
statistics were all zero. Final report red-team: **50/50 PASS**.

## Regression

Accepted clean-pushed commit `b08538d3`:

```text
focused:   242 passed
C65:       843 passed, 1 skipped, 3 deselected
C23:     1,254 passed, 1 skipped, 3 deselected
full:    2,178 passed, 1 skipped, 3 deselected
```

All accepted stderr files are empty. The sole skip is finalized C78F; the three
deselections are established C79 authorization-state tests. Jobs were monitored
with `squeue`; `sacct` was not used.

The initial regression attempt is not hidden. Six historical tests enumerated
the current C84S lock set only through V3. Focused, C65 and C23 exposed the same
six failures; the initial full job was canceled once that deterministic class
was established. The assertions were updated to recognize additive V4, then all
four suites passed. Runtime code and the V4 lock were unchanged by that test-only
repair.

## Evidence Boundary

C84SR2 establishes implementation and lock readiness only. It does not establish
target accuracy, selector performance, Q1/Q2, a label-budget frontier, level
effects, cross-dataset same-method recurrence or external validity.

At this gate:

```text
V4 authorization record:       absent
real evaluation-label access:  0
real selector scores:          0
real scientific statistics:    0
training / forward / GPU:      0 / 0 / 0
same-label oracle:             0
C85 authorization:             false
```

The next valid action is a fresh direct PI statement:

```text
授权 C84S
```

It must bind to the unique V4 lock. No earlier authorization migrates.
