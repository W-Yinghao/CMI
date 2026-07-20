# C85UR1 Protocol Timing Audit

## State Before Protocol Commit

```text
HEAD == origin/oaci:                            yes
branch:                                         oaci
worktree clean:                                 yes
real evaluation-label rows opened:              0
real target/logit payloads opened:              0
real Q0 shards/direct C84S tables opened:       0
real candidate utilities computed:              0
C85U authorization record:                      absent
C85U authorization consumption:                 0
```

The C85URP implementation and V1 lock were inspected only through repository
source, committed reports, and compact metadata. No protected empirical object
was opened to discover the four execution-governance blockers.

## Prospective Repair

This protocol is committed before any C85UR1 implementation change. It fixes:

```text
U1-only runtime registry with no Stage-B or scientific-result path;
U2-only runtime registry with no label, target-artifact, or logit path;
semantic protected-input replay receipt V2;
same-authorization/same-attempt O_EXCL U1 and U2 stage receipts;
versioned U1 and U2 manifests and handoffs;
one atomic final C85U acceptance transaction;
post-rename recovery and primary-exception precedence;
resource and exact-coverage enforcement.
```

The historical utility formula, candidate universe, contexts, evaluation
labels, Q0 actions, and C84S decision rows are unchanged. No tolerance, metric,
selector, threshold, p-value, theorem status, or scientific claim is chosen
from reconstructed values.

## Protected-Read Boundary

C85UR1 uses shadow and adversarial fixtures only. It does not read:

```text
evaluation-label rows;
target NPZ/logit payloads;
Q0 shards;
direct C84S result tables.
```

Real protected reads remain unavailable until a future standalone
`授权 C85U` is bound to the unique V2 lock.

## Historical Status

The V1 lock remains byte-identical and is classified prospectively as:

```text
SUPERSEDED_BEFORE_AUTHORIZATION_OR_REAL_PROTECTED_ACCESS
```

The C85EP availability blocker remains valid. A future U1 artifact is
provisional until U2 and the final acceptance transaction both succeed.

