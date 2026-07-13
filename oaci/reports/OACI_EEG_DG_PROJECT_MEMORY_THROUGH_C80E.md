# OACI EEG-DG Project Memory Through C80E Preflight

## Current gate

```text
C80E_AUTHORIZATION_PROTOCOL_LOCK_VIEW_OR_DEPENDENCE_BLOCKER
```

Direct C80E authorization was received on 2026-07-13 for protocol `f5d83b3`,
protocol SHA-256
`c6292597f5610cb96e8eaf0313eaa741f8fa9b11dd89ff9e4d84db1fa33add85`,
analysis lock `972f47c`, and the seed-3/seed-4 field/view manifests enumerated
by that lock. Preflight stopped before any real budget statistic or C80
evaluation-label value was computed.

## Accepted C80P base

```text
C80P protocol:                 f5d83b3
C80P synthetic implementation: c98e084
C80P analysis lock:            972f47c
C80P readiness result:         1b02454
C80P final handoff:            0a55fff
registry:                      80 / 80
locked grid:                   [1,2,4,8,16,32,FULL]
Monte Carlo chains:            2,048
```

Budget 64 was removed before protocol hashing by the registered
availability-only rule because the minimum construction class count was 61.
No candidate score or evaluation outcome informed that decision.

## Preflight passes

```text
HEAD == origin/oaci at entry:        0a55fff
protocol hash replay:                pass
analysis-lock hash replay:           pass
locked implementation hashes:        4 / 4
accepted-input hashes:               10 / 10
registry cells:                      80 / 80, blank 0
all five paths unconditional:         yes
target 4 primary:                      0
same-label oracle reachable:           0
real budget statistics:                0
evaluation-label value reads for C80:  0
```

## Blocking findings

### B1: final taxonomy is not locked

No committed C80 protocol, analysis lock, registry, claim contract,
implementation, or test contains the required C80-A through C80-E overlap
precedence. `near-FULL`, which distinguishes C80-C from other positive-frontier
states, is also undefined. The C80E handoff requires the exact decision table
to come from the committed protocol and explicitly prohibits executor choice.

### B2: authorization guard and lock schema disagree

The locked `assert_c80e_authorized()` function reads
`lock.get("protocol_sha256")`. The committed lock stores the protocol hash at
`lock["protocol"]["sha256"]`. With the authorization record present, the
locked `run-real` command fails before data access with:

```text
RuntimeError: C80E authorization/lock binding mismatch
```

### B3: the real-data adapter is not part of the lock

The locked C80 module deliberately contains no real-data loader and its
`run-real` path raises after authorization. `972f47c` hashes only the pure
frontier module, synthetic module, C80P tests, and regression script. It does
not bind an executor for selection freezing, simultaneous target bands, paired
cross-seed heterogeneity, S1/S2/S3, result freezing, or report schemas against
the external arrays.

## Scientific state

No C80-A through C80-D result exists. The seed-specific B* values, budget
curves, reliability/actionability relationship, top-k behavior, target
heterogeneity, and S3 moderation remain uncomputed.

The accepted C79E component result remains unchanged: P1-M failed while P1-A
passed; P2-L failed while fixed-kernel LOTO/LORO qualification failed again;
aggregate directions were concordant across seeds. This is not reversal or
absence of all signal.

## Required repair

Before any C80 budget outcome, an additive prospective repair must:

1. preserve `f5d83b3`, `972f47c`, and the blocker ledger in history;
2. bind complete C80-A through C80-E precedence and define `near-FULL`;
3. repair and test authorization against the actual nested lock schema;
4. implement, test, hash, and lock the real-data adapter without changing the
   registered grid, selector, RNG, thresholds, dependence, or five paths;
5. issue a new operative protocol hash and analysis lock;
6. obtain new direct PI authorization bound to those repaired objects.

The current authorization cannot be reused for objects that do not yet exist.
No C81, seed 5, BNCI2014_004, active acquisition, oracle analysis, new
feature/kernel/model search, or manuscript drafting is authorized.
