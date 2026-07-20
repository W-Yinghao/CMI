# C84SR1 Overall Report

## Decision

C84SR1 repairs the missing real Stage-A to Stage-B to Stage-C orchestration and
the finite-budget Q0 chain-integration contract. It preserves the historical V1
and V2 analysis locks and stops before real target-label access.

Final gate:

```text
C84S_REAL_EXECUTION_ORCHESTRATION_Q0_INTEGRATION_REPAIRED_AND_LOCKED_READY_FOR_FRESH_PI_AUTHORIZATION
```

This is a readiness gate only. C84S real execution is not authorized. The
earlier direct statement against the nonoperative V2 lock was never consumed
and does not migrate to V3.

## Chronology And Identities

```text
base blocker HEAD:             9cdeb9ae2794226ff789411dea0ced10026a216f
repair protocol commit:        764fed0219acd412785f240a395d9236879c9b9b
production implementation:     86dfda5191dc00d4c9f70265657784626a0370a6
readiness-binding repair:       97c1305b1574dcf1cbe35318b1ac61f13fbaad6a
V3 lock/readiness commit:       4774f72d4c2674fed409bab950bce8ce70df2264

repair protocol SHA-256:       3bdfbf67f1e1697a1488ccb5b7148494db06586ea9ff4318f16e030b88e7be2a
historical V2 lock SHA-256:     94c896f0f00c53441095da6225f9ac574eb4a9baa904821a5dab3f11ea76f75c
replacement V3 lock SHA-256:   15d8a84b7870021a90b1f0f103a8a4d733523b249321b143a89091978c0aa9fc
complete field SHA-256:        cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8
model field SHA-256:           d8931b81a3d68f4b1e098ac6e3ede3cd44cdb6c70cdef9f18a76e0a8c62ecdb2
target registry SHA-256:       52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8
```

The additive repair protocol was committed before implementation. At protocol
and lock time, real target-label access, real selector scores, scientific
statistics, training, forward, GPU use and same-label oracle access were all
zero.

## Blocker Reconciliation

The historical V2 lock contained useful components but no lock-bound production
path that enumerated all contexts, materialized Stage B, froze selection before
evaluation, integrated Q0 chains, built all method-context rows and invoked the
final inference/taxonomy writer. Its synthetic benchmark entered Stage C with
fabricated method-context rows and therefore did not exercise that missing
transformation.

C84SR1 adds one public coordinator:

```text
python -m oaci.multidataset.c84sr1_execute run-real \
  --authorization-record <fresh-V3-record> \
  --output-root /projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v3
```

No notebook, shell glue or unbound callable is needed to complete the future
workflow.

## One-Way Production Stages

Stage A is a label-only subprocess. Candidate artifacts are unavailable. It
replays the frozen 9,621-row trial registry and creates physically separate
construction and evaluation roots. Its Stage-B handoff contains only the
construction descriptor; the evaluation descriptor remains sealed.

Stage B enumerates contexts from frozen field descriptors, not filenames or
directory order. It receives source-audit, target-unlabeled, genealogy and
construction-label inputs but no evaluation descriptor, oracle, regret or held
utility. It atomically freezes all deterministic and stochastic selections.

Stage C starts only after the Stage-B manifest validates. It receives the
immutable selection freeze and the sealed evaluation descriptor, but no
construction descriptor or selector callable. It cannot alter a score, rank,
sample or selected candidate.

Authorization is consumed before Stage A, and the coordinator plus each stage
write attempt ledgers, input receipts, output handoffs, protected counters and
failure manifests. There is no automatic retry.

## Exact Arithmetic

| Object | Count |
|---|---:|
| target contexts | 944 |
| candidates per context | 81 |
| S1/zero-label candidate score rows | 535,248 |
| S1/zero-label candidate rank rows | 535,248 |
| fixed default selections | 4,720 |
| Q0 records | 9,110,448 |
| immutable Q0 shards | 944 |
| final method-context rows | 18,608 |

The 944 contexts are Lee 176, Cho 160 and Physionet 608. Stage C produces 21
methods per Lee/Cho context and 19 per Physionet context, yielding 18,608 rows
exactly.

## Q0 Integration

Each finite Q0 budget stores all 2,048 chains, the chain seed, sample digest,
selected index, complete `uint8[81]` candidate order, top-5/top-10 identities and
construction-metric digests. `FULL` is stored once per context. Shards contain
no object arrays.

Stage C computes each endpoint per chain and then takes the arithmetic mean
within the context. It integrates expected regret, selected utility, top-k
probabilities and ranking metrics. Chain fractions define stochastic selected
regime distributions; Monte Carlo standard errors are numerical diagnostics
only. Chains never enter target-cluster sample size or max-T inference.

## Measurement Schema

Method-context V2 adds explicit `rank_measurement_applicable` and
`performance_estimate_applicable` flags. Semantically inapplicable values are
null rather than fabricated zeros. Q0 is not treated as a calibrated performance
estimator. The ambiguous context-level `catastrophic_failure` field was removed;
registered target-level Q1/Q2 catastrophic tables remain operative.

The final atomic result registry covers 18 tables, including Q1/Q2, level,
panel/seed, LOTO, label frontier, top-k, selected utility, coverage, regime,
measurement-versus-decision and cross-dataset method intersections.

## Runtime Replay And Isolation

Before future label access the V3 guard will verify a clean synchronized `oaci`
branch, lock/protocol hashes, environment and loader identities, authorization
binding, and a fresh output root. It then streams and rehashes:

```text
target artifacts and context sidecars:   3,888 / 3,888
source-audit and training sidecars:       3,888 / 3,888
all external analysis-input files:        7,776 / 7,776
bytes replayed at readiness:              48,072,941,176
```

The lock binds 21 implementation files by both SHA-256 and Git blob, 15
readiness tables, all frozen protocols/registries, the full-scale synthetic
summary and the external artifact identity registry. Static process isolation
passed 24/24 checks.

## Full-Scale Synthetic Calibration

The production public entrypoints executed a complete 944-context fixture. It
did not inject precomputed method-context rows.

```text
Q0 chains:                    2,048
Q0 records:                   9,110,448
method-context rows:          18,608
selection-freeze SHA-256:     bd697d31c10c2bb8c2290f6066e2d7aaf220ccb69bef741f818938dd3012c1ac
synthetic result SHA-256:     fd4562ef9cb690d982d8ff3c89d8dc04ff27ee7af7aa584ce92ae68e674784e0
calibration summary SHA-256:  26e80934c75caae512d038e7939283ddef0b0d620c1c0686fe8cf55c1d5e8799
Stage-B wall:                 31.037 seconds
Stage-C wall:                 1,294.621 seconds
total wall:                   1,439.742 seconds
```

The same production result path passed C84-A/B/C/D/E and C84-L1/L2/L3/L4
branch fixtures, method-identity and level heterogeneity, partial-stage failure,
immutable selection, measurement-without-Q1 and top-k-without-Q1 contracts.
Real field-array, target-label, selector-score and scientific-statistic access
during calibration were all zero.

## Resources

The V3 envelope is CPU-only: 32 workers, 128 GiB RAM, 40 GiB output and 48
hours. The uncompressed Q0 candidate-order payload is 737,946,288 bytes. The
full-scale synthetic output used 369,775,815 logical bytes. All estimates are
inside the locked envelope; no target, context, method, chain, budget, candidate,
LOTO panel or max-T draw may be reduced at runtime.

## Verification

```text
focused C84SR1:  19 passed
C65 cumulative:  832 passed, 1 skipped, 3 deselected
C23 cumulative:  1,243 passed, 1 skipped, 3 deselected
full OACI:        2,167 passed, 1 skipped, 3 deselected
final red team:   65 / 65 PASS
risk register:    30 / 30 CLOSED
```

All accepted regression stderr files are empty. Scheduler monitoring used
`squeue`; no `sacct` claim is made. The sole skip is the finalized C78F test,
and the three deselections are fixed historical C79 authorization-state tests.

The initial cumulative run exposed six stale current-tree tests that enumerated
only V1/V2 locks. Their historical-commit assertions were preserved, while the
current-tree expectation was extended to V3. The replacement run passed on the
clean pushed lock commit. No lock-bound implementation or lock hash changed.

## Evidence Boundary And Next Step

At C84SR1 completion:

```text
real target construction/evaluation labels: 0 / 0
real selector scores:                       0
real scientific statistics:                0
training / forward / GPU / oracle:          0 / 0 / 0 / 0
C84S authorization record:                 absent
C84S real output root:                     absent
C85 authorization:                         false
```

Future C84S execution requires a fresh direct statement:

```text
授权 C84S
```

That future authorization must bind the unique V3 lock. It does not authorize
C85, new methods, retuning, new datasets or manuscript changes.
