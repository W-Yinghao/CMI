# C85TR1 Final Report Red Team

## Final Gate Under Review

```text
C85T_EXECUTION_GUARD_RNG_REPLICATE_PERSISTENCE_AND_PROOF_REVIEW_REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION
```

## Result

```text
checks:   96 / 96 PASS
blockers: 0
```

## Check Ledger

| # | Category | Check | Result |
|---:|---|---|---|
| 1 | chronology | Entering HEAD equals the accepted C85TL final HEAD | PASS |
| 2 | chronology | Entering HEAD was synchronized with `origin/oaci` | PASS |
| 3 | chronology | C85TR1 repair protocol is additive and has its own sidecar | PASS |
| 4 | chronology | Repair-protocol SHA-256 replays exact bytes | PASS |
| 5 | chronology | Repair-protocol commit precedes implementation commit | PASS |
| 6 | chronology | Implementation commit precedes V2 lock commit | PASS |
| 7 | chronology | Timing audit records zero registered draw/proof/status before repair | PASS |
| 8 | chronology | Pre-lock direct statement is disclosed as unbound and non-reusable | PASS |
| 9 | preservation | C85P protocol SHA-256 remains exact | PASS |
| 10 | preservation | C85R repair-protocol SHA-256 remains exact | PASS |
| 11 | preservation | C85R V2 generator SHA-256 remains exact | PASS |
| 12 | preservation | C85TL operationalization SHA-256 remains exact | PASS |
| 13 | preservation | Historical C85T lock SHA-256 remains exact | PASS |
| 14 | preservation | Historical lock authorization was absent | PASS |
| 15 | preservation | Historical lock registered execution count is zero | PASS |
| 16 | preservation | Historical lock is marked superseded, not rewritten or relabelled successful | PASS |
| 17 | RNG bytes | Operative S9 call requests `numpy.int64` exactly | PASS |
| 18 | RNG bytes | Operative S9 call contains no `uint8` draw request | PASS |
| 19 | RNG bytes | Draw order is exactly 51 L then 46 H | PASS |
| 20 | RNG bytes | Mapping is exactly 0 to -1 and 1 to +1 | PASS |
| 21 | RNG bytes | Canonical mapped-draw persistence dtype is little-endian int64 | PASS |
| 22 | RNG bytes | Canonical L, H, and combined SHA-256 fields are present | PASS |
| 23 | RNG bytes | Historical uint8 helper rejects registered scenarios | PASS |
| 24 | RNG bytes | Historical uint8 and V2 int64 byte streams are demonstrably distinct | PASS |
| 25 | RNG bytes | Seed namespace remains `C85_SYNTHETIC_V1` | PASS |
| 26 | RNG bytes | Bit generator remains PCG64DXSM | PASS |
| 27 | RNG bytes | Replicate IDs remain exactly 0 through 4095 | PASS |
| 28 | RNG bytes | No registered S9 stream was opened during C85TR1 | PASS |
| 29 | intervals | Probability raw interval is mean plus/minus 1.96 MC SE | PASS |
| 30 | intervals | Reported probability interval is clipped to [0,1] | PASS |
| 31 | intervals | `interval_clipped` records whether an endpoint changed | PASS |
| 32 | intervals | Probability point estimate is not silently clipped | PASS |
| 33 | intervals | Mean-regret interval remains unbounded and unmodified | PASS |
| 34 | intervals | Boundary shadow fixture replays raw and clipped values | PASS |
| 35 | intervals | Interior shadow fixture reports no clipping | PASS |
| 36 | replicate persistence | S6 NPZ schema contains all five required arrays | PASS |
| 37 | replicate persistence | S7 NPZ schema contains all five required arrays | PASS |
| 38 | replicate persistence | S9 NPZ contains both design schemas and paired arrays | PASS |
| 39 | replicate persistence | S6/S7 logical row total is fixed at 8,192 | PASS |
| 40 | replicate persistence | S9 logical replicate-design row total is fixed at 8,192 | PASS |
| 41 | replicate persistence | S9 raw-draw digest row total is fixed at 4,096 | PASS |
| 42 | replicate persistence | Missing, duplicate, or reordered replicate IDs fail | PASS |
| 43 | replicate persistence | Object-dtype arrays fail before persistence | PASS |
| 44 | replicate persistence | Nonfinite arrays fail before persistence | PASS |
| 45 | replicate persistence | NPZ members use deterministic order and ZIP metadata | PASS |
| 46 | replicate persistence | S6 aggregate replays exactly after disk reload | PASS |
| 47 | replicate persistence | S7 aggregate replays exactly after disk reload | PASS |
| 48 | replicate persistence | S9 aggregate replays exactly after disk reload | PASS |
| 49 | authorization | V2 authorization schema is unique and explicit | PASS |
| 50 | authorization | Future record must contain exact direct statement | PASS |
| 51 | authorization | Future record must bind lock SHA and lock commit | PASS |
| 52 | authorization | Future record must bind one content-addressed absolute output root | PASS |
| 53 | authorization | Future record must bind one external consumption-ledger path | PASS |
| 54 | authorization | Protected C85E/active/real-data/manuscript fields must be false | PASS |
| 55 | authorization | Authorization binding SHA normalizes only the self-referential ledger path | PASS |
| 56 | authorization | Ordinary authorization-file SHA is separately recorded at preflight | PASS |
| 57 | authorization | Consumption uses atomic `O_CREAT|O_EXCL` | PASS |
| 58 | authorization | Existing receipt blocks same-root reuse | PASS |
| 59 | authorization | Existing receipt blocks different-root reuse | PASS |
| 60 | authorization | CLI root differing from authorization root fails before consumption | PASS |
| 61 | authorization | Current V2 authorization record is absent | PASS |
| 62 | authorization | Current V2 authorization consumption receipt is absent | PASS |
| 63 | capability | Registered capability constructor requires a module-private sentinel | PASS |
| 64 | capability | Capability is issued only after durable exclusive consumption | PASS |
| 65 | capability | Capability binds authorization SHA, lock SHA, attempt, and root | PASS |
| 66 | capability | Process-issued object identity is checked on every registered use | PASS |
| 67 | capability | Static string cannot unlock a registered seed | PASS |
| 68 | capability | Arbitrary object and `None` cannot unlock registered execution | PASS |
| 69 | capability | Capability cannot migrate to another attempt | PASS |
| 70 | capability | Capability cannot migrate to another output root | PASS |
| 71 | capability | Shadow fixtures reject a registered capability | PASS |
| 72 | proof governance | C85T produces proof candidates, not proof verdicts | PASS |
| 73 | proof governance | Candidate disposition vocabulary is closed | PASS |
| 74 | proof governance | Candidate check label states schema/internal consistency only | PASS |
| 75 | proof governance | Candidate text explicitly denies independent review | PASS |
| 76 | proof governance | All C85T formal statuses are forced to OPEN | PASS |
| 77 | proof governance | Automatic transition helper refuses even fabricated PASS | PASS |
| 78 | proof governance | Simulation cannot change theorem status | PASS |
| 79 | proof governance | Seven proof-candidate filenames are fixed | PASS |
| 80 | proof governance | C85V is separately authorized and cannot rerun Monte Carlo | PASS |
| 81 | proof governance | C85TR1 created zero canonical proof candidates | PASS |
| 82 | lifecycle | Lifecycle storage is canonical append-only JSONL | PASS |
| 83 | lifecycle | Successful stage order is exact and closed | PASS |
| 84 | lifecycle | Every event binds authorization SHA, lock SHA, and attempt | PASS |
| 85 | lifecycle | Artifact/receipt SHA is recorded where applicable | PASS |
| 86 | lifecycle | Each event append is flushed with `fsync` | PASS |
| 87 | lifecycle | Skipped or reordered stage fails | PASS |
| 88 | lifecycle | FAILED is terminal and reports last completed stage | PASS |
| 89 | lifecycle | FAILED preserves primary exception type and message | PASS |
| 90 | manifest/atomicity | V2 manifest requires every registered artifact class | PASS |
| 91 | manifest/atomicity | Manifest replays artifact path, size, and SHA-256 | PASS |
| 92 | manifest/atomicity | Manifest requires 11 scenarios, exact replicate counts, and seven OPEN statuses | PASS |
| 93 | manifest/atomicity | Three injected failures leave no final root | PASS |
| 94 | lock/isolation | V2 lock/sidecar and all 133 object bytes/Git blobs replay | PASS |
| 95 | lock/isolation | Operative modules import no Torch, MNE, MOABB, empirical selector, or GPU stack | PASS |
| 96 | regression/hygiene | Four suites pass with empty stderr; `squeue` is clear; prohibited Git payload is absent | PASS |

## Historical Failure Reproduction

The red team replays the original mismatches rather than erasing them:

```text
historical S9 implementation dtype:
  uint8

operative protocol dtype:
  int64

historical result persistence:
  aggregates and digests only

operative V2 persistence:
  exact replicate arrays plus digests

historical proof review:
  same-module token scan with automatic transition path

operative V2 proof governance:
  proof candidates only, formal OPEN, later independent C85V

historical lifecycle:
  one-shot JSON

operative V2 lifecycle:
  ordered append-only JSONL
```

## Adversarial Authorization Cases

| Case | Consume | Registered execution | Result |
|---|---:|---:|---|
| fresh valid record and exact root | 1 | eligible | PASS |
| same record after receipt exists | 0 | 0 | FAIL CLOSED |
| same record with another root | 0 | 0 | FAIL CLOSED |
| root not content-addressed | 0 | 0 | FAIL CLOSED |
| lock SHA drift | 0 | 0 | FAIL CLOSED |
| lock commit drift | 0 | 0 | FAIL CLOSED |
| protected field true | 0 | 0 | FAIL CLOSED |
| static capability string | 0 | 0 | FAIL CLOSED |
| pre-lock direct statement without V2 record | 0 | 0 | NOT AUTHORIZATION |

## Atomic Failure Injection

Shadow-only failures were injected:

```text
before C85T_RESULT.json;
after partial artifacts but before manifest;
after manifest replay but before final rename.
```

At all three points:

```text
final root exists: false
failed staging preserved: true
automatic retry: false
registered S0-S10 draw: 0
```

## Claim Red Team

The following claims are explicitly rejected:

```text
C85TR1 executed S0-S10;
shadow fixtures are synthetic scientific results;
the pre-lock authorization statement authorizes V2;
the historical lock is still operative;
uint8 and int64 streams are byte-equivalent;
aggregate digests substitute for replicate persistence;
clipped probability intervals alter point estimates;
a public string can create a registered capability;
a consumed authorization can be moved to a new root;
same-process internal checks are independent proof review;
any T1-T7 theorem status changed;
C85T proof candidates are accepted proofs;
C85V is authorized;
C85E or active acquisition is authorized;
real data were accessed;
manuscript work is authorized.
```

No rejected statement is asserted affirmatively in the V2 protocol, lock,
contract tables, readiness report, or tests.

## Protected Counters

```text
registered S0-S10 draws:          0
registered MC rows:               0
registered scenario result roots: 0
canonical proof candidates:       0
independent proof verdicts:        0
theorem-status transitions:        0
C85T authorization records:        0
authorization consumptions:        0
real project data accesses:        0
active acquisition executions:     0
C85V/C85E authorizations:           0 / 0
training/forward/GPU:               0 / 0 / 0
manuscript changes:                 0
```

## Regression Evidence

```text
focused: 375 passed
C65:     986 passed, 1 skipped, 3 deselected
C23:   1,397 passed, 1 skipped, 3 deselected
full:  2,321 passed, 1 skipped, 3 deselected

accepted stderr:
  0 bytes in every suite

post-lock focused guard replay:
  27 / 27 passed
```

## Disposition

The red team found no remaining authorization, RNG-byte, persistence,
proof-independence, lifecycle, lock-replay, or isolation blocker. C85T remains
unexecuted and requires a new direct authorization bound to the exact V2 lock.
