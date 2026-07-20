# C85UR1 Final Report Red Team

## Verdict

```text
72 / 72 PASS

C85U_PROCESS_ISOLATION_PROTECTED_REPLAY_AND_ACCEPTANCE_TRANSACTION_REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION
```

`PASS` means the repaired V2 path and lock are ready for PM review. It does not
mean C85U ran, protected data were opened, or C85E became authorized.

## Identity And Chronology - 10/10

| Check | Result |
|---|---|
| Starting HEAD and origin identity replay | PASS |
| Historical C85U protocol SHA exact | PASS |
| Historical V1 lock SHA exact | PASS |
| V1 files remain byte-identical | PASS |
| V1 superseded before authorization/protected access | PASS |
| C85UR1 protocol committed before implementation | PASS |
| Implementation committed before V2 lock | PASS |
| V2 lock self-hash exact | PASS |
| 54/54 repository objects byte/Git bound | PASS |
| Pushed `oaci` chronology exact | PASS |

## U1 Isolation And Input Policy - 12/12

| Check | Result |
|---|---|
| U1 imports dedicated runtime registry | PASS |
| U1 registry excludes selection manifest | PASS |
| U1 registry excludes candidate scores/ranks | PASS |
| U1 registry excludes fixed selections | PASS |
| U1 registry excludes Q0 index/shards | PASS |
| U1 registry excludes scientific result/manifest | PASS |
| U1 registry excludes method-context decisions | PASS |
| U1 registry excludes Q1/Q2/LOTO/frontier/taxonomy | PASS |
| Dynamic open trap observed no forbidden U1 path | PASS |
| 1,944 target and sidecar identities retained | PASS |
| 944 contexts and canonical 81-order retained | PASS |
| Historical utility formula and tie rule unchanged | PASS |

## Protected Replay And U1 Guard - 10/10

| Check | Result |
|---|---|
| Receipt schema is `c85u_protected_input_replay_receipt_v2` | PASS |
| Authorization file/binding/ID linked | PASS |
| Lock SHA/commit linked | PASS |
| Attempt and exact output root linked | PASS |
| Evaluation table hash/row count linked | PASS |
| Evaluation-view manifest linked | PASS |
| 1,944 artifacts and 48,018,748,054 bytes linked | PASS |
| Artifact and sidecar registry digests linked | PASS |
| Valid-hash wrong-schema receipt rejected | PASS |
| Cross-authorization/attempt/root receipt rejected pre-access | PASS |

## U2 Isolation And Attempt Binding - 10/10

| Check | Result |
|---|---|
| Real U2 requires execution context | PASS |
| Real U2 requires U1 handoff and utility root | PASS |
| Real U2 requires explicit output root | PASS |
| U2 registry contains no hard-coded project path | PASS |
| U2 registry contains no label/logit/target path | PASS |
| U1 manifest/handoff lock/auth/attempt replay | PASS |
| `STAGE_U1_COMPLETED` required | PASS |
| Fresh U2 `O_EXCL` receipt required | PASS |
| Direct/cross-attempt invocation fails before open | PASS |
| 18,432 rows and 8,749,056 finite Q0 actions retained | PASS |

## V2 Stage Objects - 8/8

| Check | Result |
|---|---|
| U1 V2 manifest schema/version exact | PASS |
| U1 V2 handoff schema/version exact | PASS |
| U1 binds protected replay and stage receipt | PASS |
| U1 records allowed and forbidden access counters | PASS |
| U2 V2 replay schema/version exact | PASS |
| U2 V2 handoff schema/version exact | PASS |
| U2 binds U1 and protected Stage-B/result hashes | PASS |
| U1/U2 remain provisional and not C85E-accepted | PASS |

## Acceptance And Failure Semantics - 10/10

| Check | Result |
|---|---|
| Final bundle includes all required identities | PASS |
| Lifecycle has exact ordered stages | PASS |
| U1 and U2 replay before acceptance | PASS |
| Manifest and completion receipt written in staging | PASS |
| Terminal lifecycle written before publication | PASS |
| Files/directories fsynced before publication | PASS |
| Publication uses one final `os.replace` | PASS |
| No required operation after rename | PASS |
| Valid post-rename bundle recovers as success | PASS |
| Primary exception survives secondary failures | PASS |

## Resource And Scientific Boundary - 6/6

| Check | Result |
|---|---|
| 2 GiB U1 output envelope enforced | PASS |
| Exact 944 x 81 scope cannot be reduced | PASS |
| C84-D and C84-L4 unchanged | PASS |
| T1-T7 formal statuses unchanged | PASS |
| No C85E/C86/active/new-zoo/manuscript authorization | PASS |
| Real labels/NPZ/Q0/direct-result access counters all zero | PASS |

## Regression And Publication - 6/6

| Check | Result |
|---|---|
| Post-lock C85UR1 tests 21/21 | PASS |
| Focused regression accepted | PASS |
| C65 cumulative accepted | PASS |
| C23 cumulative accepted | PASS |
| Full OACI accepted | PASS |
| All accepted stderr empty and active C85 jobs zero | PASS |

## Preserved Failure Evidence

The initial shadow run (`12 passed / 4 failed`) and incorrect-SHA lock build are
retained in `readiness_attempt_ledger.csv`. Both failed before authorization,
before protected access, and before creation of an operative lock. The repaired
shadow run and exact lock build then passed without weakening the contract.

## Residual Risk

Future U1 remains a protected approximately 48 GB read and can still stop on a
filesystem, historical identity, or deterministic replay mismatch not exposed
by shadow fixtures. The V2 lock forbids automatic retry, runtime formula or
tolerance changes, and partial acceptance. U2 must exactly replay all 18,432
historical rows; U1 alone remains insufficient for C85E.

The next permissible action is PM review and, if approved, a fresh standalone
`授权 C85U`. It is not C85U execution or C85E authorization.
