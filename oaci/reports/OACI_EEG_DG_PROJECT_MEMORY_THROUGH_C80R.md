# OACI/EEG-DG Project Memory Through C80R

## Current Gate

```text
C80_REPAIRED_PROTOCOL_AND_REAL_ADAPTER_LOCKED_READY_FOR_PI_REAUTHORIZATION
```

C80R is protocol/implementation readiness only. C80E scientific execution is
not authorized. No C80 label-budget outcome has been computed.

## Accepted Scientific Base

C79E closed the two compound training-seed replication objects under the
locked rules while preserving component-level directional evidence:

```text
P1-M reliability gate: fail on seed4
P1-A material actionability: pass on both seeds
P2-L local association gate: fail on seed4
P2-T fixed LOTO/LORO nonqualification: pass on both seeds
H2R/H4R/H5R: exact candidates nonqualified again
H6R: family-wise inactive
```

The next question remains policy-specific label sufficiency: how many
independent construction labels per class are needed for low-regret selection
on a physically disjoint evaluation view? This is an existing-field design
study, not new-subject or external confirmation.

## C80P Safe Stop

The first C80E authorization reached preflight only. Commit `6c18fd4` preserved
three blockers before any outcome access:

1. no complete C80-A--E priority table or near-FULL definition;
2. guard code read `lock.protocol_sha256` rather than
   `lock.protocol.sha256`;
3. analysis lock `972f47c` bound synthetic primitives but no real adapter.

Protected state at that stop:

```text
real budget statistics:       0
evaluation-label value reads: 0
oracle accesses:              0
target4 primary rows:         0
training/forward/GPU:         0
```

The historical C80P protocol/hash/lock and authorization evidence remain in
Git but are superseded for execution. The old authorization is not reusable.

## C80R Additive Objects

```text
repair protocol commit: e88a24484590636f87d0f22798401a762875046a
protocol SHA-256:       2d72eb5119056a6520fd33fc0ac14ee6270bfd573b59c36b74be6aa3dc25fe39
final adapter commit:   e5cb41a5cd389674e3ec201d5c5f68a361c2fed3
final adapter SHA-256:  7e5ac0ba829bf5f233ed469f6fb8f6da4054d0bf4d024a0736a45e3674f1b56c
final lock commit:      f19acd8775f9b0ddf60401739741bec0019d021c
final lock SHA-256:     e18f2b5f1d79b6fcd96207339c5842e30b7aecb5bc22b8939a475487068b1b82
manifest digest:        6180275dcef26bdda4ae4b291d1ef6dc83434462ecacee0350fa94ae9c6a7fef
```

Lock revision `9617760` is preserved and superseded by `f19acd8`. Its
preauthorization red team found reporting-path completeness gaps. The additive
refinement introduced no outcome-dependent choice and changed no registry
entry, endpoint, budget, selector, threshold, RNG, MC count, dependence rule,
or materiality rule.

## Locked Taxonomy

Apply exactly in priority order:

```text
1 C80-E: any execution/view/dependence/protocol/provenance blocker
2 C80-D: either seed-specific B* is absent
3 C80-B: both B* exist but registered cross-seed stability fails
4 C80-C: stability passes and both B* are in {32,FULL}
5 C80-A: stability passes and C80-C is false
```

`near-FULL = {32,FULL}` is ordinal. `FULL` is all available construction
labels in each exact cell, not a universal 61 labels/class, and no interpolation
between 32 and FULL is permitted.

## Locked Runtime Boundary

The repaired adapter runs only after a new direct PI authorization binds the
protocol, lock, and manifest digest. Its order is:

```text
guard and manifest replay
-> construction-only Q0 selection
-> content-addressed selection freeze
-> selection hash replay
-> evaluation view opening
-> P1/P2/S1/S2/S3 unconditionally
-> machine-readable result freeze
-> scientific and report red teams
```

Primary targets remain `[1,2,3,5,6,7,8,9]`; target 4 is excluded. The grid is
`[1,2,4,8,16,32,FULL]`, MC count is 2,048, target is the scientific cluster,
and seeds are paired repeated training factors. The same-label oracle remains
unreachable.

## Next Authorization

The PI must issue a new direct authorization in the execution conversation
binding all four values:

```text
protocol commit e88a24484590636f87d0f22798401a762875046a
protocol SHA-256 2d72eb5119056a6520fd33fc0ac14ee6270bfd573b59c36b74be6aa3dc25fe39
analysis lock commit f19acd8775f9b0ddf60401739741bec0019d021c
analysis lock SHA-256 e18f2b5f1d79b6fcd96207339c5842e30b7aecb5bc22b8939a475487068b1b82
manifest digest 6180275dcef26bdda4ae4b291d1ef6dc83434462ecacee0350fa94ae9c6a7fef
```

That authorization may cover only the frozen seed3/seed4 existing-field C80E
analysis. It does not authorize training, forward/re-inference, GPU, target4,
oracle work, BNCI2014_004, active acquisition, seed5, C81, or manuscript
drafting.

## Final Readiness Verification

The C80R pre-execution red team passed `40/40`. Final regressions were run on
the exact clean commit `93d2099f14b8739089e640c0e6078f02ed5cc435`:

```text
focused job 894616: 53 passed
C65 job 894617:     368 passed, 1 conditional skip, 3 deselected
C23 job 894618:     775 passed, 1 conditional skip, 3 deselected
full job 894619:   1703 passed, 1 conditional skip, 3 deselected
```

All failures and stderr byte counts were zero. The skip is the finalized C78F
guard; the deselections are the three historical C79P preauthorization-state
tests. No C80R test or registry path was skipped. Earlier log-placement and
cumulative-glob issues are retained as closed, non-scientific repair events.

The operative final report is `C80R_PROTOCOL_READINESS.{md,json}`. It does not
change the protected state: no new C80E authorization record exists, no real
budget result exists, and `run-real` remains fail-closed.
