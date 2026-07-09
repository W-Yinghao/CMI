TTA_MECH_02B0 - Acceptance Criteria

PASS requires:

```text
condition registry frozen
artifact inventory complete
target-label quarantine contract pass
no-weight-update guard pass
BN/dropout/mutation guard defined
no real BN audit run
no target metrics computed
tests pass
compileall pass
git diff --check pass
```

READY_FOR_02B additionally requires:

```text
>= 1 backbone full LOSO has usable checkpoint + BN buffers + target X + source split
forward path available
model copy mutation can be isolated
dropout can be disabled
no target labels needed
```

NOT_FEASIBLE is the correct result when:

```text
checkpoint missing
BN buffers missing
target/source X missing
raw or preprocessed input missing
forward path unavailable
```

Current 02B0 result

```text
preflight_status: PASS
feasibility: TTA_MECH_02B_NOT_FEASIBLE_FROM_CURRENT_ARTIFACTS
ready_for_02b: false
ready_backbones: []
```

This meets 02B0 PASS but does not meet READY_FOR_02B.
