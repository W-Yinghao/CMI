# C82E Pre-Adapter Scheduler Guard Repair

## Failed Attempt

Slurm job `895213` was rejected by the submission wrapper before the locked
Python entrypoint started:

```text
expected HEAD typed in wrapper:
  5644157d7e2ccddf1fc66b12cfc9c8a8c2af2927

actual committed HEAD:
  5644157ff20d519db37e5061f773875131453041

job state: FAILED
exit code: 1:0
runtime:   00:00:00
stdout:    0 bytes
stderr:    0 bytes
```

The failure occurred in the shell-level repository identity guard. The
`run-real` entrypoint was never invoked. The authorization-consumption marker
and external result directory remained absent.

## Scientific Boundary

```text
protocol changed:                       false
analysis lock changed:                  false
scientific implementation changed:      false
selection payload opened:               false
selection recomputed:                   false
target-evaluation descriptor opened:    false
target-evaluation labels read:          0
scientific rows computed:                0
authorization consumed:                 false
target4 primary rows:                    0
same-label-oracle accesses:              0
training/forward/re-inference/GPU:       0 / 0 / 0 / 0
outcome-dependent decision introduced:  false
```

## Additive Correction

The replacement submission changes only the wrapper's HEAD equality operand.
It will use the exact result of `git rev-parse HEAD` after this ledger is
committed and pushed. The operative C82 protocol, analysis lock, implementation,
frozen selection, views, estimands, thresholds, inference, output schemas, and
authorization record remain unchanged.

Because the locked entrypoint was never reached and authorization was not
consumed, the existing direct C82E authorization remains active for the first
actual adapter invocation. Job `895213` is preserved permanently in the
execution-attempt ledger.
