# C81R2 Selection-Descriptor Repair Red Team

## Result

```text
52 / 52 PASS
open blockers before reauthorization: 0
```

Selection job `894915` is retained as completed selection evidence. It wrote one
self-hashed manifest and one content-addressed payload, then the independent
freeze audit stopped because the generic C74 shard verifier requires every
array to share one first dimension. The registered C81 payload intentionally
has 32 context rows and 19 method identifiers.

The failure occurred before any evaluation-label descriptor was opened. Held-
evaluation statistics, oracle access, target-4 primary rows, training, forward,
re-inference, and GPU work remain zero.

The C81R2 repair is local to the C81 selection descriptor. It verifies the file
hash, byte size, exact field set, and the complete registered per-array shape
map. The shared C74 verifier is unchanged. The exact frozen manifest and payload
are bound into lock `f82ffa4`; score or rank recomputation is forbidden.

No method, representative, score formula, score direction, prior, temperature,
candidate universe, information view, Q1/Q2 margin, dependence procedure,
max-T family, LOTO rule, taxonomy, or report schema changed.

## Regression

```text
focused:    47 passed                           job 894924
C65-C81R2: 416 passed, 1 skip, 3 deselected    job 894922
C23-C81R2: 827 passed, 1 skip, 3 deselected    job 894923
full OACI: 1,751 passed, 1 skip, 3 deselected  job 894925
stderr:     0 bytes for all four jobs
```

The skip is the finalized C78F guard. The three deselections are the historical
C79P preauthorization-state tests. No C81 or C81R2 path was skipped.

The previous direct authorization is deliberately rejected by the new runtime
guard. Under policy `3d9dd76`, a new direct PI statement is sufficient; no
token or hash recital is required.
