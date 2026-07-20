# C81P Protocol Timing Audit

## Status

`C81P` is designed after the C79 and C80 outcomes and is prospective only to
the new baseline computations. It is an existing-field comparative audit, not
independent confirmation or external validation.

## Protected boundary

The accepted C80E base is commit `111df254dc562bc59342ad823cba3c8b79a64abe`.
Before this C81 protocol was hashed:

| Protected event | Count |
|---|---:|
| New real-field baseline score computations | 0 |
| New real candidate rankings | 0 |
| New baseline-specific evaluation-label reads | 0 |
| Same-label oracle access | 0 |
| Target-4 primary use | 0 |
| Training, forward, or re-inference jobs | 0 |
| GPU jobs | 0 |

Only committed reports, manifests, schemas, source code, public method
specifications, and C80 aggregate/frozen results were replayed. No external
array payload was loaded.

## C80 lock chronology

The C80 lock reported in the PM handoff at `f19acd8` is preserved as the first
complete real-adapter lock. Job `894641` then froze construction selections and
failed on a descriptor ABI check before opening the evaluation view. The
additive repair at `c19ef34`/`37e38d0` changed no scientific object and recorded
zero evaluation-label reads. Commit `0797599` is therefore the operative C80
lock for replay; its SHA-256 is
`2149895865bd44b4ab8358c76848bb6774abb59d4a203b261864be0ec599ff62`.
The historical lock remains recorded and is not rewritten.

## Lock order

The required prospective order is:

```text
C80E final base 111df25
  < C81 protocol and method-registry commit
  < C81 synthetic/schema implementation commit
  < C81 scope-specific analysis execution-lock commit
  < direct PI authorization
  < first new real-field baseline score
  < first baseline-specific evaluation-label read
```

The current milestone stops before direct PI authorization and before any
real-field baseline computation.
