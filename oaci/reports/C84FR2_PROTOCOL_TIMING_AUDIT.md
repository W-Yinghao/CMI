# C84FR2 Protocol Timing Audit

C84FR2 was designed after two preserved engineering-only failures. C84F job
`896185` froze the complete 1,944-unit model field and stopped while ordering
raw-file dictionaries. C84FR1 job `896550` then froze the complete label-free
118-subject, 9,621-row target trial registry and stopped during target
instrumentation when a NumPy float32 reconstruction exceeded the historical
`2e-5` functional gate.

At entry to this protocol:

```text
model retraining:       0
target-y operations:    0
target-label fields:    0
selector scores:        0
scientific metrics:     0
same-label oracle:      0
C84S:                   0
```

No target NPZ from job `896550` was opened before this protocol was written and
hashed. The failure description, counters, unit identity, and reported maximum
were replayed only from committed failure evidence. After the protocol commit,
the six immutable partial NPZ files may be inspected only for numerical-backend
calibration. Such inspection cannot read labels or compute scientific outcomes.

The repair does not widen `2e-5`. It replaces the cross-backend NumPy matmul
gate with a same-device GPU/PyTorch `torch.nn.functional.linear` identity gate
at `1e-6`, exact pre-write/post-reload dtype-shape-byte digests, and strict
saved-output replay at `1e-6`. CPU PyTorch, NumPy float32, and NumPy float64
reconstructions become finite-value diagnostics only.

Jobs `896185` and `896550`, their consumed authorizations, roots, logs,
attempt ledgers, partial manifests, and partial target artifacts remain
immutable. C84FR2 cannot reload target X, run forward, use a GPU, train, access
target labels, compute selector/scientific outputs, or start C84S. A future
target-only execution requires a new lock and fresh direct PI authorization.
