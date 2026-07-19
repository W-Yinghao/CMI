# C86H attempt ledger

Durable record of C86H real-execution attempts. An attempt is a *scientific* confirmation
only if it generates a field AND opens held labels. Implementation-blocked attempts do NOT
consume the terminal stop rule (one field generation → one confirmation → one audit → stop).

## Attempt 1

```text
direct authorization        : accepted
F0 bindings                 : PASS
real field                  : absent
F1 executable               : unavailable (f1_train_zoo was a gated stub -> RuntimeError)
stopped before              : EEG / label / GPU access
scientific rows             : 0
field generation completed  : 0
confirmation executed       : 0
confirmation status         : NOT RUN
authorization               : CONSUMED_BY_FAILED_ATTEMPT_1
disposition                 : IMPLEMENTATION_BLOCKER (NOT a measurement->control result)
untouched population        : PRESERVED
```

This is not a 7th "measurement ≠ control" outcome and not a data/method scientific boundary;
the authorized entrypoint stopped by design at an unimplemented F1 stub. A fresh direct
`授权 C86H` is required after real F1/F2 are implemented + frozen (the authorization above does
not migrate to the modified code identity).
