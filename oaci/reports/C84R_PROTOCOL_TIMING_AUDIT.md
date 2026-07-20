# C84R Protocol Timing Audit

The accepted C84P HEAD is `df95f1375f1883dd706a63f65ee9b6313fa1a779`. Its committed final red team records zero real EEG arrays, labels, downloads, training/forward runs, GPU jobs, candidate units and selector outcomes.

The C84R repair is availability-only and was instantiated before any real C84 adapter. It drops FCz from every dataset, uses no Fz substitution or interpolation, and changes no subject partition, method, candidate count, budget or inference rule.

The four historical 21-channel hashes remain content-valid, preserved and non-operative. This repair protocol must be committed before the C84C real adapter and V2 execution lock.

```text
real EEG access before repair:       0
real label access before repair:     0
outcome-dependent decisions:         0
history rewritten:                    0
active C84C/F/S authorization:        0
```
