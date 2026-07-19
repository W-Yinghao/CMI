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

## Attempt 2

```text
direct authorization        : accepted (bound to commit b40e3b10, real F1/F2 implemented + frozen)
F0 bindings                 : PASS
engineering-only canary     : FAIL (no scientific outcome inspected)
  - CUDA                    : unavailable (device_count = 0); FAITHFUL production requires GPU
  - ds007221 BIDS root      : absent (/projects/EEG-foundation-model/yinghao/ds007221)
  - source/target MOABB data: absent (~/mne_data has only unrelated BNCI; no Lee2019_MI/Cho2017/
                              PhysionetMI/Brandl2020)
F1 executable               : fail-closed at the CUDA gate (C86EError) BEFORE any training / data access
stopped before              : EEG / label / GPU access (no CPU fallback for the frozen campaign)
scientific rows             : 0
field generation completed  : 0
confirmation executed        : 0
confirmation status         : NOT RUN
disposition                 : ENVIRONMENT_RESOURCE_AND_DATA_UNAVAILABLE (C86-E-class engineering
                              blocker; matches the pre-registered Option-C conditions "data not
                              available in the permitted environment" / "resources exceed the locked
                              envelope")
untouched population        : PRESERVED
authorization               : CONSUMED_BY_ATTEMPT_2 (does not migrate)
terminal stop rule          : NOT consumed (0 field generation, 0 confirmation)
```

This is NOT a measurement->control scientific result and NOT a fabricated field/prediction/gate.
The frozen code (b40e3b10) is authorization-ready and would run on a CUDA GPU node with the real
Brandl2020 / Lee2019_MI / Cho2017 / PhysionetMI (MOABB) data downloaded and the ds007221 BIDS tree
present at the bound root — none of which exist in this environment. Executing C86H requires
provisioning that environment; the code did the right thing by refusing before any data/GPU access.
