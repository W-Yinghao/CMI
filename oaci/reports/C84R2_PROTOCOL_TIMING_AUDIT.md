# C84R2 Protocol Timing Audit

## Chronology

```text
C84P blocked HEAD df95f1375f1883dd706a63f65ee9b6313fa1a779
  < C84R additive repair protocol 482a725abc6bf1f0e5d33be76ea17d37bcfaa6c3
  < C84C historical lock 4eaad36cafefb2645f1d5c6e393ae5a51ff33af9
  < C84R handoff 2fc5e797119ce1defc5e24c9063bb103b219a705
  < this C84R2 repair protocol
  < any C84R2 runtime implementation
  < future C84C execution lock V2
  < future direct PI authorization
  < first real dataset access
```

## Protected State

At protocol creation, the repository contains no C84C authorization record and
no C84F/C84S execution lock. The historical C84C V2 lock is preserved but is
insufficient and non-operative for execution.

```text
real EEG arrays loaded:        0
real labels read:              0
dataset downloads:             0
training/forward/GPU jobs:     0
candidate units created:       0
source-audit artifacts:        0
target-unlabeled artifacts:    0
target scientific metrics:     0
same-label oracle accesses:    0
```

The repair is prospective to all C84 real data and outcomes. It changes runtime
binding and engineering verification only. The 20-channel interface, subject
partitions, 1,944 candidate identities, method registry, budgets and scientific
inference remain unchanged.

## Historical Defects

The historical guard verifies protocol and lock hashes, authorization fields,
Git ancestry, cleanliness, Python version and Conda prefix. It does not replay
all lock-bound implementation/registry bytes, package versions or loader source
identities. The historical adapter also lacks exact returned-Epochs metadata
checks, strict-source audit instrumentation, saved-artifact replay and a
deterministic-prefix fingerprint.

No real-data access occurred before these defects were identified. C84R2 can
therefore make an additive prospective repair without outcome contamination.
