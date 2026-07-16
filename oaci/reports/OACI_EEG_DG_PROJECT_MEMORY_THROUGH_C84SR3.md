# OACI / EEG-DG Project Memory Through C84SR3

## Current Gate

```text
C84S_SECONDARY_Q0_AVAILABILITY_AND_ATOMIC_FAILURE_REPAIRED_V5_LOCK_READY_FOR_FRESH_PI_AUTHORIZATION
```

C84SR3 is complete readiness evidence, not a C84 scientific result. C84S V5
has no authorization record and has not reloaded target labels, accessed the
evaluation view, computed real selector scores or produced a scientific
taxonomy.

## Current Operative Identities

```text
C84F complete-field manifest:
  cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8

C84SR3 repair protocol commit:
  91f984503fa84b53fae32948d0cf49e7ede12b8f

C84SR3 repair protocol SHA-256:
  5c783db9113697b2c710af4c1f1bafd66a3096be7a1b5cbac8aa03ca2a9c3080

lock-bound implementation commit:
  815d0ccd3f2ef245ea66c734165905d3a08ac105

C84S V5 lock commit:
  2d03eb05e0cec352d08cdb6f48170be56876e77b

C84S V5 lock SHA-256:
  030be9c9ebac401ca9e7ae5e51bb1ce99b592faceac00fac8781070420b0b846

accepted regression commit:
  4cbe49d68b280d90c3b49ec82cbbbf9e8df95ed9
```

## Scientific State

The latest valid C84 object remains the label-free C84F field:

```text
datasets:                  Lee2019_MI / Cho2017 / PhysionetMI
candidate units:           1,944
levels:                    972 / 972
target subjects:           118
target trial rows:         9,621
target contexts:           944
candidate-context slices:  76,464
target artifacts:          1,944
context/digest sidecars:   1,944
```

It contains no target labels, selector scores or scientific results. The latest
completed scientific comparison remains C82-D on BNCI2014_001. C84 has not yet
established external validity.

## Preserved C84S Attempts

V3 job `897843` consumed its authorization, created physically separate label
views and failed before selector scoring because 243 reused C84C sidecars
omitted a later descriptor field. C84SR2 repaired that compatibility gap
additively and produced V4.

V4 job `898192` consumed its authorization and replayed immutable Stage A. It
failed before one complete selector context because Lee secondary Q0 B32 was
infeasible. NFS cleanup then masked the primary error. V4 protected counters:

```text
construction-label access:    1
evaluation-label access:      0
selector contexts:            0
scientific rows:              0
training / forward / GPU:     0 / 0 / 0
same-label oracle:            0
selection freeze published:  no
evaluation descriptor sealed: yes
```

Both failed roots, consumed authorization records, attempt ledgers and partial
evidence are immutable. Neither authorization can migrate to V5.

## C84SR3 Repair

The construction-only availability audit fixes the operative secondary grids:

```text
primary all datasets: [1,2,4,8,FULL]
Lee secondary:        [16]
Cho secondary:        [16,32]
Physionet secondary:  []
Lee B32:              input-unavailable, no selection/result row
```

Lee has exactly 25 construction labels/class for all 22 targets; Cho has 50;
Physionet ranges from 9 to 15. The primary grid and every scientific method,
formula, threshold, split, inference rule and taxonomy are unchanged. No
sampling with replacement, FULL substitution or target-specific budget repair
is allowed.

Stage B closes streams before cleanup, preserves the primary exception across
Python 3.9 and NFS cleanup failures, and never publishes a partial freeze. V5
uses a receipt-specific immutable Stage-A replay entrypoint and keeps the
evaluation descriptor sealed until complete Stage-B publication.

## Readiness Evidence

```text
real descriptor contexts:      944 / 944
candidates per context:          81 / 81
external objects rehashed:    7,776 / 7,776
external bytes rehashed:      48,072,941,176

synthetic contexts:              944
synthetic Q0 chains:           2,048
synthetic Q0 records:      8,750,000
synthetic final rows:          18,432
red team:                       51 / 51 PASS
```

Synthetic A/B/C/D/E and L1/L2/L3/L4 branches all passed. No precomputed final
rows were injected and no real field array or label was read by calibration.

Accepted regressions:

```text
focused:   367 passed
C65:       853 passed, 1 skipped, 3 deselected
C23:     1,264 passed, 1 skipped, 3 deselected
full:    2,188 passed, 1 skipped, 3 deselected
```

All accepted stderr files are empty. `squeue` was used; `sacct` was not.

## Authorization Boundary

The only next execution authorization phrase is:

```text
授权 C84S
```

It must be recorded against the V5 lock SHA above. C84SR3 does not authorize
C85, training, forward, GPU, new methods, retuning, same-label oracle access or
manuscript changes.
