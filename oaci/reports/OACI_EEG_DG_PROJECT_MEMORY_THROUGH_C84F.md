# OACI EEG-DG Project Memory Through C84F

## Scientific State

The latest valid scientific result remains C82-D: the frozen-field zero-label
comparison was heterogeneous by training seed, method identity and target
composition. C83 froze the single-dataset AAAI evidence contract. C84 is a
separate prospective external-validity program and has not yet produced a
scientific result.

## C84 Design

C84 uses three independent left/right motor-imagery cohorts: Lee2019_MI,
Cho2017 and PhysionetMI with subject 88 prospectively excluded. The exact common
interface is 20 physical channels, 160 Hz and a half-open [0,3) epoch with 480
samples. No interpolation, Fz substitution, zero filling or dataset-specific
mask is allowed.

Each dataset has disjoint 16-subject source panels A and B, each split into 12
training and four audit subjects. External seeds are 5 and 6. Levels are paired:
level 0 uses the full source panel, while level 1 removes the registered first
source subject's left-hand cell. Each dataset/panel/seed/level zoo contains one
ERM, 40 OACI and 40 SRC candidates.

## Engineering Chronology

- C84P stopped before data because Lee2019_MI lacked FCz.
- C84R prospectively adopted the exact 20-channel intersection.
- C84R2 made runtime lock-to-byte replay and complete canary instrumentation
  executable.
- C84C job `895441` validated 243 level-0 units.
- C84L1P prospectively defined the target-independent level-1 deletion.
- C84L1C job `896066` validated 243 level-1 units.
- C84FL2 locked the complete dual-level field implementation.
- C84F job `896185` trained/froze all 1,944 model units but stopped after target
  X access at a dictionary-order registry defect. Its model field remains valid.
- C84FR1 job `896550` froze the exact 118-subject/9,621-row label-free target
  registry and stopped after five complete target artifacts at a cross-backend
  NumPy float32 replay threshold. No target outcome was accessed.
- C84FR2 separated same-backend functional identity, exact persistence and
  cross-backend diagnostics without widening the historical threshold.

Historical failed jobs and consumed authorizations remain immutable. Their
partial target artifacts are evidence only and were never reused.

## C84F Complete Field

Direct authorization for the C84FR2 replacement was committed at `fc40914`.
Target-only job `897048` ran the locked V2 entrypoint on one V100 and completed
in 7,322.999 seconds. It reused the frozen model field without retraining and
replayed the exact target-input registry before candidate forward.

The atomic field contains:

```text
model units:                  1,944
target artifacts:             1,944
context/digest sidecars:      1,944
target subjects:                118
target contexts:                944
candidate-context slices:    76,464
canary witnesses:               486
```

Complete-manifest SHA-256:
`cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8`.
A separate post-execution replay verified every NPZ and sidecar hash, exact
directory membership and all registered counts.

Same-GPU/PyTorch direct classifier, saved Wz/logits, repeat logits and repeat z
all had maximum error zero. Saved softmax replay was
`1.1920928955078125e-07`, below `1e-6`. Finite CPU/NumPy differences are
diagnostic only.

## Protected State

The completed field records zero model retraining, zero target-y operations,
zero target-label fields, zero construction/evaluation labels, zero oracle
access, zero selector scores and zero scientific statistics. C84S is false and
has no execution lock.

## Verification

The final report red-team passed 68/68. Post-execution regression results were:

```text
focused:       30 passed
C65:          758 passed, 1 skip, 3 deselected
C23:        1,169 passed, 1 skip, 3 deselected
full OACI:  2,093 passed, 1 skip, 3 deselected
```

All regression stderr files are empty. The skip is the finalized C78F
conditional test, and the three deselections are the established C79
authorization-state fixtures.

## Next Boundary

C84F ends at
`C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED`.
It does not establish target accuracy, selector performance, Q1/Q2, a
label-budget frontier, level effects, same-method recurrence or external
validity. C84S requires a separately committed scientific adapter, analysis
lock, PM review and fresh direct PI authorization.
