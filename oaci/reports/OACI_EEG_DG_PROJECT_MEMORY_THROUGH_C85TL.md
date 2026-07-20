# OACI EEG-DG Project Memory Through C85TL

## Current State

```text
milestone:
  C85TL

gate:
  C85T_PROOF_AND_SYNTHETIC_EXECUTION_IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION

operative lock commit:
  9d414ebb889b2cfc3fefa19fa98d7ea5ca9fd691

operative lock SHA-256:
  4a289a46040b10855c6f23def53c328bdce0a8b1c71b7e90523887b6c1db7991

C85T authorized:
  false

C85E authorized:
  false
```

The current repository contains an executable, fail-closed proof/synthetic
pipeline but no C85T result.

## Scientific Line Preserved

C84S remains the only confirmatory multi-dataset result:

```text
primary:
  C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

label frontier:
  C84-L4
```

C84A is a read-only post-outcome audit. Its COTT average/tail, MaNo action
collapse, label-policy, and theory-gap summaries are exploratory and do not
change C84-D or C84-L4.

C85P prospectively separated:

```text
information experiment;
unrestricted decision value;
registered-policy value;
realized action dependence;
partial identification and minimax regret;
mean, worst-group, and CVaR risk;
near-optimal action geometry;
costly full-information label testing.
```

C85R repaired the V1 synthetic contract without executing it:

```text
S10:
  rich registered policy changed to always action 0

S9:
  exact four-action two-stratum Rademacher loss-vector law

S6/S7:
  iid Gaussian action errors at pairwise_sigma/sqrt(2)

T7:
  primary Delta_i union-bound target separated from looser diagnostic
```

The operative V2 contract SHA-256 is:

```text
e055c2a785374a3067ce90746a5941b39847b88a4f33e4ff8da5ca8adfde355a
```

## C85TL Contribution

C85TL fixed every implementation choice that could otherwise be invented after
seeing synthetic output:

```text
NumPy/PCG64DXSM source identity;
low64 SHA seed and little-endian conversion;
replicate/action/draw order;
float64 and exact-rational representations;
exact versus Monte Carlo authority;
S9 common-random-number coupling;
all-action S9 estimators and canonical selection;
S6/S7 and S9 MC uncertainty;
S8 rational LP certificate;
S5 candidate alpha-region proof target;
proof document schemas;
theorem-specific status transitions;
one coordinator;
runtime hash/blob replay;
attempt and failure ledgers;
atomic result publication.
```

## Environment

```text
prefix:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

Python:
  3.13.7

NumPy runtime:
  2.4.4

NumPy metadata first match:
  2.3.3

bit generator:
  PCG64DXSM
```

The dual NumPy dist-info state is explicitly bound. Eleven NumPy source,
binary, METADATA, and RECORD files must replay before future execution.

## Future Scenario Modes

```text
exact:
  S0 S1 S2 S3 S4 S5 S8 S10

exact plus 4,096 Monte Carlo replicates:
  S6 S7 S9
```

Exact output is authoritative wherever available.

S6/S7 draw one canonical Gaussian action vector per replicate. S9 draws 51 L
then 46 H Rademacher values from one generator and uses prefixes 51/13 and
18/46 for passive/Neyman. Monte Carlo chains are numerical replicates only.

## Proof Boundary

At C85TL completion:

```text
T1: OPEN
T2: OPEN
T3: OPEN
T4: OPEN
T5: OPEN
T6: OPEN
T7: OPEN

canonical proof artifacts:
  0

proof audits:
  0

status transitions:
  0
```

Future proof files require exact statements, assumptions, proof or
counterexample, boundary cases, independent red team, and final status.
Simulation and citation alone cannot create PROVED status. T5 may remain OPEN.

## Lock And Authorization

The lock binds 106 repository objects plus its runtime registry. Every bound
object has path, size, SHA-256, and Git blob identity.

Current status:

```text
LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED
```

No authorization record exists. The future shortest direct statement is:

```text
授权 C85T
```

The future record must bind both the lock SHA and actual lock commit discovered
from Git.

## Verification

Final-lock regression:

```text
focused: 348 passed
C65:     959 passed, 1 skipped, 3 deselected
C23:   1,370 passed, 1 skipped, 3 deselected
full:  2,294 passed, 1 skipped, 3 deselected
```

All accepted stderr files are empty. Final red team is 75/75 PASS. There are
no active C84/C85/OACI jobs according to `squeue`; `sacct` was not used.

## Zero Counters

```text
registered S0-S10 execution: 0
registered MC replicates:    0
real project data:           0
training/forward/GPU:        0 / 0 / 0
active acquisition:          0
C85T authorization:          0
C85E authorization:          0
manuscript work:             0
```

## Next Step

After fresh authorization, run only:

```text
python -m oaci.theory.c85t_execute run-locked \
  --execution-lock oaci/reports/C85T_EXECUTION_LOCK.json \
  --output-root <fresh-root>
```

A successful C85T stops at:

```text
C85T_DECISION_THEORY_PROOF_AUDIT_AND_SYNTHETIC_VALIDATION_COMPLETE_C85E_PROTOCOL_REVIEW_REQUIRED
```

It does not automatically authorize C85E, real data, active acquisition, new
data/model zoos, or manuscript changes.

