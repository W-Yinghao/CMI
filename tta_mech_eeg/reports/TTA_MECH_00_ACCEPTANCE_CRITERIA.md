TTA_MECH_00 - Acceptance Criteria

Status: design-package criteria only. TTA_MECH_00 does not include a method
PASS gate and does not authorize TTA_MECH_01 execution.

Why there is no method PASS

TTA-MECH is a benchmark / mechanism-audit project. Success is not defined as
beating ERM, TTA-Control, CORAL, SPDIM, or T3A. The goal is explanatory
coverage: which existing baseline gains, on which backbone/fold, and through
which observable mechanism.

TTA_MECH_00 PASS

TTA_MECH_00 passes only if all design conditions hold:

```text
docs complete
baseline universe frozen
no new method files
no real experiment run
target-label quarantine rules explicit
PM boundaries explicit
old CEDAR negative result encoded as a hard prohibition
old TALOS negative result encoded as a hard prohibition
CMI-control failure encoded as a hard prohibition
CutClean / pruning framing encoded as forbidden
```

TTA_MECH_00 FAIL

TTA_MECH_00 fails if any condition appears:

```text
new adapter introduced
new objective introduced
real EEG workload run
target labels permitted for selection
P1 / P2 requested
source-free deployment claim made
CEDAR / TALOS / CMI / CutClean rescue path opened
baseline universe left mutable
```

Future TTA_MECH_01 gate

TTA_MECH_01 is not approved by this document. It may run only after PM
approval and must be a replay/audit with a frozen baseline universe.

Required TTA_MECH_01 red-team checks:

```text
target-label noninterference
baseline universe freeze
no dynamic variant addition
no target-informed selection
manifest immutability
replay determinism
no new method artifact
```

Future success metric

TTA_MECH_01 success is explanatory, not competitive:

```text
Which existing baseline gains, on which backbone/fold, and through which
observable mechanism?
```

Acceptable future readout categories:

```text
entropy_confidence_gain
balance_prior_shift_gain
geometry_alignment_gain
classifier_template_adjustment_gain
source_replay_stabilized_gain
normalization_dominated_gain
calibration_only_gain
accuracy_calibration_tradeoff
no_identifiable_mechanism
```

Forbidden future claims

```text
No SOTA claim.
No new method claim.
No source-free deployment claim.
No safety or harm-control claim.
No CMI-control rescue claim.
No pruning / mask / surgery claim.
No claim that target-unlabeled adaptation is solved.
```
