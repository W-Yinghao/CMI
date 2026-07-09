TTA_MECH_00 - PM Boundaries

Status: binding PM contract for the TTA-MECH design line.

Project identity

```text
TTA-MECH is a diagnostic benchmark.
TTA-MECH is not a new method.
TTA-MECH is not TALOS_01.
TTA-MECH is not CEDAR_02.
TTA-MECH is not a CMI / CITA / CutClean / ACAR / TOS / OACI / h2cmi / FSR rescue.
```

Allowed work

```text
replay existing adaptation baselines
audit existing result summaries
compare entropy / balance / geometry / template / replay / normalization mechanisms
separate bAcc gain from NLL / ECE gain
write benchmark protocols
write red-team contracts
write mechanism taxonomies
```

Forbidden work

```text
No new method.
No new adapter.
No new objective.
No P1 request.
No P2 request.
No source-free deployment claim.
No target-label-based method selection.
No dynamic variant addition after target metrics.
No CMI objective or CMI-control rescue.
No CEDAR mask / pruning / surgery / graph surgery.
No TALOS-LR / TALOS-full / low-rank rescue.
No safety gate, harm router, or deployment controller.
No CutClean-style privacy-head pruning.
No MI-aware pruning objective.
No admissible sparsity grid.
No target-validation selection.
```

Inherited hard boundaries

```text
CEDAR_01N:
  source-only frozen-latent surgery route is closed negative
  0 / 54 ACCEPT on real EEG
  no mask / pruning / surgery continuation

TALOS_00B:
  low-degree-of-freedom adapter route is closed negative
  Conformer clean effect too small
  EEGNet gains boundary-hit dependent
  no TALOS-LR / TALOS-full / trust-region rescue

CMI / CIGL / CITA:
  CMI audit may remain diagnostic
  CMI control is not approved as a rescue mechanism
  CITA target gain is treated as non-CMI TTA-Control behavior

CutClean:
  privacy-head + MI-aware training + structured pruning + fine-tuning pipeline
  retained only as a boundary warning against pruning framing
```

Future experiment rule

Any future experiment must be a replay/audit with a frozen baseline universe.
It must use a predeclared manifest, predeclared baselines, and target-label
quarantine. Target labels may only be used after adaptation for final metrics
and mechanism stratification.

New method rule

Any new adapter, new objective, deployment controller, source-free method, or
safety gate requires a separate PM-approved project. It cannot be added to
TTA-MECH as a convenient extension.

Design package boundary

TTA_MECH_00 is docs-only:

```text
no real EEG workload
no result artifact
no new code path
no model checkpoint
no deployable output
```
