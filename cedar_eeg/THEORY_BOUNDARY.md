# CEDAR-EEG Theory Boundary

CEDAR starts from the measurement-to-control gap established by the previous
project line: detecting conditional domain leakage, difficulty, or shift does
not by itself imply stable control of deployed loss.

## Licensed Claims

- Conditional subject/session/site information can be audited in frozen EEG
  representations.
- Some structural units may be domain-rich and task-light under source guards.
- When deletion passes pre-registered guards, CEDAR can report lower extractable
  conditional domain information with non-inferior task performance.
- Structured sparsity is a valid deployment-facing property when measured
  directly.

## Forbidden Claims

- Do not say leakage reduction guarantees target generalization.
- Do not call CEDAR selection a safety gate.
- Do not select thresholds using target labels.
- Do not revive LPC-CMI, CITA-CMI, CMI-screened TTA, OACI, H2CMI, or FSR as a
  core dependency.
- Do not turn abstention into a hidden positive result.

## Method Boundary

CMI-style evidence is demoted from a training objective to a localization signal:

1. audit frozen representations;
2. propose candidate deletions;
3. pass source-risk, R3, random-control, rank/entropy, and label-balance guards;
4. delete only when the component is leakage-rich and task-light;
5. otherwise output an atlas and `ABSTAIN`.

P0 is mask-only. P1 structured pruning and P2 TTA preconditioning remain blocked
until P0 passes.
