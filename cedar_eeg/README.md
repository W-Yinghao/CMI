# CEDAR-EEG

Conditional Evidence-Driven Artifact Removal for Privacy-Preserving and
Shift-Robust EEG Decoding.

This project is a scientific exploration package, not a paper-writing package.
The first phase is P0 frozen-latent surgery: train or load a normal EEG decoder,
audit where label-conditional domain information is extractable, and remove only
domain-rich / task-light latent components under explicit guards.

## Hard Boundary

CEDAR does not claim that lower leakage implies better target generalization.
Leakage is evidence for localization, not a deployed-loss certificate. If a
domain-rich component is also task-bearing, CEDAR must abstain and report the
atlas instead of forcing a deletion.

Do not use `oaci`, `h2cmi`, or FSR as CEDAR dependencies. Historical CMI losses
may be used only as comparators or archived context.

## Current Gate State

CEDAR source-only frozen-latent route is closed negative as of commit
`107192f`. CEDAR_01 completed the approved real EEG shadow audit on
BNCI2014_001 with EEGNetMini and EEGConformerMini feature dumps, and found
`0/54` accepted candidates under the frozen source-only contract.

Do not start P1/P2 from this branch. CEDAR artifacts are diagnostic-only.
Target diagnostics were quarantined and never used for selection.

Frozen state:

```text
CEDAR_01F_FEATURE_SUPPLY: PASS
CEDAR_01_REAL_SHADOW_AUDIT: COMPLETE_NEGATIVE
CEDAR_SOURCE_ONLY_LATENT_SURGERY_TO_P1: CLOSED_NEGATIVE
P1_CHANNEL_PRUNING: DENIED
P2_TTA_PRECONDITIONER: DENIED
DEPLOYABLE_MASK_ARTIFACT: FORBIDDEN
GENERALIZATION_OR_SAFETY_CLAIM: FORBIDDEN
CEDAR_RETAINED_ROLE: DIAGNOSTIC_ONLY / MEASUREMENT_TO_CONTROL_NEGATIVE_EVIDENCE
```

See:

- `cedar_eeg/reports/CEDAR_01N_NEGATIVE_CLOSEOUT.md`
- `cedar_eeg/reports/CEDAR_01N_FAILURE_TAXONOMY.md`
- `cedar_eeg/reports/CEDAR_01N_PM_DECISION.md`
- `cedar_eeg/reports/CEDAR_01_REAL_FROZEN_LATENT_READOUT.md`
- `cedar_eeg/reports/CEDAR_01_REAL_FROZEN_LATENT_PROTOCOL.md`
- `cedar_eeg/reports/CEDAR_01_ACCEPTANCE_CRITERIA.md`

## P0 / CEDAR_01 Scope

P0 operates on frozen feature arrays:

- `z`: source latent features, shape `[n, d]`
- `y`: task labels
- `domain` or `d`: subject/session/site labels
- required `groups`: recording/session groups for grouped cross-fit
- optional `z_target`, `y_target`: evaluation-only held-out target features

The runner never trains an EEG backbone. Real feature extraction and any
nontrivial audit must be submitted through Slurm from this login node. Missing
groups hard-fail unless `--allow-ungrouped-smoke` is explicitly used for a local
smoke test.

## P0 Gates

Continue only if all hold:

- conditional leakage drops by at least 30 percent
- source balanced accuracy drops by at most 1 point
- target balanced accuracy, if evaluated, is reported only as a continuation
  diagnostic and never used to select the mask
- R3 task reliance does not increase
- matched random-subspace control is approximately zero
- stability across folds/probes is acceptable

If leakage drops but task metrics do not improve, the valid direction is
privacy/compression/diagnostic evidence, not a generalization claim. In the
completed CEDAR_01 real EEG audit, no candidate crossed the actionability gate,
so the method pipeline stops here.

## Runner

Use a saved feature dump:

```bash
python -m cedar_eeg.runners.run_p0_frozen_latent \
  --feature-npz path/to/features.npz \
  --out results/cedar/p0_real_shadow/<run_id>/report.json
```

For real feature dumps, submit via Slurm instead of running the full audit on the
login node.

Run red-team validation before reporting a P0 result:

```bash
python -m cedar_eeg.runners.run_red_team \
  --p0-json results/cedar_p0/p0.json \
  --out results/cedar_p0/red_team.json
```
