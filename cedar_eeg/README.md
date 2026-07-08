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

## P0 Scope

P0 operates on frozen feature arrays:

- `z`: source latent features, shape `[n, d]`
- `y`: task labels
- `domain` or `d`: subject/session/site labels
- optional `groups`: recording/session groups for grouped cross-fit
- optional `z_target`, `y_target`: evaluation-only held-out target features

The runner never trains an EEG backbone. Real feature extraction and any
nontrivial audit must be submitted through Slurm from this login node.

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
privacy/compression/diagnostic evidence, not a generalization claim.

## Runner

Use a saved feature dump:

```bash
python -m cedar_eeg.runners.run_p0_frozen_latent \
  --feature-npz path/to/features.npz \
  --out results/cedar_p0/p0.json
```

For real feature dumps, submit via Slurm instead of running the full audit on the
login node.

Run red-team validation before reporting a P0 result:

```bash
python -m cedar_eeg.runners.run_red_team \
  --p0-json results/cedar_p0/p0.json \
  --out results/cedar_p0/red_team.json
```
