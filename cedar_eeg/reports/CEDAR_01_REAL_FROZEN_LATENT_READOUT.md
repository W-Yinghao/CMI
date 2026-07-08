CEDAR_01 — Real-EEG Frozen-Latent Shadow Audit Readout

Status: NOT RUN. This is a preflight / contract update, not a scientific
readout.

What was completed

- CEDAR_01 protocol was added in
  `cedar_eeg/reports/CEDAR_01_REAL_FROZEN_LATENT_PROTOCOL.md`.
- CEDAR_01 acceptance criteria were added in
  `cedar_eeg/reports/CEDAR_01_ACCEPTANCE_CRITERIA.md`.
- Red-team validation now requires:
  - target perturbation invariance
  - complete candidate metadata
  - deterministic source-only tie-breaks
  - grouped split metadata for every candidate
  - per-candidate permutation null metadata
  - per-candidate random-control metadata
- The runner now defaults to the pre-registered CEDAR_01 candidate universe:
  `drop_top_1`, `drop_top_2`, `drop_top_4`.
- Grouped split labels now hard-fail by default. `--allow-ungrouped-smoke` is
  available only for local smoke tests.

Preflight result

No compliant real frozen-latent feature dump was found in the active non-archive
workspace.

Required schema:

```text
z or Z or features
y or labels
domain or d or domains
groups or recording or session
```

Required CEDAR_01 cells:

```text
BNCI2014_001 / EEGNetMini
BNCI2014_001 / EEGConformerMini
BNCI2015_001 / EEGNetMini       optional if dump exists
BNCI2015_001 / EEGConformerMini optional if dump exists
```

Observed active workspace feature-like arrays:

```text
results/cigl/smoke_edge_leakage_matrix.npy
results/cigl/smoke_node_leakage_map.npy
```

These are not compliant CEDAR_01 frozen feature dumps. Archive prediction files
and old LPC feature dumps were not used, because CEDAR_01 is constrained to
existing real frozen latent dumps for the approved datasets/backbones and must
not revive old failed-line artifacts as CEDAR evidence.

Validation completed

```text
PYTHONPATH=. pytest -q cedar_eeg/tests/test_p0_contracts.py
11 passed

python -m compileall -q cedar_eeg
passed

synthetic grouped smoke runner + red-team
passed / 11 checks / 0 warnings
```

Gate state after this preflight

- CEDAR_00: frozen as scaffold / red-team contract only.
- CEDAR_01 real EEG shadow audit: approved by PM but NOT RUN locally because
  compliant frozen dumps are absent from the active workspace.
- P1 channel pruning: still blocked.
- P2 TTA preconditioner: still blocked.
- Generalization / safety claim: forbidden.

Next executable requirement

Provide or generate, without training during the audit run, compliant frozen
feature dumps for BNCI2014_001 with EEGNetMini and EEGConformerMini. Once those
dumps exist, run CEDAR_01 through Slurm and produce compact
`results/cedar/p0_real_shadow/<run_id>/` artifacts.
