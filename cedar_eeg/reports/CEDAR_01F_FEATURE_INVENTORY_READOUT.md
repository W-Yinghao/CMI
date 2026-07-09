CEDAR_01F — Frozen Feature Supply / Provenance Gate Readout

Status: CEDAR_01F inventory complete, with Route C completion addendum. The
initial inventory found no pre-existing compliant BNCI2014_001 / BNCI2015_001
frozen feature dump. Route C subsequently produced 18 compliant BNCI2014_001
feature-supply artifacts under
`results/cedar/feature_supply/cedar01f_bnci2014_001_seed0`.

Scope

This readout is an inventory/provenance gate only. It does not run CEDAR
selection, does not produce a scientific CEDAR_01 result, and does not authorize
P1/P2/generalization claims.

Initial inventory command

```bash
PYTHONPATH=. python -m cedar_eeg.data.feature_inventory \
  --root results \
  --root archive \
  --include-archive \
  --csv-out /tmp/cedar01f_feature_inventory.csv \
  --json-out /tmp/cedar01f_feature_inventory.json
```

Initial inventory summary

```text
total .npz files inventoried: 236
COMPLIANT:                  0
ADAPTER_POSSIBLE:           41
REJECT:                     195
```

All initially inventoried `.npz` files came from legacy archive paths. No active
non-archive compliant CEDAR feature dump was found before Route C.

BNCI2014_001 / BNCI2015_001 status

Relevant BNCI2014_001 archive files were prediction archives, not frozen latent
feature dumps. They were rejected because they lack the required CEDAR_01F
source-selection schema.

Representative rejected rows:

```text
dataset        backbone  seed  status  reject_reason                path
BNCI2014_001   EEGNet    0     REJECT  missing_z_y_domain_groups    archive/lpc-cmi-failed/results/BNCI2014_001_EEGNet_cal_s0.preds.npz
BNCI2014_001   EEGNet    1     REJECT  missing_z_y_domain_groups    archive/lpc-cmi-failed/results/BNCI2014_001_EEGNet_cal_s1.preds.npz
BNCI2014_001   EEGNet    2     REJECT  missing_z_y_domain_groups    archive/lpc-cmi-failed/results/BNCI2014_001_EEGNet_cal_s2.preds.npz
BNCI2014_001   EEGNet          REJECT  missing_z_y_domain_groups    archive/lpc-cmi-failed/results/route2_BNCI2014_001_EEGNet.preds.npz
BNCI2014_001   TSMNet          REJECT  missing_z_y_domain_groups    archive/lpc-cmi-failed/results/route2_BNCI2014_001_TSMNet.preds.npz
```

No BNCI2015_001 CEDAR_01 candidate dump was found.

Adapter possible inventory

The 41 `ADAPTER_POSSIBLE` rows are legacy split feature dumps such as
`z_se/y_se` with subject or group metadata. They are not CEDAR_01 minimum cells:
they are PD/SCZ legacy archive candidates and carry legacy provenance only.

Representative rows:

```text
dataset       status             reject_reason                           path
PD_ds002778   ADAPTER_POSSIBLE   legacy_split_schema_requires_adapter_manifest  archive/lpc-cmi-failed/results/feat_dump_v3/audit_PD_ds002778_erm_0.npz
PD_ds004584   ADAPTER_POSSIBLE   legacy_split_schema_requires_adapter_manifest  archive/lpc-cmi-failed/results/feat_dump_v3/audit_PD_ds004584_erm_0.npz
SCZ_ds003944  ADAPTER_POSSIBLE   legacy_split_schema_requires_adapter_manifest  archive/lpc-cmi-failed/results/feat_dump_v3/audit_SCZ_ds003944_erm_0.npz
SCZ_ds004367  ADAPTER_POSSIBLE   legacy_split_schema_requires_adapter_manifest  archive/lpc-cmi-failed/results/feat_dump_v3/audit_SCZ_ds004367_erm_0.npz
```

Every legacy candidate is tagged by the inventory code as:

```json
{
  "provenance": "legacy_archive_diagnostic_candidate",
  "cedar_role": "feature_supply_candidate_only",
  "deployable": false
}
```

Schema / loader gate implemented

Added:

```text
cedar_eeg/data/feature_schema.py
cedar_eeg/data/feature_inventory.py
cedar_eeg/data/load_frozen_features.py
cedar_eeg/tests/test_feature_schema.py
scripts/cedar_feature_inventory.slurm
```

The strict loader hard-fails missing schema fields, length mismatch, NaN/Inf in
`z`, singleton groups/domains, sample-id grouping without justification, and
source-selection views with no source rows. It exposes:

```text
source_selection_view
diagnostic_full_view
```

The source-selection view contains only source rows and excludes target-role
rows and role metadata.

Validation

```text
PYTHONPATH=. pytest -q cedar_eeg/tests/test_p0_contracts.py cedar_eeg/tests/test_feature_schema.py
23 passed

python -m compileall -q cedar_eeg
passed
```

Route C completion addendum

Route C source-ERM feature supply was run after the initial inventory:

```text
job id: 890263
script: scripts/cedar_source_erm_feature_dump.slurm
dataset: BNCI2014_001
seed: 0
backbones: EEGNetMini, EEGConformerMini
output root: results/cedar/feature_supply/cedar01f_bnci2014_001_seed0
```

Post-completion inventory command:

```bash
PYTHONPATH=. python -m cedar_eeg.data.feature_inventory \
  --root results/cedar/feature_supply/cedar01f_bnci2014_001_seed0 \
  --json-out results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/feature_inventory.json
```

Post-completion inventory summary:

```text
total:            18
COMPLIANT:        18
ADAPTER_POSSIBLE: 0
REJECT:           0
```

Artifact coverage:

```text
EEGNetMini:        9/9 LOSO target folds
EEGConformerMini:  9/9 LOSO target folds
```

Strict validation artifacts:

```text
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/feature_inventory.json
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/schema_validation.json
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/manifest_freeze.json
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/source_selection_view_quarantine.json
```

Detailed completion provenance is recorded in:

```text
cedar_eeg/reports/CEDAR_01F_ROUTE_C_COMPLETION_READOUT.md
```

Gate outcome

CEDAR_01F inventory: PASS as supply/provenance inventory.

CEDAR_01F Route C feature supply: PASS as feature-supply completion.

CEDAR_01 real EEG shadow audit: still not run. It is now ready to request the
PM gate for real CEDAR_01 execution, but this readout does not itself authorize
selector execution.

Boundary remains unchanged: no selector, mask deployment, P1/P2, target
comparison, or target-generalization claim has been run.

Next permitted action

Commit the Route C completion package and request the CEDAR_01 real shadow audit
gate from PM.
