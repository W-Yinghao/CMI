CEDAR_01F — Frozen Feature Supply / Provenance Gate

Status: approved next step after CEDAR_01 preflight. This is a feature supply
and provenance gate, not CEDAR_01 scientific execution.

Purpose

CEDAR_01 is blocked because no compliant real frozen-latent dump exists in the
active workspace. CEDAR_01F creates the supply contract needed before any real
shadow audit can run.

CEDAR_01F does not:

- run mask selection
- run CEDAR scientific readout
- train CEDAR method components
- authorize P1 / P2
- authorize generalization or safety claims

Route A: Read-only Inventory

Scan active workspace and allowed archive paths for `.npz` candidates. Inventory
records must include:

```text
dataset
backbone
fold
seed
path
has_z
has_y
has_domain
has_groups
has_role
has_subject
has_session_or_recording
n_samples
z_dim
status = COMPLIANT / ADAPTER_POSSIBLE / REJECT
reject_reason
```

Legacy archive candidates must carry:

```json
{
  "provenance": "legacy_archive_diagnostic_candidate",
  "cedar_role": "feature_supply_candidate_only",
  "deployable": false
}
```

Route B: Adapter Repair

Adapters are allowed only when groups can be deterministically reconstructed
from original subject/session/recording metadata. Disallowed:

- reconstructing groups from target labels
- reconstructing groups from outcomes or leakage results
- sample-order grouping
- random grouping

Any adapter output must include an adapter manifest and hash. Dumps without
deterministic group provenance remain REJECT.

Route C: Fresh Source-ERM Feature Dump Generation

If Route A/B find no compliant CEDAR_01 cell, a fresh source-ERM
feature-dump-only Slurm run is allowed.

Constraints:

- train only source ERM baseline backbones
- do not run CEDAR selector
- do not use target diagnostics to keep/drop dumps
- freeze folds, seeds, datasets, backbones before training
- hash dumps and write manifest before CEDAR_01 readout

Minimum fresh matrix:

```text
dataset: BNCI2014_001
backbones: EEGNetMini, EEGConformerMini
seed: 0
folds: full LOSO
roles: source_train / source_audit / target_audit
```

Compliant `.npz` Schema

Required:

```text
z              float32/float64 [n, d]
y              int64 [n]
domain         int64 or str [n]
groups         int64 or str [n]
role           str [n]
fold_id        scalar or str
dataset        scalar or str
backbone       scalar or str
seed           scalar or int
```

Preferred:

```text
sample_id
subject_id
session_id or recording_id
```

Hard Fail Conditions

- missing `z`, `y`, `domain`, or `groups`
- missing `role` for compliant direct loading
- length mismatch
- NaN / Inf in `z`
- single group
- single domain
- source-selection view has no source rows
- target role included in source-selection view
- `groups == sample_id` for every row without explicit grouping justification
- manifest hash mismatch at readout time

Loader Views

The loader must expose:

```text
source_selection_view
diagnostic_full_view
```

`source_selection_view` may contain only source rows and must not contain target
labels, target metrics placeholders, or role fields. `diagnostic_full_view` may
retain roles for audit and reporting.

CEDAR_01 Execution Gate

CEDAR_01 real readout may run only after CEDAR_01F PASS. Minimum entry:

- at least BNCI2014_001 x EEGNetMini x full LOSO seed0 compliant
- grouped cross-fit hard-fail enabled
- fixed candidates `drop_top_1`, `drop_top_2`, `drop_top_4`
- random-control metadata complete
- target perturbation invariance pass
- no deployable mask artifact emitted

P1/P2/generalization remain blocked after CEDAR_01F unless PM explicitly opens
the next gate.
