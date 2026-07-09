CEDAR_01F Route C — Source-ERM Feature Dump Plan

Status: prepared for Slurm execution. This is feature supply only.

Reason

CEDAR_01F inventory found:

```text
COMPLIANT:        0
ADAPTER_POSSIBLE: 41
REJECT:           195
```

The `ADAPTER_POSSIBLE` files are legacy PD/SCZ archive dumps, not the required
BNCI2014_001 CEDAR_01 minimum cells. Therefore Route C is the next permitted
action.

Frozen Route C Matrix

```text
dataset:   BNCI2014_001
seed:      0
folds:     full LOSO subjects 1-9
backbones: EEGNetMini, EEGConformerMini
roles:     source_train / source_audit / target_audit
```

Backbone mapping

```text
EEGNetMini        -> cmi EEGNet
EEGConformerMini  -> cmi EEGConformer
```

Window settings

```text
EEGNetMini:        tmin=0.5, tmax=3.5, resample=128
EEGConformerMini:  tmin=0.0, tmax=4.0, resample=250
```

Source split

For each LOSO fold, target subject rows are assigned `target_audit`. Non-target
rows are deterministically split by recording group into:

```text
source_train
source_audit
```

The split never uses target labels, target metrics, leakage results, or random
window grouping.

Outputs

All generated arrays are under ignored result paths:

```text
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/
```

Each fold output must include:

```text
*.npz
*.manifest.json
run_manifest.json
feature_dump_plan.json
```

The `.npz` file must satisfy the strict CEDAR_01F loader:

```text
z, y, domain, groups, role, dataset, backbone, seed, fold_id
sample_id, subject_id, session_id, recording_id
```

Non-goals

This job does not:

- run CEDAR selector
- run CEDAR_01 real shadow audit
- emit deployable masks
- train CEDAR method components
- authorize P1 or P2
- make target-generalization or safety claims

Slurm entry point

```bash
sbatch -p V100 scripts/cedar_source_erm_feature_dump.slurm
```

The Slurm array has two tasks:

```text
array 0: EEGNetMini
array 1: EEGConformerMini
```

After completion

Run CEDAR_01F inventory again over:

```text
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0
```

Only if compliant dumps are present and manifests hash-check should CEDAR_01
real frozen-latent shadow audit be requested.
