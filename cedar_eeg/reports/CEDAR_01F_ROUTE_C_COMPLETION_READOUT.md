CEDAR_01F Route C - Source-ERM Feature Dump Completion Readout

Status: PASS as CEDAR_01F feature-supply completion only. This is not a
CEDAR_01 scientific execution readout, not a selector result, and not evidence
for target generalization.

Scope boundary

This completion readout covers only Route C feature supply:

```text
dataset: BNCI2014_001
seed: 0
backbones: EEGNetMini, EEGConformerMini
fold universe: LOSO targets 1..9
roles per artifact: source_train, source_audit, target_audit
selector run: false
scientific readout run: false
deployable: false
```

Slurm provenance

```text
submitted job id: 890263
partition: V100
script: scripts/cedar_source_erm_feature_dump.slurm
array 0: EEGNetMini
array 1: EEGConformerMini
output root: results/cedar/feature_supply/cedar01f_bnci2014_001_seed0
```

`sacct` was unavailable on this cluster at completion time:

```text
Problem talking to the database: Connection refused
```

`scontrol show job 890263` retained the array-1 controller record:

```text
JobState=COMPLETED
ExitCode=0:0
RunTime=01:00:37
StartTime=2026-07-09T02:20:12
EndTime=2026-07-09T03:20:49
NodeList=node42
Command=/home/infres/yinwang/CMI_AAAI/scripts/cedar_source_erm_feature_dump.slurm
StdOut=/home/infres/yinwang/CMI_AAAI/logs/cedar01f-srcerm-890263_1.out
StdErr=/home/infres/yinwang/CMI_AAAI/logs/cedar01f-srcerm-890263_1.err
```

`scontrol show job 890263_0` no longer retained the array-0 controller record.
Array-0 provenance is therefore taken from its stdout/stderr logs and the 9
EEGNetMini artifact manifests. Both array stderr files contained MOABB and
Braindecode warnings only; no traceback, exception, Slurm failure, OOM, or
cancelled marker was found.

Log and runtime JSON hashes

```text
340f63f85b674d6be900de85a2c47c98fbd6def5100260f1e28433c3ec130ffd  logs/cedar01f-srcerm-890263_0.out
0cc457a18da0488a3336dcad567b0bbbe164c7345f43dd941067b3fa4d4689b3  logs/cedar01f-srcerm-890263_0.err
54eb06d2c63d9e90ab29bea517760cd55534768a70f2f5e4168cff1f7c3cf860  logs/cedar01f-srcerm-890263_1.out
72a39795f43db1de9588328b147f1045073e631d7fa1e9e09e939ee1f7d334ab  logs/cedar01f-srcerm-890263_1.err
bb8f6880f6df6a2819454c658b77b480d6664ff50088f586eef2e3f36185b6e5  feature_dump_plan.json
755bceaa030697227296200cdf1b6f9be57c6cc5a29c77986e3283eb1e878572  run_manifest.json
```

Plan and manifest freezing

The submitted Slurm array ran one backbone per array task. Each array emitted a
pre-execution plan hash in stdout:

```text
array 0  EEGNetMini          plan_hash=687902ede491d16c22eda3e1e9dc481b7bc0f679b01e401de32eb6fdc16511ae  n_items=9
array 1  EEGConformerMini    plan_hash=55a20d92d8173d2b302e5ef82554eb70d7d59ddc6d08b84b0e9cb17db2bbb8be  n_items=9
```

Each array also emitted a run manifest hash:

```text
array 0  EEGNetMini          run_manifest_hash=7819555e5d171528de68ce7f0ef87631f68c0414bad4ea5a981ec621b960e1d0
array 1  EEGConformerMini    run_manifest_hash=938e2b87d28a40e66852b3c2aad216d0fc16d1595047a7df14f3b49e62c5ae60
```

Important provenance note: `feature_dump_plan.json` and `run_manifest.json` are
shared output paths. Array 1 overwrote these files after array 0 completed. The
current files therefore represent the final EEGConformerMini array. Plan
freezing for this completion is evidenced by the per-array stdout plan hashes,
the committed Slurm script, and the per-artifact manifest hashes, not by a
single immutable shared plan file.

Output completion

The output root contains 18 frozen feature `.npz` artifacts and 18 matching
per-artifact manifest JSON files:

```text
EEGNetMini:        9/9 folds complete
EEGConformerMini:  9/9 folds complete
total:            18/18 artifacts complete
```

The `.npz` arrays are intentionally not committed. The small validation and
manifest JSON files are retained for review:

```text
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/feature_inventory.json
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/schema_validation.json
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/manifest_freeze.json
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/source_selection_view_quarantine.json
```

Inventory validation

Command:

```bash
PYTHONPATH=. python -m cedar_eeg.data.feature_inventory \
  --root results/cedar/feature_supply/cedar01f_bnci2014_001_seed0 \
  --json-out results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/feature_inventory.json
```

Result:

```text
total:            18
COMPLIANT:        18
ADAPTER_POSSIBLE: 0
REJECT:           0
```

Schema, manifest, and source-view validation

Command:

```bash
PYTHONPATH=. python -m cedar_eeg.data.validate_feature_supply \
  --root results/cedar/feature_supply/cedar01f_bnci2014_001_seed0 \
  --expected-count 18 \
  --expected-backbones EEGConformerMini EEGNetMini \
  --schema-out results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/schema_validation.json \
  --manifest-out results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/manifest_freeze.json \
  --source-view-out results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/source_selection_view_quarantine.json
```

Result:

```text
artifact_count: 18
backbones: EEGConformerMini, EEGNetMini
complete: true
errors: []
per-artifact status: 18 PASS
```

Per-backbone schema summary:

```text
EEGNetMini:        9 artifacts, z_dim=16, n_samples=5184 each
EEGConformerMini:  9 artifacts, z_dim=32, n_samples=5184 each
n_groups:          108 each
n_domains:         9 each
role_counts:       source_train=3696, source_audit=912, target_audit=576
source view:       4608 source rows, 96 source groups, 3 grouped folds feasible
```

No-selector proof

Validation confirmed the following for all 18 artifacts:

```text
selection_run=false
scientific_readout_run=false
deployable=false
cedar_role=feature_supply_candidate_only
forbidden selector keys present=[]
source_selection_view keys=[domain, groups, y, z]
target audit rows quarantined from source_selection_view
```

The job did not run CEDAR selector, candidate utility scoring, mask deployment,
P1/P2, target comparisons, or any target-generalization claim.

Test validation

```text
PYTHONPATH=. pytest -q cedar_eeg/tests/test_feature_schema.py cedar_eeg/tests/test_p0_contracts.py
23 passed
```

Gate outcome

CEDAR_01F Route C feature-supply completion: PASS.

CEDAR_01 real EEG shadow audit: still not run. The next permitted action is to
commit this completion package and request the CEDAR_01 real gate from PM.
