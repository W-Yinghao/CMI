CEDAR_01F Route C — Source-ERM Feature Dump Submission Readout

Status: Slurm submitted. Feature dump generation is running / queued; this is
not a completion readout.

Submitted job

```text
job id: 890263
partition: V100
script: scripts/cedar_source_erm_feature_dump.slurm
array 0: EEGNetMini
array 1: EEGConformerMini
```

Initial queue state

```text
890263_0  V100  cedar01f-srcerm  R   node42
890263_1  V100  cedar01f-srcerm  PD  QOSMaxGRESPerUser
```

Initial log check

`890263_0` started on a Tesla V100 and wrote:

```text
CEDAR_01F Route C source-ERM feature dump
backbone=EEGNetMini
out_dir=results/cedar/feature_supply/cedar01f_bnci2014_001_seed0
plan_hash=687902ede491d16c22eda3e1e9dc481b7bc0f679b01e401de32eb6fdc16511ae
n_items=9
```

The first error log check showed MOABB epoch warnings only, with no Python
traceback at submission time.

Outputs expected after completion

```text
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/
  feature_dump_plan.json
  run_manifest.json
  BNCI2014_001_EEGNetMini_seed0_<fold>.npz
  BNCI2014_001_EEGNetMini_seed0_<fold>.manifest.json
  BNCI2014_001_EEGConformerMini_seed0_<fold>.npz
  BNCI2014_001_EEGConformerMini_seed0_<fold>.manifest.json
```

Boundary

This job trains source ERM baselines only to supply frozen features. It does not
run CEDAR selector, CEDAR_01 scientific readout, P1 pruning, P2 TTA, deployment,
or target-generalization claims.

Next check

After Slurm completion, rerun the CEDAR_01F inventory on the output directory
and validate every manifest hash before requesting the CEDAR_01 real shadow
audit gate.
