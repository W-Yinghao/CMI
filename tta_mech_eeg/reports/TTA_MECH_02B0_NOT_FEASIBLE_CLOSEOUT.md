TTA_MECH_02B0N - Not-Feasible Closeout

TTA_MECH_02B0 BN / normalization preflight is accepted as PASS.
TTA_MECH_02B real audit is DENIED because it is not feasible from the current
artifact set.

Accepted preflight commit

```text
5b9b0f8 Add TTA-MECH_02B0 BN preflight
```

Gate summary

```text
TTA_MECH_02B0_PREFLIGHT: PASS
red-team failures: 0
real forward run: false
BN refresh run: false
target metrics computed: false
new baseline/method: false
feasibility: TTA_MECH_02B_NOT_FEASIBLE_FROM_CURRENT_ARTIFACTS
ready_for_02b: false
ready_backbones: []
```

Primary hashes

```text
condition_registry_hash: 3909b80268adbd88e666f1cc4cd2810ae3d91ba82eb65b97596ec3312e902d6e
bn_artifact_inventory_hash: 54c9f7cbcc345991f7b4044deefb1b66a7e4da5ddfbf82628020ea0cfdb52656
preflight_payload_hash: 203ca6ee1df1b56004c06729c59bc4b051fa0008bab0461bce7bad4ac8abac27
```

Artifact conclusion

```text
18/18 CEDAR_01F feature artifacts hash matched
0/18 READY
0/9 EEGConformerMini folds READY
0/9 EEGNetMini folds READY
```

The current artifacts support frozen-feature replay and mechanism benchmarking.
They do not support BN-state or normalization causal audit because the required
checkpoint, BN-buffer, raw/preprocessed input, and forward-path artifacts are
absent.

Supported by current artifacts

```text
frozen-feature replay
baseline mechanism axes
geometry / recentering audit
calibration replay
class-balance / entropy audit
```

Not supported by current artifacts

```text
BN-state causal audit
normalization forward audit
source replay causal ablation
raw model TTA dynamics
checkpoint-level reproducibility
```

Frozen prohibitions

```text
No TTA_MECH_02B real audit from current artifacts.
No new artifact generation under TTA_MECH_02B.
No checkpoint reconstruction.
No raw EEG forward reconstruction.
No BN refresh.
No target-metric computation.
No new method or baseline.
No CEDAR/TALOS/CMI/CutClean rescue.
```

Future work boundary

A future BN / normalization audit would require a separate PM-approved artifact
acquisition protocol. That would be a new project phase, not a continuation of
02B0. No such artifact preflight is approved here.
