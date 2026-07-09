TTA_MECH_01S - Mechanism Synthesis

This is a read-only synthesis of TTA_MECH_01 outputs.
No new replay was run.
No baseline was added.
No new method was introduced.
No baseline was selected for deployment.
No P1/P2 training request is made here.

Status: SUBMITTED_FOR_PM_REVIEW.

Scope

```text
source_replay_commit: db57f6d
source_outputs_root: results/tta_mech/tta_mech01_bnci2014_001_seed0/
source_status: MECHANISM_INFORMATIVE_PASS
source_payload_hash: f8e99c99b9b61f66f0011da60620220021f861b4b409b8e46d6a781cc131a816
mechanism_matrix_rows: 10
uses_only_tta_mech01_outputs: true
new_replay_run: false
new_baseline_added: false
deployment_baseline_selected: false
```

Derived outputs

```text
results/tta_mech/tta_mech01_bnci2014_001_seed0/mechanism_matrix.csv
results/tta_mech/tta_mech01_bnci2014_001_seed0/mechanism_synthesis.json
```

Source coverage

```text
per_fold_metrics: 90 rows
audit_axes_table: 90 rows
aggregate_baseline_table: 10 rows
red_team_failures: 0
```

Mechanism conclusions

TTA_CONTROL_REPLAY is calibration-only in this replay. It does not change bAcc
on either backbone, changes no predictions relative to ERM, and reduces NLL/ECE
on both backbones. This is not an accuracy gain signal.

MATCHED_CORAL has a consistent geometry-alignment signal across backbones. It
improves bAcc on both backbones, but the calibration relation is
backbone-specific: EEGNetMini improves NLL/ECE, while EEGConformerMini improves
bAcc with worse NLL/ECE.

SPDIM has a consistent recentering / geometry signal across backbones. It
improves bAcc, NLL, and ECE on both backbones, but the frozen-feature replay
cannot test whether this reflects normalization or BN behavior upstream.

T3A is backbone-specific. EEGConformerMini loses bAcc and worsens calibration;
EEGNetMini gains bAcc but worsens NLL/ECE, so the EEGNetMini result is an
accuracy-calibration tradeoff rather than a deployment signal.

Unavailable axes

```text
source_replay_axis: NOT_AVAILABLE_IN_THIS_REPLAY
BN_axis: NOT_TESTED_IN_FROZEN_FEATURE_REPLAY
normalization_axis: NOT_AVAILABLE_IN_FROZEN_FEATURE_REPLAY
```

These unavailable axes are not negative results. They mean the frozen-feature
TTA_MECH_01 replay cannot identify source-retention or BN/normalization
mechanisms directly.

Cross-backbone assessment

| Baseline | Cross-backbone status | Synthesis |
| --- | --- | --- |
| ERM_NO_ADAPT | INCONCLUSIVE | Reference row; no adaptation mechanism inferred. |
| TTA_CONTROL_REPLAY | CONSISTENT_ACROSS_BACKBONES | Calibration-only; bAcc flat, NLL/ECE lower on both backbones. |
| MATCHED_CORAL | CONSISTENT_ACROSS_BACKBONES | Geometry signal consistent; calibration behavior backbone-specific. |
| SPDIM | CONSISTENT_ACROSS_BACKBONES | Recenter/geometry signal consistent; bAcc and calibration aligned. |
| T3A | BACKBONE_SPECIFIC | Classifier-template behavior differs by backbone. |

Portfolio recommendation

Recommend `TTA_MECH_02B_NORMALIZATION_BN_AUDIT` for PM consideration only.

Rationale: MATCHED_CORAL and SPDIM point to geometry / recentering effects, and
TTA_MECH_01 explicitly cannot test BN or normalization. This recommendation
does not open TTA_MECH_02 and does not authorize new replay, method work, P1/P2
training, or deployment selection.

Not recommended now:

```text
TTA_MECH_02A_SOURCE_REPLAY_ABLATION
CLOSE_AS_MECHANISM_BENCHMARK_ONLY
```

Reason: TTA_CONTROL_REPLAY does not show an accuracy gain that would make source
replay the immediate bottleneck, while the geometry/recentering signal remains
mechanistically open because BN/normalization is unavailable in frozen features.
