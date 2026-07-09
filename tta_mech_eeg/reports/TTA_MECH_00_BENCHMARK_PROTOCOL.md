TTA_MECH_00 - Benchmark Protocol

Status: benchmark design only. This protocol defines a future replay/audit
shape but does not run an experiment.

Purpose

TTA-MECH benchmarks existing target-unlabeled EEG adaptation baselines and
attributes their behavior to observable mechanisms. It does not introduce a new
adapter, objective, training phase, or deployment controller.

Approved input priority

Future TTA_MECH_01, if PM-approved, should prefer existing artifacts:

```text
CITA / TTA result summaries if present
CEDAR_01F frozen features if suitable for replay
existing source ERM feature dumps
existing baseline logs / manifests
```

TTA_MECH_00 does not approve new training and does not run real EEG workloads.

Frozen baseline universe

```text
ERM-no-adapt
TTA-Control replay
matched-CORAL
SPDIM
T3A
```

No baseline may be dynamically added after observing target metrics. Any future
change to the baseline universe requires PM approval before execution.

Audit axis A - entropy / confidence

Required observables:

```text
entropy_before_after
mean_max_probability
margin_shift
prediction_confidence_histogram
overconfidence_flag
```

Purpose:

```text
Separate useful confidence sharpening from degenerate entropy minimization.
```

Audit axis B - balance / prior

Required observables:

```text
predicted_class_marginal
source_label_prior
KL(mean_p_target || source_prior)
class_collapse_guard
single_class_dominance
```

Purpose:

```text
Identify whether gains are driven by target marginal shifts or by hidden class
reassignment errors.
```

Audit axis C - geometry

Required observables:

```text
feature_mean_shift
class_conditional_prototype_distance
covariance_distance
CORAL_distance
SPDIM_recentering_magnitude
within_class_vs_between_class_geometry_change
```

Purpose:

```text
Distinguish feature recentering / covariance alignment from logit-only or
entropy-only behavior.
```

Audit axis D - source replay

Required observables, only if existing implementation supports them:

```text
with_source_replay
without_source_replay
source_CE_retention
source_prediction_drift
target_metric_delta_from_replay
```

Purpose:

```text
Test whether source replay, not a named adaptation loss, is the stabilizer.
```

Audit axis E - normalization / BatchNorm

Required observables, only as audited replay if already supported:

```text
frozen_BN
adaptive_BN
feature_normalization_state
normalization_stat_shift
BN_or_norm_update_magnitude
```

Purpose:

```text
Detect whether normalization dynamics dominate the named algorithm.
```

No new BN method claim is allowed.

Audit axis F - calibration

Required observables:

```text
NLL
ECE
temperature_equivalent_effect
balanced_accuracy
bAcc_NLL_divergence
bAcc_ECE_divergence
```

Purpose:

```text
Separate target accuracy gain from calibration gain or degradation.
```

Target-label rule

Target labels may only be used after adaptation for final metrics and mechanism
stratification. They must not affect:

```text
baseline inclusion
variant selection
hyperparameter selection
thresholds
early stopping
reranking
fold inclusion / exclusion
failure taxonomy rescue
```

Required future red-team checks

```text
target-label noninterference
baseline universe freeze
no dynamic variant addition
no target-informed selection
manifest immutability
replay determinism
source / target role separation
no new method artifacts
```

Output shape for a future PM-approved run

```text
results/tta_mech/tta_mech01_<dataset>_<seed>/
  run_manifest.json
  baseline_universe.json
  artifact_manifest_validation.json
  per_fold_mechanism_table.csv
  aggregate_mechanism_table.csv
  red_team.json
  target_label_quarantine.json
  calibration_table.csv
  geometry_table.csv
  source_replay_table.csv
  normalization_table.csv
```

No model checkpoints, deployable adapters, masks, pruning artifacts, or new
method outputs should be emitted by TTA-MECH.
