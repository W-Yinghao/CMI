TTA_MECH_00 - Failure Priors

Status: expected failure modes for a future mechanism audit. This file does
not report experimental results.

Why failure priors are needed

TTA-MECH is an audit project, not a method project. A future benchmark may fail
to produce a clean mechanism explanation even if some target-unlabeled
adaptation baselines improve target metrics. These priors define expected
failure modes before any replay is run.

Expected failure modes

```text
dataset_backbone_specific_gain:
  Adaptation gain may exist only on specific datasets, backbones, seeds, or
  folds.

accuracy_calibration_tradeoff:
  bAcc gain may come with NLL / ECE degradation.

entropy_overconfidence:
  Entropy minimization may improve confidence while increasing overconfidence.

balance_wrong_reassignment:
  Balance KL or marginal matching may hide wrong class reassignment.

geometry_accuracy_calibration_split:
  Geometry alignment may help accuracy but hurt calibration.

neutral_t3a:
  T3A may be neutral or unstable on some EEG feature spaces.

coral_spdim_stronger_than_learned_adapters:
  SPDIM or matched-CORAL may tie or beat learned adapters.

source_replay_stabilizer:
  Source replay may be the real stabilizer behind apparent TTA gains.

normalization_dominance:
  Normalization or BatchNorm effects may dominate the named algorithm.

artifact_insufficiency:
  Existing artifacts may not contain enough state to identify the mechanism.

mechanism_nonidentifiability:
  Multiple observable mechanisms may fit the same target metric change.
```

What would count as useful negative evidence

```text
No single mechanism explains gains across backbones.
Accuracy and calibration move in opposite directions.
Target gain disappears without source replay.
Geometry metrics change but target performance does not.
Entropy sharpens while collapse risk rises.
Existing artifacts lack normalization or source-replay state.
```

What would not be a success

```text
cherry-picking one fold
claiming safety
claiming source-free deployment
claiming CMI worked
proposing a new adapter after TALOS failed
proposing pruning after CEDAR failed
relabeling boundary-hit TALOS behavior as acceptable
using target labels to choose baselines or thresholds
introducing privacy-head pruning or sparsity search
```

Expected interpretation discipline

If a future replay finds that matched-CORAL, SPDIM, T3A, or TTA-Control helps
on a subset of folds, TTA-MECH may describe where and how the observable
mechanism appears. It may not turn that observation into a new method claim.

If no mechanism is identifiable, that is an acceptable benchmark outcome:

```text
existing artifacts do not identify the mechanism behind historical TTA gains
```

That outcome should lead to a better audit design, not to a rescue method under
the TTA-MECH name.
