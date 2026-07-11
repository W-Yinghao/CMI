# External Official SPDIM Repository Assessment

status: SUPPORTING_CURRENT_AFTER_P9

canonical_result: `spdim_w1_repaired_three_seed_results.csv`

Repository checked: `https://github.com/fightlesliefigt/SPDIM`

Checked revision: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`

Temporary checkout: `/tmp/spdim_check`

## Contents

The repository contains the official SPDIM/TSMNet code accompanying the ICLR
2025 paper *SPDIM: Source-Free Unsupervised Conditional and Label Shift
Adaptation in EEG*:

- `spdnets/`: SPDNet/TSMNet modules, SPD batch normalization, trainer, and IM
  adaptation routines.
- `demo.ipynb`: MOABB motor-imagery demo.
- `simulations.ipynb`: synthetic SPD demo.
- `pretrained_model/`: official pretrained weights for the demo protocol.
- `environment.yaml`: Python 3.9 environment with PyTorch, MOABB, geoopt.

No license file was present in the repository root at this checked revision.

## Environment Check

Using `/home/infres/yinwang/anaconda3/envs/icml/bin/python` with
`PYTHONPATH=/tmp/spdim_check`:

- `python -m py_compile /tmp/spdim_check/spdnets/*.py`: passed.
- Imports passed for `torch`, `moabb`, `mne`, `geoopt`, `sklearn`,
  `spdnets.models.TSMNet`, and `spdnets.trainer.Trainer`.

The assessment-time login-node check reported CUDA unavailable, as expected.
The later P4-P9 execution history and runtime environments are recorded in the
SPDIM result audits; no additional execution is required for the frozen result.

## Fit To H2CMI Protocol

The official demo was not a drop-in repaired-W1 baseline for H2CMI:

- Demo dataset: `BNCI2015_001`.
- Demo classes: `right_hand` vs `feet`.
- Demo preprocessing: `MotorImagery(fmin=4, fmax=36, tmin=1.0, tmax=4.0,
  resample=256)`.
- Provided pretrained session weights include a 13-channel first spatial layer,
  so they are not compatible with H2CMI W1/V2P tensors:
  - `BNCI2014_001`: 22 channels
  - `Lee2019_MI`: 62 channels
  - `BNCI2014_004`: 3 channels

Therefore the provided pretrained weights were not used. P9 resolved this
mapping by training TSMNet source models from scratch on the repaired-W1 LOSO
source subjects and applying official source-free SPDIM actions on the exact
frozen adaptation/evaluation indices.

## Target-Label Use

The official demo constructs artificial label-shifted target sets with
`DomainDataset(X_test, y_test, ..., label_ratio=target_label_ratio)`, which uses
target labels for subsampling the demonstration target distribution.

For the accepted P9 run, label-based target subsampling was disabled. The SPDIM
adaptation routines ran source-free/unlabeled and optimized information
maximization over model predictions. Target labels were used only after
adaptation for final metrics.

## H2CMI Tensor Smoke

A minimal no-training forward smoke on current H2CMI data passed:

```text
dataset: BNCI2014_004
input: (8, 3, 500)
model: official TSMNet instantiated with nclasses=2, nchannels=3, nsamples=500
output: (8, 2), all finite
```

H2CMI loaded tensor shapes for one subject:

```text
BNCI2014_001  (288, 22, 500), labels [0, 1], sessions [0, 1]
Lee2019_MI    (200, 62, 500), labels [0, 1], sessions [0, 1]
BNCI2014_004  (720, 3, 500),  labels [0, 1], sessions [0, 1, 2, 3, 4]
```

## Current Verdict

Usable, executed, and complete for the repaired-W1 MI baseline; still not a
drop-in pretrained baseline.

P9 implemented all five assessment-time requirements: dataset-specific TSMNet
construction, exact repaired source training, unlabeled official adaptation,
held-out evaluation, and full seed/split/runtime provenance. The final label is
`Official SPDIM W1 repaired-split three-source-seed same-split baseline`.

This does not make H2CMI versus SPDIM an adapter-only comparison. TSMNet and
H2CMI use different backbones, source objectives, feature spaces, and source
baselines. The external repository also does not directly solve Sleep/W2
because it provides a binary motor-imagery TSMNet/SPD pipeline rather than the
H2CMI multiclass sleep representation.
