# External Official SPDIM Repository Assessment

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

The login-node check reported CUDA unavailable, as expected; GPU execution
should use SLURM.

## Fit To H2CMI Protocol

The official demo is not a drop-in same-split baseline for H2CMI:

- Demo dataset: `BNCI2015_001`.
- Demo classes: `right_hand` vs `feet`.
- Demo preprocessing: `MotorImagery(fmin=4, fmax=36, tmin=1.0, tmax=4.0,
  resample=256)`.
- Provided pretrained session weights include a 13-channel first spatial layer,
  so they are not compatible with H2CMI W1/V2P tensors:
  - `BNCI2014_001`: 22 channels
  - `Lee2019_MI`: 62 channels
  - `BNCI2014_004`: 3 channels

Therefore the provided pretrained weights cannot be used for a fair H2CMI
head-to-head on the frozen H2CMI splits. A valid comparison would need to train
TSMNet source models from the same H2CMI source split and then apply the
official source-free SPDIM target adaptation on the same adaptation/evaluation
indices.

## Target-Label Use

The official demo constructs artificial label-shifted target sets with
`DomainDataset(X_test, y_test, ..., label_ratio=target_label_ratio)`, which uses
target labels for subsampling the demonstration target distribution.

For H2CMI, this label-based target subsampling must be disabled. The SPDIM
adaptation routines themselves can be run source-free/unlabeled: they optimize
information maximization over model predictions on target inputs. Target labels
should be used only after adaptation for balanced-accuracy evaluation.

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

## Verdict

Usable, but not drop-in.

The repository is sufficiently complete to attempt an official-code SPDIM
baseline for MI/W1-style binary H2CMI splits after writing an adapter that:

1. Builds official `TSMNet` models with per-dataset channel counts.
2. Trains source models on the exact frozen H2CMI source split.
3. Runs official source-free IM/geodesic or IM/bias adaptation on the H2CMI
   unlabeled adaptation batch only.
4. Evaluates once on the H2CMI evaluation batch with labels used only for
   metrics.
5. Records source seed, split IDs, runtime/failure status, and balanced
   accuracy with the same cluster bootstrap policy as the other MI results.

The repository does not directly solve Sleep/W2 because it provides a binary
motor-imagery TSMNet/SPD pipeline, not the H2CMI multiclass sleep representation.
