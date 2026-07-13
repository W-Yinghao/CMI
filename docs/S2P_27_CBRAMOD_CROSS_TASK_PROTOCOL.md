# S2P_27 - CBraMod Cross-Task Frozen-Representation Protocol

**Phase:** C0-C2. **Status:** FROZEN BEFORE SCIENTIFIC COMPUTE. **Project:** OPEN.

## Scientific question

Phase C tests whether the Route-B pattern observed on FACED is specific to that dataset, general within affective
EEG, or directionally present in a different task family. It adds two external validations:

```text
SEED-V:   affective-domain validation with trial/session structure
ISRUC_S3: cross-task sleep-staging validation with subject-wise rotations
```

The independent hypotheses are:

1. subject-predictive NLL improves from random initialization to H200 and may continue improving at higher
   budgets;
2. equal-rank subject-task projection overlap decreases from H200 to the pooled higher budgets;
3. task-gated subject-subspace erasure does not exceed a source-validation-variance-matched random-subspace null;
4. target Cohen's Kappa and target NLL either agree or expose endpoint discordance.

FACED remains the Phase-A/B anchor and is not recomputed in Phase C. No p-value is pooled across datasets.

## Scope and exclusions

Phase C reads exactly the ten Phase-B representation contracts in the immutable closure manifest:

```text
random, released,
H200_s0, H200_s1,
H500_s0, H500_s1,
H1000_s0, H1000_s1,
H2000_s0, H2000_s1
```

The random object is the frozen deterministic initialization contract. Every physical checkpoint is read directly
from its content-addressed, read-only path. Its SHA256 is checked before and after feature extraction; strict state
loading is mandatory. Mutable source paths, `best.pt`, `latest.pt`, symlinks, and checkpoint reselection are
forbidden.

The encoder is always frozen. Phase C contains no CBraMod pretraining, encoder fine-tuning, H4000, CodeBrain
Stage-2 training, frequency-only workaround, B2 layer search, fourth downstream dataset, or manuscript work.
Training the pre-registered downstream probe or ISRUC sequence head is not encoder fine-tuning.

## Common representation contract

All samples are resampled or loaded at 200 Hz by the authoritative dataset asset contract. For each channel and
each one-second patch, input is normalized as:

```text
x_norm = (x - patch_mean) / (patch_std + 1e-6)
```

No dataset-wide, target-dependent, or CodeBrain `/100` normalization is used. For a tensor with `C` native
channels and `P` one-second patches, the frozen CBraMod feature is:

```python
patch = model.patch_embedding(x, None)
encoded = model.encoder(patch)
feature = encoded.mean(dim=2).reshape(batch, -1)  # C x 200
```

CBraMod's runtime channel dimension permits native62 SEED-V and native6 ISRUC input, but this does not license a
channel-invariance claim. Each dataset has one pinned channel order and one channel-order hash. Multiple orders,
implicit interpolation, zero-padding, or unrecorded reordering are stop conditions.

PCA is fit only on the applicable source-fit partition, uses 128 components with whitening, and has random seed
0. Fixed logistic probes use `C=1`, `lbfgs`, tolerance `1e-6`, and `max_iter=2000`. No object-specific or
dataset-test-specific tuning is allowed.

## C1 - SEED-V contract

### Assets and split

The authoritative processed dataset contains 16 subjects, three sessions per subject except one unavailable
subject-session (47 subject-sessions total), 15 trials per session, native 62-channel one-second windows, and five
classes. Keys must parse as `record-trial-window` and labels must be constant within a trial.

The split is fixed within every subject-session:

```text
train: trials 0-4
val:   trials 5-9
test:  trials 10-14
```

This is a trial-held-out, same-subject protocol. It is not an unseen-subject replication.

### Analysis unit

One-second windows are never treated as independent observations. Frozen window features are averaged within
each original trial before PCA, probe fitting, geometry, scoring, and uncertainty calculations. This gives every
trial equal weight regardless of recording duration. Subject is the biological bootstrap cluster and session is
nested within subject.

### Subject metric

The primary subject endpoint is group-cross-fitted 16-way subject-probe NLL. The three frozen trial blocks are
the cross-fit folds: fit on two of train/val/test trial blocks and score the held-out block, with every trial kept
intact. Secondary endpoints are pairwise subject AUC and standardized margin. Task labels are not required by the
subject probe.

### Task metric

Fit PCA128 and a fixed five-class logistic probe on train-trial means, use val trials only for the pre-registered
task gate, and score test trials once. The prospective primary endpoint is test Cohen's Kappa. Secondary endpoints
are test NLL, balanced accuracy, weighted F1, and source-val Kappa/NLL.

### Geometry

Fit geometry on train-trial means and evaluate captured energy on val-trial means. Subject effects are
class-centered subject means; task effects are subject-centered class means. Both subspaces have fixed rank
`K-1=4`. The primary statistic is normalized projection overlap:

```text
trace(P_subject P_task) / 4
```

An overlap is uninterpretable if held-out self-captured energy for either fitted subspace is below 0.05.

### Gate-first sequence

The gate evaluates `random`, `released`, `H200_s0`, `H1000_s0`, and `H2000_s0`. It passes only if:

```text
released test Kappa >= random test Kappa + 0.02
all trial groups are disjoint across train/val/test
feature extraction is exactly deterministic
all fitting and selection remain source-only
target-label firewall is clean
```

If the gate passes, the fleet adds H500 and all second pretraining seeds. Gate objects are recomputed in the same
fleet implementation; gate values are not copied into a different estimator. A failed gate closes SEED-V without
substituting another split or dataset.

## C2 - ISRUC_S3 contract

### Assets and rotations

The authoritative Cohort-III asset contains ten subjects, 8,500 labeled 30-second epochs, and 425 non-overlapping
20-epoch sequences. Each sequence has shape `[20,6,6000]` at 200 Hz and one label in `{0,1,2,3,4}` per epoch.
Native channel order and the processed-tree SHA256 are inherited from the accepted ISRUC recovery contract.

The ten rotations are fixed. In rotation `r`, one subject is validation, the next subject is test, and the other
eight are source-train, exactly as recorded in the accepted split manifest. Every subject is test once. No
sequence crosses subjects or the non-overlapping 20-epoch boundary.

### Frozen sequence-aware head

The CBraMod encoder produces one `6 x 200 = 1200` feature per 30-second epoch. The downstream head is fixed before
results:

```text
Linear(1200, 512) + GELU
one TransformerEncoderLayer(d_model=512, nhead=4,
                            dim_feedforward=2048,
                            batch_first=true, norm_first=true)
Linear(512, 5) at every epoch
```

Only this head is optimized. Use AdamW, learning rate `1e-4`, weight decay `5e-4`, label smoothing `0.1`, batch
size 16 sequences, 50 epochs, and downstream seeds `{0,1,2}`. Select the head epoch by aggregate validation
subject Kappa only; ties choose the earliest epoch. Encoder parameters remain frozen and must have no gradients.

The primary result for an object averages the three downstream-seed predictions within each rotation before
computing the concatenated ten-test-subject endpoint. Per-subject and per-head-seed results are retained.

### Subject metric

The primary subject endpoint is cross-fitted subject-probe NLL on the eight source-train subjects of each
rotation. Every subject's chronological sequence list is split into two contiguous halves; fit on one half and
score the other, then swap. Sequences remain intact. Secondary endpoints are pairwise AUC and standardized
margin. Rotation-level estimates are summarized without treating repeated source-subject appearances as
independent biological units.

### Task metric and geometry

The prospective primary task endpoint is Cohen's Kappa over all epochs from the ten rotating test subjects.
Secondary endpoints are NLL, balanced accuracy, weighted F1, and per-test-subject values.

Geometry is source-only and independent of the sequence-head test result. Within each rotation, fit rank-4
subject and rank-4 task effect subspaces on the first contiguous half of source-train sequences and evaluate
captured energy on the second half; then swap and average. Absolute overlap values are interpreted only within
ISRUC.

### Gate-first sequence

The gate evaluates `random`, `released`, `H200_s0`, `H1000_s0`, and `H2000_s0` under all ten rotations and all
three fixed downstream seeds. It passes only if:

```text
released aggregate test Kappa >= random aggregate test Kappa + 0.02
all ten rotating 8:1:1 subject splits are exact
all 20-epoch sequences and chronological indices are valid
released and random endpoints are finite and off the degenerate one-class floor
feature extraction is exactly deterministic
target-label firewall is clean
```

If the gate fails, ISRUC closes as an underpowered or invalid path. ISRUC_S1, flat-epoch classification, sequence
overlap, and a post-result head change are forbidden. If it passes, the same implementation runs the full ten
objects.

## Functional reliance

For checkpoints that pass the dataset-specific task gate, Phase C computes subject-subspace erasure and compares
it with 200 random orthobases whose removed source-validation feature energy is matched exactly, including a
partial final direction. The task gate is:

```text
source-val Kappa >= 0.05
and source-val Kappa >= random source-val Kappa + 0.02
```

Subject directions are fit using source data only. Random-null energy matching uses source validation only. Test
labels are used only to score the already frozen subject and null interventions. Holm correction covers all
task-gated pretrained cells within each dataset; datasets are not pooled.

SEED-V L5 is a cohort/session reliance diagnostic, not unseen-subject reliance. ISRUC reports every test subject
and is explicitly low-power. Failure to exceed the matched null means only that this measured linear subject
subspace is not a detectable task-specific lever.

## Confirmatory contrasts and uncertainty

Within each dataset, pre-registered contrasts are:

```text
C1: random - mean(H200 seeds) subject NLL
C2: mean(high-budget overlap) - mean(H200 overlap)
    where high budgets are H500, H1000, H2000 with equal budget weight
C3: task-gated L5 subject intervention vs matched random-null family
C4: budget-specific Kappa and NLL response, reported separately
```

SEED-V uses 5,000 subject-cluster bootstrap replicates with session and trial retained inside sampled subjects.
ISRUC uses the ten rotating test subjects as the exact biological units and reports leave-one-test-subject-out
sensitivity; it does not claim high-powered significance. Pretraining seeds are averaged within budget, never
selected by best performance. Holm correction is applied within a dataset's pre-registered family.

No monotonic scaling law, general EEG p-value, subject-invariance claim, or causal reliance claim is licensed.

## Outputs and launch boundary

C0 writes protocol/data/provenance checks only. C1 and C2 write separate gate packages. Full-fleet outputs are
allowed only after the corresponding gate passes.

```text
results/s2p_route_b_cross_task_c0/
results/s2p_route_b_cross_task_seedv/
results/s2p_route_b_cross_task_isruc_s3/
```

No scientific compute may auto-launch from C0. Gate failure is binding for that dataset. Gate pass may launch only
the corresponding frozen-readout fleet; it cannot launch training, fine-tuning, another dataset, or Phase B2.
