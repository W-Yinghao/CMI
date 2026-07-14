# S2P_35 - FMScope-FSR Diagnostic-to-Deployment Bridge Protocol

**Status:** PANEL-1 PROTOCOL FROZEN / NO FOUNDATION TRAINING / NO FINE-TUNING.

This is an experimental protocol, not manuscript text. It does not authorize Phase-D1 training, H4000,
CodeBrain Stage-2, layerwise search, aperiodic ablation, or any new downstream dataset.

## Scientific object

FMScope and FSR ask different intervention questions:

```text
FMScope-style diagnostic:
  fit a subject eraser with the complete cohort, transform the complete cohort,
  and ask whether a newly trained label head benefits.

FSR deployment intervention:
  fit a subject eraser with source subjects only, deploy it to unseen subjects,
  and separately ask whether a new head benefits and whether the original head relied on it.
```

The bridge estimates the gap between these information regimes on identical frozen features, outer folds,
task-head family, and preprocessing. It also separates identity-specific removal from generic dimensionality or
conditioning effects.

## External authority and assets

The exact-replication authority is the public FMScope repository:

```text
repository: https://github.com/Jimmy110101013/fmscope
commit: 09885016a00db6c7de0074304c455c50685100c9
paper: arXiv:2606.06647v1
```

Panel 1 uses only the bundled public per-window caches:

| Cell | Cache | SHA256 | Shape |
| --- | --- | --- | --- |
| EEGMAT x released CBraMod | `reproduction/data/features_cache/frozen_cbramod_eegmat_perwindow.npz` | `b4ed9917eeb9cac2eaea911903700da7ce269c40ebb53d0039e93d88403875bc` | `[1707, 200]` |
| SleepDep x released CBraMod | `reproduction/data/features_cache/frozen_cbramod_sleepdep_perwindow.npz` | `da8280e0a469f41c65cea97572dd37e6bd2fd104c05a83d49b26684645a2b091` | `[4207, 200]` |

Each cache must contain explicit window-to-recording, recording-label, and recording-subject arrays. Neither cache
is copied into this repository. The external commit and payload hashes are mandatory inputs to every run.

## Panel 1 cells

```text
Primary positive-reference cell:
  EEGMAT x released CBraMod

No-consensus negative-control cell:
  SleepDep x released CBraMod
```

Both have 36 subjects and two recordings per subject. The task label varies within subject.

## B0 exact-replication gate

B0 calls the pinned public `subject_axis_erasure` implementation on the public cache without changing its
defaults. This reproduces the public builder's cohort-global LEACE contract:

```text
eraser fit:
  all feature windows + all subject IDs

label evaluation:
  5-fold StratifiedGroupKFold at recording level, groups=subject

head:
  train-fit 1st/99th-percentile clipping
  StandardScaler
  LogisticRegression(liblinear, C=1, class_weight=balanced)

label seeds:
  42, 123, 2024
```

The gate passes only if EEGMAT has rank 35, post-erasure linear subject BA at most `chance + 0.01`, finite
nonlinear residual subject BA, an interpretable pre-erasure label probe, positive label BA delta, and a live delta
within 0.01 of the bundled historical CBraMod/EEGMAT delta (`0.0601851852`). SleepDep must execute under the same
contract, but its label delta is a negative-control observation and has no required sign.

The exact public builder and bundled historical aggregate are recorded separately. A drift between them is never
silently resolved by selecting the more favorable number.

## Unified bridge folds and head

The bridge uses the cache's explicit recording arrays. For each seed in `{42,123,2024}`:

```text
outer split:
  StratifiedGroupKFold(n_splits=5, shuffle=true, random_state=seed)

group:
  subject ID

fit samples:
  source-recording windows only

score unit:
  recording; test-window probabilities are mean-pooled per recording
```

The head is the public FMScope linear head with one prospective determinism repair: liblinear `random_state` is
pinned to the outer seed. All four removal protocols use exactly the same fold and head implementation.

## Four removal protocols

### A. Global-oracle subject LEACE

LEACE whitening and the subject cross-covariance are fit once with all cohort windows and subject IDs. The same
operator transforms source and held-out windows. A fresh task head is then fit within each outer source fold.
This is a transductive cohort diagnostic and is not deployable.

### B. Fold-wise source-only subject LEACE

Within every outer fold, LEACE is fit only with source windows and source subject IDs. Its source-fitted mean and
operator transform both source and held-out windows. No held-out feature or subject ID enters the eraser fit.

### C. Global same-rank random removal

For each of 100 preregistered draws, use the global whitening operators from A and replace the whitened subject
basis by a Haar-random orthonormal basis of the same rank. Apply the resulting LEACE-form oblique operator to the
whole cohort, then evaluate with the same outer folds.

### D. Fold-wise source-only same-rank random removal

For each outer fold and each of 100 preregistered draws, use the source-only whitening operators from B and a
Haar-random whitened basis with exactly the source subject-axis rank. Fit and score only within that fold.

Random seeds are SHA256-domain-separated by dataset, outer seed, fold, information regime, and draw. Results are
paired at all those levels. There is no best-draw selection.

Empirical random-direction tail probabilities are Holm-corrected across the complete eight-cell bridge family
(`2 datasets x 2 information regimes x 2 endpoint families`). Same-rank and variance-matched p-values are corrected
as separate null families. An identity-specific claim requires both adjusted values to be at most 0.05.

## Removed-variance-matched sensitivity

Same-rank removal is the primary null. A secondary null uses a random Euclidean orthobasis traversed until its
source-fit removed squared feature energy equals the corresponding LEACE removed energy. The last direction is
partially removed to achieve an absolute match error at most `1e-10`. This sensitivity changes effective rank and
is not allowed to replace the same-rank primary null.

An identity-specific utility claim requires the subject-removal effect to exceed unchanged features, the paired
same-rank random null, and the variance-matched sensitivity in the same direction.

## Separate endpoints

### Fresh-probe utility

For each transformed representation, refit clipping, scaling, and a new task head on transformed source windows.
Report held-out recording balanced accuracy and NLL. This is the direct FMScope-style endpoint.

### Exact-head functional reliance

Fit clipping, scaling, and the task head on unchanged source features once. Freeze that complete predictor and
compare its held-out outputs on unchanged and transformed features. This tests whether the original predictor
used the removed directions. It is never merged with fresh-probe utility.

### Subject-axis transferability

For source-only LEACE report:

```text
source-vs-held-out subject-subspace projection overlap and principal angles
held-out between-subject mean-scatter removed by the source operator
held-out cross-recording subject decoding after source-only erasure
```

Held-out subject IDs may be used only to compute these final diagnostics. They never fit the source eraser or task
head. The cross-recording subject probe trains on one recording per held-out subject and tests on the other, then
reverses the direction; it does not randomly split adjacent windows. As in the public grouped-CV contract, task
labels are read once to construct the frozen stratified outer folds and later for final scoring. They never select a
method, rank, null, threshold, checkpoint, or hyperparameter.

## Effect modifiers

All primary modifiers are fit on source folds only:

```text
within-subject task-direction consistency
subject-task projection overlap
unchanged task-probe strength
post-erasure nonlinear subject decodability
```

For a binary task, the task direction is the mean of normalized within-subject label contrasts. Subject-task
overlap is the squared norm of that direction projected into the source LEACE subject basis. Held-out versions are
diagnostic-only and cannot select a protocol, threshold, or claim.

## Decision matrix

| Result | Licensed interpretation |
| --- | --- |
| A helps, B does not, C/D do not | Benefit depends on test-cohort subject geometry; transductive diagnostic upper bound |
| A and C help similarly | Generic dimensionality or conditioning benefit |
| A helps, C does not, and B helps beyond D | Source subject axis transfers; deployable erasure regime is supported |
| Fresh head helps, exact head does not | Representation is easier for a newly fit head; original predictor did not rely on that axis |
| Exact head is harmed | Removed subject geometry is task-entangled for the original predictor |
| Only high-consistency/low-overlap cells help | Task geometry is an effect modifier of erasure utility |
| B0 does not reproduce | Stop for implementation/cache mismatch; no bridge claim |

## Output package

```text
results/s2p_fmscope_fsr_bridge_panel1/
  bridge_external_asset_manifest.csv
  bridge_operator_equivalence.json
  bridge_b0_exact_replication.csv
  bridge_fold_assignments.csv
  bridge_fresh_probe_results.csv
  bridge_exact_head_results.csv
  bridge_random_null_results.csv
  bridge_transferability.csv
  bridge_effect_modifiers.csv
  bridge_panel1_inference.json
  bridge_target_information_firewall.json
  bridge_transform_canaries.json
  bridge_panel1_go_nogo.json
  bridge_independent_verification.json
```

## Scope boundary

Panel 2 on S2P immutable checkpoints is held until Panel 1 passes B0, completes all four protocols and controls,
and an independent verifier reproduces fold assignments, transformations, metrics, and claim status. Phase D1
remains frozen but its training stays held. No 1/f audit begins automatically.
