# S2P_33 - Phase D1 Unique-Data Volume x Cumulative Exposure Protocol

**Status:** PROTOCOL PREFLIGHT ONLY / TRAINING HELD FOR PM AUTHORIZATION.

This document is a prospective scientific protocol, not a manuscript. It does not authorize pretraining,
downstream evaluation, fine-tuning, H4000, layerwise analysis, CodeBrain work, or a new dataset.

## Scientific object

D1 separates two factors that were confounded in the original Route-B budget curve:

```text
U: unique-data volume and breadth = {200h, 1000h}
P: cumulative training exposure = {18,750, 93,750 optimizer updates}
```

With fixed batch size 64, P jointly changes optimizer updates, sample presentations, repeated window exposure,
and mask draws. It is not a pure optimizer effect. U jointly changes unique windows, subjects, recordings, and
population breadth. It is not a pure subject-count effect.

## Nested-compute trajectories

The design uses eight high-horizon trajectories:

```text
2 U arms x 2 subset seeds x 2 initialization seeds = 8 trajectories
```

Each trajectory runs to 93,750 updates and emits two primary snapshots:

| U arm | P_low snapshot | Epoch | P_high snapshot | Epoch |
| --- | ---: | ---: | ---: | ---: |
| 200h | 18,750 | 50 | 93,750 | 250 |
| 1000h | 18,750 | 10 | 93,750 | 50 |

P_low and P_high are points on the same optimization trajectory. Training continues from the in-memory P_low
state; it does not restart from a snapshot or a new initialization. The design therefore yields 16 immutable
representation checkpoints from eight jobs.

## Corpus contract

For each `subset_seed` in `{0,1}`:

1. Records are deterministically ordered within each pinned channel-order/reference group.
2. The exact 1000h allocation is taken from that order using the frozen group quotas.
3. The exact 200h core is taken from the same order using the frozen 200h group quotas.
4. Every 200h window identity must occur in the corresponding 1000h manifest.
5. The 1000h arm is exactly the 200h core plus 96,000 additional windows.
6. Both initialization seeds use the same manifest for a given subset seed and U arm.
7. The fixed 128-subject, 3,072-window pretrain-validation pool remains disjoint and unchanged.

The preflight persists row-level allocations, canonical window hashes, subject/recording overlap, per-subject
exposure, and exact group proportions. A subject-only hash is insufficient for launch.

## Randomness contract

Three roles are separate:

```text
subset_seed: selects the corpus allocation only
init_seed: selects initial model parameters only
stream_seed: identifies the trajectory's stochastic stream contract
loader_seed_root: controls epoch shuffle seeds
mask_seed_root: controls update-indexed mask seeds
```

`stream_seed` is a preregistered SHA256-derived function of `(subset_seed, init_seed)`. Loader and mask roots use
separate SHA256 namespaces and cannot be the same integer. Both U arms in a block share these roots. Epoch
shuffle seeds additionally include U and epoch; mask seeds include the global optimizer update. The model
initialization hash must be exactly equal between U200 and U1000 in every block.

## Optimizer and LR contract

```text
optimizer: AdamW
base LR: 5e-4
weight decay: 5e-2
batch size: 64
gradient accumulation: 1
mask ratio: 0.5
high horizon: 93,750 updates
warmup: 0 updates
eta_min: 1e-5
```

All arms use one common step-indexed cosine schedule with `T_max=93,750`. P_low is only a snapshot at update
18,750; it does not receive a separately compressed cosine schedule. Thus U200 and U1000 have identical LR at
every matched update.

## Validation and snapshots

Validation occurs every 1,875 optimizer updates. It is diagnostic and does not select either primary snapshot.
The primary checkpoints are exactly updates 18,750 and 93,750. Best-pretrain-val may be recorded as a secondary
sensitivity only.

At each primary update, the job must immediately create a content-addressed snapshot with no-overwrite semantics,
SHA256 pinning, mode `0444`, strict model/optimizer/scheduler reload, global-update verification, and an unlabeled
fixed pretrain-validation feature canary with maximum absolute difference zero. The P_low snapshot cannot remain
only as a mutable `latest` or `last` path while training continues.

## Estimands

For endpoint Y:

```text
Delta_U_low  = Y(U1000,P_low)  - Y(U200,P_low)
Delta_U_high = Y(U1000,P_high) - Y(U200,P_high)
Delta_P_200  = Y(U200,P_high)  - Y(U200,P_low)
Delta_P_1000 = Y(U1000,P_high) - Y(U1000,P_low)
Interaction  = Delta_U_high - Delta_U_low
```

All contrasts are paired within `(subset_seed, init_seed)` blocks. The design has only two corpus-composition and
two initialization replicates, so uncertainty that remains wide at the training level must be labelled a
`PILOT_FACTORIAL_ESTIMATE`; downstream biological samples do not replace pretraining replication.

## Downstream hierarchy after a separate post-training gate

No downstream job is authorized by this protocol package. If training is later approved and all 16 snapshots
pass provenance closure:

```text
Primary dataset: FACED
Primary utility endpoint: Cohen's Kappa
Key secondary: target NLL
External validation: SEED-V and ISRUC_S3, reported separately
Mechanism: subject-probe NLL, K-1 subject-task overlap, task-gated L5
```

Target test labels are final-score-only. Variance partition remains permanently excluded after its Phase-B
stability failure. No cross-dataset pooled p-value or best-seed result is allowed.

## Launch boundary

The protocol preflight can recommend scientific readiness, but `launch_phase_d1` remains false until a separate PM
training authorization. Historical H200/H1000 checkpoints remain historical anchors and are not factorial cells.
