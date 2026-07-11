# STAR_00B Real-Path Launch Preflight Protocol

## Authority and gate

STAR_00A commit `26c3fca009d0ecbde7e92e7c759c0256caf1361d` is the immutable design baseline. STAR_00B may add artifact closure, real source-only manifests/loaders, bounded CUDA smoke, telemetry, and held launch templates. It cannot run a 3,750-step cell, inspect FACED source_val/test, score a target metric, select a checkpoint, add a variant, or change any frozen scientific hyperparameter.

The required state throughout STAR_00B is:

```text
STAR_00_PROJECT_CHARTER: PASS
STAR_00A_DESIGN_AND_RED_TEAM_PREFLIGHT: PASS
STAR_H200_ARTIFACT_SUPPLY: AVAILABLE_WITH_IMMUTABILITY_ACTION / AVAILABLE_IMMUTABLE
STAR_00B_REAL_PATH_PREFLIGHT: IN_PROGRESS / PASS
STAR_01_SCIENTIFIC_TRAINING: BLOCKED
STAR_TARGET_SCORING: BLOCKED
STAR_MANUSCRIPT_CLAIM: FORBIDDEN
S2P_PHASE_B: INDEPENDENT / UNCHANGED
H2CMI_AND_OACI: PROTECTED / UNCHANGED
```

## Immutable H200 starts

The two STAR_00A H200 source SHAs are copied once to external, SHA-named payloads. Closure requires source SHA stability across copy/reload, destination SHA equality, strict reload, mode `0444`, and a stable relative `best.pth` symlink. The training launcher reads only `launcher_accepted_path` from the committed closure manifest and rejects a symlink, writable path, filename/SHA mismatch, or changed payload.

No H200 retraining is permitted. The source Route-B files remain untouched.

## Real source-only FACED path

`FACEDSourceTrainAnchorLoader` is hard-coded to the LMDB `train` index member. Inventory generation and runtime loading admit exactly 6,720 sample IDs from subjects 1–80, 84 per subject, with the frozen per-subject nine-class histogram. They do not deserialize, count, tensorize, or compute class statistics for any other split.

The full anchor stream contains 750 batches × 64 = 48,000 exposures. Every subject appears exactly 600 times. Each of 84 source samples receives seven exposures and a deterministic set of 12 samples per subject receives an eighth. The fixed shuffled mapping permutes labels separately within each subject's bonus/non-bonus strata, preserving both the full-corpus and exposure-level label marginal. C and D therefore have identical sample IDs, batch boundaries/order, and X tensors within seed; only the frozen label mapping differs.

## Real H200 Route-B SSL path

For each model seed, the exact H200 Route-B 24,000-window source pool is rebuilt from its frozen manifest. The common stream is eight deterministic full passes (192,000 exposures, 3,000 batches); B's replacement stream is two separately keyed full passes (48,000 exposures, 750 batches). Common batch IDs are identical across B/C/D within seed.

## Frozen update semantics

- CBraMod full parameter scope; temporary linear `6400 -> 9` head in the same optimizer registry for B/C/D.
- AdamW, LR `5e-4`, weight decay `5e-2`.
- CosineAnnealingLR defined on 3,750 optimizer steps, eta-min `1e-5`.
- Batch size 64.
- Native CBraMod `generate_mask`, mask ratio 0.5.
- Masked reconstruction MSE with mean reduction.
- Per-channel, per-one-second-patch z-score with epsilon `1e-6` for both source streams.
- Explicit `optimizer.zero_grad(set_to_none=True)`.
- Model and temporary head in train mode.
- Full FP32; mixed precision disabled.
- Model gradient clip norm 1.0; temporary-head gradient clipped separately to 1.0 so it cannot alter native model clipping.
- Checkpoints include model/head/optimizer/scheduler states, optimizer step, immutable source SHA, frozen config, telemetry hash, and firewall flags. Strict reload is mandatory.
- The temporary head is never an evaluation head and must be discarded before frozen probing.

## Approved CUDA smoke

Use H200_s0 only and run B/C/D for exactly ten optimizer steps each. This is a bounded real-path integrity smoke, not a scientific training cell. It must show identical starts/update scope, identical common SSL batches, identical C/D anchor X, differing true/shuffled labels, two B replacement slots, finite loss/gradients/clipping/deltas, strict checkpoint reload, unchanged immutable start SHA, and zero non-source FACED access.

Any failure keeps STAR_01 blocked. Learning rate, anchor ratio, batch size, layer scope, or head cannot be changed as an automatic remedy.

## Telemetry and selection firewall

Every smoke/training step records step, semantic slot, data stream, ID/content batch hashes, loss, encoder/model/head gradient norms before and after clipping, parameter delta norm, LR, mask policy, finite status, immutable source SHA, and current model-state hash. Telemetry is integrity evidence only and cannot select a checkpoint or protocol.

## Blind chain (prepared, not submitted)

All six B/C/D × seed training jobs must be launched together after a new PM approval. Only after all six complete may an afterok chain close immutable final-step payloads, run a source-only integrity/task-gate audit, execute one all-cells scoring job covering A/B/C/D × two seeds plus all frozen references, and run an independent verifier. Partial, sequential, seed-screened, or adaptive scoring is forbidden.

Even after STAR_00B PASS, STAR_01 remains blocked pending a new PM instruction.

The single frozen semantic permutation and two model seeds make any future STAR_01 success a **two-seed positive screen**, not final confirmation. Seed 2 or a second frozen permutation is outside the current universe and may be requested only after a positive STAR_01 PM review.
