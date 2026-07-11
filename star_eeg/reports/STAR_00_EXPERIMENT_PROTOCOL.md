# STAR_00 Experiment Protocol

## Status and scope

This document preregisters future STAR_01. STAR_01 is not approved or executed by STAR_00A. No value below is a sweep candidate; any change requires a new PM gate.

## Starting points and references

The only starting points are H200_s0 and H200_s1 from frozen Route B. Each future model seed stays paired with its own H200 checkpoint and exact H200 Route B TUEG source pool/manifest.

H500/H1000/H2000/released/random are evaluation-time descriptive references. They do not enter continuation, variant selection, schedule selection, endpoint selection, or checkpoint selection.

## Frozen variant universe

| Variant | Continuation |
|---|---|
| `H200_BASE` (A) | Original H200 checkpoint, no additional update |
| `H200_SSL_CONT` (B) | Native CBraMod SSL in all continuation slots |
| `H200_STAR_TRUE` (C) | Four TUEG SSL updates followed by one FACED source_train true-label anchor update |
| `H200_STAR_SHUFFLED` (D) | Identical to C, using the frozen shuffled-label manifest |

No TASK_ONLY, additional anchor ratio, CE weight, freeze policy, head, budget, checkpoint, or label-budget variant exists.

## Fixed continuation compute

The H200 Route B run used 50 × 375 = 18,750 optimizer updates. STAR continuation is frozen at exactly **3,750 optimizer steps**, 20% of that update count. Epoch is not a compute-matching unit.

- Cycle: optimizer steps 1–4 native SSL; step 5 anchor slot; repeat 750 times.
- C/D: 3,000 common SSL batches and 750 anchor batches.
- B: the same 3,000 common SSL slots plus 750 replacement SSL batches at anchor slots.
- B/C/D total optimizer steps: 3,750 each.
- B/C/D total batches: 3,750 each.
- C/D semantic and update schedules are byte-content identical; only the frozen label mapping differs.
- Shared SSL slots use identical common-stream indices within seed/variant block. B's replacement slots use a separately keyed SSL stream from the same pinned H200 source pool.
- Batch size: 64 for both streams.
- Optimizer: fresh AdamW, base LR `5e-4`, weight decay `5e-2`.
- Scheduler: fresh cosine schedule defined on optimizer step 1–3750, eta-min `1e-5`.
- Checkpoint-relevant parameter scope: full CBraMod parameters in B/C/D. The same temporary-head parameter registry is instantiated in B/C/D; it has no forward path/gradient on B SSL-only updates.
- Save cadence: steps 750, 1500, 2250, 3000, and 3750.
- Primary checkpoint: fixed final optimizer step 3750 only.
- A diagnostic pretrain-val-best checkpoint may be retained but cannot select or replace the primary checkpoint.
- Wall clock is recorded but not matched.

The frozen temporary task head is one linear `6400 -> 9` layer on the exact FACED audit representation: CBraMod patch embedding, encoder, mean over the 10 patch positions, then flatten 32 channels × 200 features. Source cross-entropy weight is exactly 1.0. The head is discarded before frozen evaluation.

## RNG separation

Model seeds are 0 and 1. Within a model seed, B/C/D use the same start state, common SSL ordering, native SSL objective randomness, base optimizer schedule, save cadence, and CBraMod update scope. RNG streams are keyed separately:

- temporary head: `12000 + model_seed`
- common/replacement SSL streams: `13000 + model_seed` with distinct stream identifiers
- native SSL objective randomness: `14000 + model_seed`
- anchor batch stream: `15000 + model_seed`
- semantic permutation: fixed seed `20260711`, independent of model/training RNG

## FACED anchor and shuffled control

Training may read TUEG unlabeled data and FACED source_train X/y only. FACED source_train is subjects 1–80. Each full anchor manifest must be generated before launch from stable source_train sample identifiers, sorted and hash-frozen.

The shuffled mapping is generated once within each source_train subject. It preserves each subject's nine-class histogram. Source_val and target_test cannot participate. The same semantic permutation manifest is used for model seeds 0 and 1; training and permutation RNGs are independent. Labels are never reshuffled per epoch or cycle.

## FACED firewall

- source_train: subjects 1–80; X/y permitted for anchor gradients.
- source_val: subjects 81–100; labels permitted only for protocol validation, source-only task gate, and diagnostic reporting.
- target_test: subjects 101–123; neither labels, metrics, class distribution, nor endpoint information may be read before all training and final checkpoints are frozen.

Source_val cannot select the primary checkpoint. Target information cannot affect training, normalization selection, variant selection, head/PCA selection outside the frozen source-only pipeline, endpoint selection, stopping, or checkpoint selection.

## Future frozen evaluation

Evaluation reuses the S2P FACED frozen-probe path:

1. Freeze and discard the temporary anchor head.
2. Extract frozen native32 encoder representations.
3. Fit a fresh PCA on source_train only (`n_components` in the already frozen set 32/64/128).
4. Fit a fresh source_train logistic head (`C` in 0.01/0.1/1/10); choose PCA/C by source_val Kappa then balanced accuracy only.
5. Score target_test labels once after all primary checkpoints are fixed.
6. Report primary Cohen's Kappa; secondary balanced accuracy and weighted F1.
7. Use 5,000 paired target-subject cluster bootstrap replicates over subjects 101–123 with seed `20260710`; average the two training seeds inside each replicate.
8. Include random, released, and H500/H1000/H2000 as frozen descriptive references.

Task gate precedes any L4/L5/L6 interpretation. The frozen gate is source-val Kappa at least `0.056362325458434434`. The primary positive classification requires both C cells to pass; every other cell's mechanism readout is suppressed if its task gate fails.

Recompute for every STAR checkpoint cell:

- L1 pairwise subject separability.
- L4 task-head energy in the measured rank-5 source-train subject subspace.
- L5 subject-subspace intervention versus a source-val-energy-matched random orthobasis null.
- L6 target consequence of measured subject-subspace erasure.

L5 uses exact source-val removed-energy matching with a partially erased final random direction, never target-label fitting. Use 200 null bases with the frozen S2P seed 92015. Apply preregistered Holm correction across the eight A/B/C/D × seed checkpoint-level Kappa intervention comparisons. Budget/reference summaries remain descriptive.

## Scientific gates

The primary positive gate requires all of:

1. C minus B mean target Kappa at least +0.020.
2. Paired target-subject CI for C minus B excludes zero.
3. C minus D mean target Kappa at least +0.015.
4. C minus B is positive for both seeds.
5. C point estimate clears the existing random +0.02 floor rule.
6. No target-label firewall violation.
7. Compute-match PASS.
8. Both C cells pass the source-val task gate.

Strong mechanism classification additionally requires increased target Kappa, L1 remaining high without a required decrease, improved L4/class accessibility, and L5 not exceeding the matched null. Allowed wording is: “Source-task anchoring advances task accessibility without requiring measured subject-information erasure.”

If C improves target performance but L5 exceeds the matched null, role separation is forbidden. Record only that the task anchor recruits the measured subject subspace and return to PM.

Scientific FAIL includes C ≤ B, C approximately equal to D, a one-seed-only gain, compute mismatch, any target-label influence, a gain only at a diagnostic checkpoint, source-val task-gate failure, or equal improvement from shuffled labels. FAIL does not authorize a rescue experiment.
