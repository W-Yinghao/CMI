# S2P_36 - FMScope-FSR Bridge Design Red-Team

**Disposition:** PANEL-1 MAY RUN ONLY UNDER THE FROZEN S2P_35 CONTRACT.

This document records failure modes and claim boundaries. It is not a manuscript and does not authorize Panel 2,
Phase D1 training, fine-tuning, layerwise analysis, or aperiodic ablation.

## R1. Global LEACE is transductive

The public FMScope implementation fits whitening and subject cross-covariance with every cohort window and subject
ID before grouped task CV. Subject-disjoint task folds do not undo that information use. Protocol A is therefore a
cohort-conditioned diagnostic, not a source-to-unseen-subject deployment estimate.

**Guard:** A and B are named and reported separately. No deployment wording is licensed by A.

## R2. Fresh-probe gain is not functional reliance

Refitting a head after erasure allows the classifier, scaler, and clipping bounds to adapt to a new feature
geometry. A gain can arise even when the original head did not use the removed axis.

**Guard:** fresh-probe utility and exact-head reliance are separate endpoint families. Neither substitutes for the
other.

## R3. Generic dimensionality and conditioning can mimic benefit

LEACE removes rank `n_subjects-1` in Panel 1. A fresh linear head may benefit from lower dimension, lower variance,
or changed conditioning without any identity-specific mechanism.

**Guard:** same-rank random removal is primary; removed-variance-matched random removal is a mandatory sensitivity.
An identity-specific claim must exceed both.

## R4. Same rank is not same removed variance

LEACE is oblique in the original feature coordinates. Equal-rank random bases can remove different feature energy.

**Guard:** persist source-fit and held-out removed-energy fractions for every operator and execute the partial-last-
direction variance-matched sensitivity. Never call the latter a same-rank control.

## R5. Public live builder and bundled aggregate can drift

At public commit `09885016...`, the live erasure builder and a historical aggregate are separate provenance
objects. Solver tolerance, cache refreshes, or implementation updates can move a few recording decisions.

**Guard:** pin repository commit and cache SHA; store live output and historical value separately. B0 uses a
prospective tolerance and cannot select whichever result is more favorable.

## R6. Liblinear randomness is incompletely pinned upstream

The public task head fixes fold seeds but does not pass `random_state` to liblinear. This can make threshold-adjacent
predictions platform-sensitive.

**Guard:** B0 calls public code unchanged. The unified bridge prospectively pins liblinear `random_state` while
retaining every other head and preprocessing choice. Exact-replication and bridge estimates are not conflated.

## R7. Window-random subject probes leak recording adjacency

The public subject probe uses stratified window folds. It is suitable for reproducing the public diagnostic but can
overstate identity transfer across recordings.

**Guard:** retain it only for B0. The deployment transferability endpoint uses cross-recording held-out-subject
decoding in both directions.

## R8. Target subject IDs are still target information

Protocol B cannot use held-out IDs while fitting whitening, subject axes, random controls, task heads, or
hyperparameters.

**Guard:** target IDs are accessed only after task predictions are frozen, for final transferability diagnostics.
The run manifest records every access class. Any earlier access is a stop rule.

## R9. Target task labels can leak through geometry analysis

Direction consistency and task directions can accidentally use held-out labels to select cells or removal ranks.

**Guard:** primary modifiers use source labels only. Held-out labels are used only by the preregistered stratified
outer-fold constructor and final scoring; they cannot change rank, null family, threshold, method, or inclusion.

## R10. Source and target subject spaces have different identities

There is no shared subject-class label across source and held-out cohorts. A subject classifier trained on source
IDs cannot directly score unseen IDs.

**Guard:** transferability is evaluated through subspace geometry, target mean-scatter removal, and a fresh
cross-recording decoder among held-out identities. It is not called source-ID classification accuracy.

## R11. High global/source subspace overlap can be rank-driven

When a source subject basis has high rank, random overlap with a smaller held-out basis is nonzero.

**Guard:** report observed normalized projection overlap, principal angles, and a rank-matched random-overlap
reference. Absolute overlap alone is not evidence of transfer.

## R12. EEGMAT and SleepDep are paired-state cells, not trait cells

Erasing subject geometry in a trait-label dataset can erase the label by construction. Such a result is not a clean
test of deployment benefit.

**Guard:** Panel 1 is restricted to the two within-subject paired cells. No trait cell is silently added.

## R13. SleepDep is a negative-control observation, not a required null

Sampling noise can produce a positive or negative SleepDep delta. Requiring exactly zero would be outcome-driven.

**Guard:** only successful execution, provenance, and interpretability are gated. Its effect sign is reported, not
used to tune the method.

## R14. Random draws cannot become a model-selection resource

Reporting the best random draw or changing the number of draws after seeing p-values invalidates the null.

**Guard:** exactly 100 domain-separated draws per cell/fold/regime; empirical tail probabilities use all draws.
Holm correction covers all eight dataset-by-regime-by-endpoint comparisons within each null family.

## R15. Dataset-level pseudoreplication

Windows and recordings are not independent biological replicates. Three split seeds are also not three datasets.

**Guard:** retain fold/seed rows, cluster task summaries by subject/recording as frozen, and do not convert window
counts into inferential sample size. Panel 1 is a two-cell mechanistic bridge, not a population-wide meta-analysis.

## R16. External released checkpoint provenance is limited

The bundled representation is a public released-CBraMod cache, not a locally regenerated immutable feature dump.

**Guard:** claims are conditional on the public cache commit and hash. They do not assert training-data provenance
or equivalence to Route-B checkpoints.

## R17. Panel-1 success does not authorize Panel 2

The S2P budget panel changes task, class count, rank, feature extraction, and biological split.

**Guard:** Panel 2 requires a separate PM review after independent Panel-1 verification. No automatic fleet.

## R18. This bridge does not resolve unique data versus exposure

The bridge uses frozen released features and cannot answer Phase D1's factorial question.

**Guard:** D1 remains scientifically valid and frozen, but compute stays held until the bridge identifies which
deployment endpoint is informative.

## Fail-closed stop rules

Stop Panel 1 and return without mechanism claims if any of the following occurs:

```text
1. Official repository commit differs from the pinned commit.
2. Either public cache SHA differs from the pinned SHA.
3. Cache recording/subject/label arrays are missing or inconsistent.
4. B0 EEGMAT rank, linear-erasure, interpretability, sign, or tolerance gate fails.
5. Any outer fold has subject or recording overlap.
6. Protocol B/D fits any operator with held-out features or IDs.
7. A removal rank differs from its paired subject-LEACE rank.
8. A variance-matched draw exceeds absolute source-fit error 1e-10.
9. Fresh and exact-head endpoints reuse the wrong fitted head.
10. A target label or ID changes rank, nulls, thresholds, folds, or inclusion.
11. A result is selected by best random draw, seed, or fold.
12. The verifier cannot reproduce metrics and decision status from low-level artifacts.
```

## Red-team verdict

The design is scientifically identifiable for the narrow question:

> Does a cohort-global subject eraser provide a benefit that survives source-only deployment and identity-specific
> random-removal controls, and is that benefit fresh-head utility or exact-head reliance?

It does not establish that subject removal is universally helpful, that all subject information is harmful, or
that a source-fitted eraser transfers beyond the two frozen public cells.
