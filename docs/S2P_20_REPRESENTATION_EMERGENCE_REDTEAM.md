# S2P_20 - Representation Emergence Red-Team

**Phase:** B0 adversarial review. **Status:** initial blocker resolved; B1 remains held for PM approval.

> Post-closure update: provenance job 892980 and the independent verifier closed the seven-object immutable
> blocker with no failures. This document retains the original pre-closure attack analysis. The authoritative
> closure result is S2P_22 and B1 remains unauthorized.

## Executive decision before closure

The FACED clip contract, final feature path, source/target firewall, rank-8 geometry definition, and balanced
subject-class design are technically viable. B1 is nevertheless NO-GO at this checkpoint because H200/H500/H1000
and the released reference still resolve to writable, non-SHA-named payloads. Only H2000 satisfies the immutable
contract. B0 does not repair this silently and does not launch scientific compute.

## Evidence audit

| Item | Evidence | Decision |
|---|---|---|
| Original clip ID | Raw file shape `(28,32,7500)`; all 123 LMDB subjects contain keys `28 clips x 3 segments` | PASS |
| Raw-to-LMDB mapping | Five clip/segment checks reproduce resample-and-slice output with max error `<6e-14` | PASS |
| Clip/class map | Clip labels are identical for checked subjects 0, 40, and 122; all nine classes represented per fold | PASS |
| Clip leakage prevention | Three folds assign all segments from each clip together; every clip held out once | PASS |
| Feature path | Phase-A final verifier reproduced 10/10 objects exactly; repeat-feature max difference is 0 | PASS |
| Target firewall | Source-only fit and target-final-score split already exists and remains enforceable | PASS |
| Geometry rank | Equal rank fixed at 8 before B1; no target-dependent rank choice | PASS |
| Variance cells | Source train has 80 x 9 complete cells, with three clips per class except four for class 4 | PASS |
| Checkpoint immutability | H2000 passes; six lower-budget checkpoints and released reference are writable | **FAIL** |
| Empirical non-saturation | Cannot be measured in B0 without running the prohibited scientific endpoint | DEFERRED with fail-closed B1 rule |

## Adversarial findings

### 1. The prior segment split was invalid for Phase-B subject probes

The existing FACED audit named the second LMDB key token `condition_id` and used an `item_index % 5` holdout for
L1. For Phase B, the second token is proven to be the original clip ID, while the third token is the segment. A
segment-index holdout can place neighboring 10 s windows from the same 30 s clip on both sides and is forbidden.

Mitigation: B1 must parse `clip_id` explicitly and use only the frozen three-fold clip grouping. Phase-A L1 remains
valid for its original descriptive purpose but cannot be reused as a Phase-B continuous subject endpoint.

### 2. Clip grouping does not remove every nuisance

All subjects viewed the same clips. A subject probe evaluated on unseen clips cannot memorize a training segment,
but it can still exploit subject-specific responses that interact with video content. This is part of the scientific
object, not pure biometric identity.

Mitigation: every fold contains every emotion class; class-conditional subject NLL is reported; and the variance
analysis separately estimates subject, class, and interaction components. Wording must remain
`subject-identifiable structure`, not immutable biometric identity.

### 3. Class 4 has one extra clip

Class 4 contains four clips while every other class contains three. Naive segment weighting would give it excess
influence and fold 0 holds two class-4 clips.

Mitigation: class-conditioned endpoints macro-average the nine classes. Geometry and variance use equal
subject-class cell weights. Overall task metrics retain the actual dataset distribution and report macro metrics as
a sensitivity.

### 4. Subject log loss can fail despite being continuous

Regularized multinomial probabilities may still approach one under near-perfect separation. Lower NLL would then
reflect numerical clipping or regularization more than representational change.

Mitigation: fixed whitening and `C=1`; report the true-subject probability distribution, pairwise AUC, standardized
margin, and retrieval mAP; apply the preregistered saturation rule. If it fires, label the endpoint
`UNINFORMATIVE_UNDER_THIS_METRIC`. Do not tune C or search another probe after seeing the result.

### 5. Probe whitening and raw variance answer different questions

Whitening is useful for comparing fixed-regularization probes but destroys the natural embedding variance spectrum.
Using one space for both analyses would make at least one interpretation wrong.

Mitigation: probe endpoints use fold-fit PCA128 whitening; variance and effect-subspace geometry use raw final
features. Claims must identify the space. PCA128 is fixed for all objects and cannot be selected per checkpoint.

### 6. Canonical-correlation and projection-overlap aggregates are not independent

For two equal-rank orthonormal bases, normalized Frobenius projection overlap equals mean squared canonical
correlation. Reporting both as confirming tests would duplicate one statistic.

Mitigation: normalized projection overlap is primary. The sensitivity is the full principal-angle spectrum,
including maximum canonical correlation and median angle.

### 7. Low observed overlap does not prove independence

A non-significant high-minus-H200 overlap contrast cannot establish equivalence or no nonlinear interaction. Linear
effect subspaces may miss conditional or curved subject-task organization.

Mitigation: report held-out captured energy, confidence intervals, and interaction trace. Allowed wording is
`largely distinct under the measured rank-8 linear effect subspaces` only when the subspaces pass their cross-fit
stability gate. Never write `independent representations`.

### 8. Cell-mean interaction is upward biased without cross-fitting

Squaring noisy subject-class cell means assigns within-cell sampling noise to interaction variance. The problem is
especially serious with only three segments per clip.

Mitigation: use fit/held-out effect cross-products, retain negative unbiased estimates, report all fold values, and
apply the frozen instability rule. A non-cross-fitted ANOVA table is forbidden.

### 9. Bootstrap pairs are not biological replicates

There are 3160 source-subject pairs, but only 80 biological source subjects. Treating pairs as independent would
produce severely narrow intervals.

Mitigation: sample source subjects first. Recompute pair aggregates with induced `w_i*w_j` weights. Target task
endpoints continue to cluster over the 23 FACED test subjects.

### 10. Primary contrast direction must not become a monotonic claim

Pooling H500/H1000/H2000 tests a late-versus-H200 contrast. It does not imply ordering among the high budgets and
cannot rescue a monotonic scaling claim.

Mitigation: weight the three budgets equally, report each separately, and perform the frozen leave-one-high-budget
sensitivity. `Scaling law`, `optimal budget`, and `monotonic improvement` remain forbidden.

### 11. Target NLL can be used for scoring, not selection

Target NLL is a primary endpoint, but it cannot choose PCA dimension, C, calibration, pooling, layer, rank, or a
dataset. Source-val NLL is reported independently.

Mitigation: the entire metric and model contract is hashed before B1. The target firewall must explicitly record
false for every selection path and true only for final scoring/bootstrap scoring.

### 12. The released reference is not a training-scale anchor

Released CBraMod has unmatched data, epochs, sampling, and provenance. Its position cannot define a threshold or
select an analysis that favors Route B.

Mitigation: released is descriptive and path-validating only. It receives the same immutable/SHA/strict-reload
contract but no role in the three primary contrasts.

### 13. Current checkpoint paths violate the new authority rule

The six H200-H1000 `best.pth` files and released checkpoint are mode 0644, not SHA-named immutable payloads. Their
current SHA256 values can be recorded, and Phase A reproduced their metrics exactly, but either file can still be
overwritten. This fails the explicit Phase-B contract.

Mitigation required before B1:

1. validate each lower-budget run summary and training-time strict reload;
2. copy each selected checkpoint to a SHA-named read-only payload;
3. copy released CBraMod to a SHA-named read-only reference payload;
4. strict-reload every immutable payload;
5. record source/destination SHA, bytes, mode, and provenance;
6. repeat the B0 provenance check and obtain PM approval.

No scientific feature extraction may be combined with this repair.

## B1 fail-closed conditions

B1 must stop without partial interpretation if any of the following occurs:

- a checkpoint SHA differs before and after extraction;
- strict reload fails or a state key is missing/unexpected;
- a clip appears in both fit and held-out sets within a fold;
- feature repeat max difference exceeds `1e-6`;
- fixed PCA128/C=1/rank8 settings differ across objects;
- random or released no longer reproduces the Phase-A path-validity band;
- target labels enter fitting or selection;
- a primary subject metric triggers its saturation rule;
- subspace held-out self-capture is below 0.05;
- variance instability crosses its preregistered threshold.

Failures are reported as `UNINFORMATIVE` or `NO-GO`, not repaired by an unregistered metric, rank, layer, split, or
dataset.

## Initial B0 verdict

```text
clip identity and grouping:       PASS
feature path and determinism:     PASS from Phase-A 10/10 reproduction
metric/rank definitions:          FROZEN
variance cell structure:          PASS
checkpoint immutable contract:    FAIL for released + H200/H500/H1000
Phase B1 compute:                  NOT AUTHORIZED
next allowed action:               provenance-only immutable closure, then repeat B0 review
```

This verdict launches no training, fine-tuning, downstream feature extraction, H4000, CodeBrain run, layerwise
analysis, or manuscript work.
