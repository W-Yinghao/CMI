# S2P_39 - FMScope-FSR Bridge Panel 2 Protocol

**Status:** PANEL-2 FROZEN-FEATURE COMPUTE AUTHORIZED / NO NEW PRETRAINING.

This is a prospective experiment contract, not manuscript text. It does not authorize fine-tuning, H4000,
CodeBrain, a new dataset, a layer search, a 1/f audit, or submission writing.

## Scientific question

Panel 2 asks whether increasing Route-B pretraining budget changes the gap between cohort-global diagnostic
erasure and source-only deployment erasure:

```text
Can a subject axis that is transferable and removable become task-beneficial to remove
as subject-task geometry changes with pretraining budget?
```

The three endpoints remain separate: fresh-head utility, exact-head reliance, and subject-axis transferability.

## Immutable representation objects

Only the Phase-B closure manifest's ordered ten objects are eligible:

```text
random, released,
H200_s0, H200_s1, H500_s0, H500_s1,
H1000_s0, H1000_s1, H2000_s0, H2000_s1
```

Every physical checkpoint is read directly from its content-addressed path and SHA256-checked before and after
feature extraction. Random initialization is reconstructed from its deterministic code/seed contract.

## FMScope-aligned representation

The Panel-2 representation is the final CBraMod block pooled over both patch and channel axes, yielding exactly
200 dimensions for every native montage:

```text
encoded: [B,C,P,200]
panel2_feature: mean(encoded, axes=(C,P)) -> [B,200]
```

This fixed operator uses no labels or target distribution and avoids a target-fitted PCA before global/source
comparison. It is aligned with the 200-dimensional pooled frozen-CBraMod representation used in Panel 1. It is a
new Panel-2 representation contract and does not replace the flattened PCA128 authority used by Phase B/C.

## Dataset roles

### FACED primary

```text
source train: subjects 1-80
source validation: subjects 81-100
target test: subjects 101-123
task unit: original clip, pooling its three 10-second segment probabilities
classes: 9
```

Source-only erasers and heads use subjects 1-80 only. Validation is task-gate-only. Test labels are final-score
only. Target subject IDs may enter final transferability diagnostics but never fit the source operator.

### SEED-V secondary

The frozen 5/5/5 trial split is retained. Each trial is one feature unit and subject is the biological cluster.
Because all splits contain the same 16 subjects, source-only means train-trial-only, not unseen-subject deployment.

### ISRUC-S3 directional external panel

The ten rotating 8:1:1 subject splits and 20-epoch sequence task contract remain mandatory. ISRUC is not pooled
with FACED or SEED-V inference. Its low subject count and sequence-head cost make it directional rather than a
familywise confirmation panel; any identity-specific positive requires paired training-seed agreement.

## Four arms and controls

For every eligible representation object:

```text
A global cohort subject LEACE
B fold-wise/fixed-split source-only subject LEACE
C global same-rank random removal
D source-only same-rank random removal
```

One hundred domain-separated random draws form the FACED and SEED-V same-rank nulls. Removed-variance-matched
random removal is a preregistered sensitivity. No best draw is selected. ISRUC may use a separately recorded
directional random-draw count because each draw requires sequence-head retraining; it cannot support a
familywise identity-specific claim with fewer than 100 draws.

## Endpoints

Fresh-head utility refits the fixed clipping, scaling, and `C=1` linear head on transformed source features.
Exact-head reliance freezes the complete head fit on unchanged source features and only transforms its input.
Both report Kappa, balanced accuracy, and NLL. Axis transferability reports target subject scatter removed,
pre/post subject decoding, source-target subject-subspace overlap, and removed feature energy.

The FACED task gate remains:

```text
source-val Kappa >= 0.05
and source-val Kappa >= random source-val Kappa + 0.02
```

Exact-head effects from cells below this gate are diagnostic only.

## Primary hypotheses

```text
P2-H1: pooled H500/H1000/H2000 source-only fresh-head subject removal exceeds its random null.
P2-H2: random-corrected global fresh-head benefit exceeds random-corrected source-only benefit.
P2-H3: exact-head removal cost is smaller at pooled higher budgets than at H200.
P2-H4: source-only target-subject scatter removal remains high across H200-H2000.
```

P2-H1 through P2-H3 form the Holm-corrected FACED primary family. Training seeds are averaged within budget;
target-subject cluster bootstrap is reported separately from the empirical random-direction null. H500, H1000,
and H2000 are all retained; there is no monotonic scaling claim or best-budget selection.

## Scope boundary

Panel 2 cannot establish that the complete high-rank LEACE axis is identical to Phase B's rank-8 subject
subspace. A null source-only utility result licenses only the pooled final-representation contract. SEED-V cannot
license unseen-subject claims, and ISRUC cannot be pooled into an overall EEG p-value.
