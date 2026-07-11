# S2P_19 - Representation Emergence Protocol

**Phase:** B0. **Status:** FROZEN. **B1 scientific compute:** HELD pending the B0 go/no-go and PM review.

## Scientific question

Phase B asks what representational structure changes as Route-B pretraining moves from 200 h to the sampled
500-2000 h range. It distinguishes three mechanisms:

1. class structure is added alongside an already mature subject structure;
2. subject and class structure strengthen in overlapping directions;
3. subject-by-class interaction, rather than an independent class component, accounts for the later transfer.

It does not retest whether L1 is high or whether H500-H2000 exceed the random FACED floor. Those are Phase-A
facts locked at commit `a9134eb5eb7f8486a5e1ee41831823dab39381ed`.

## Scope and exclusions

Included objects are random initialization, released CBraMod, and H={200,500,1000,2000} x seeds={0,1}. The model
path is CBraMod Route B, the downstream dataset is FACED native32, and the primary representation is the final
encoder output used in Phase A.

Phase B contains no pretraining, fine-tuning, H4000, CodeBrain science grid, new downstream dataset, layer search,
or manuscript work. Released CBraMod is a path-validity and scale reference only. It cannot select a metric,
pooling operation, rank, regularizer, layer, or conclusion.

## FACED clip identity contract

The local raw subject files have shape `(28,32,7500)`: 28 original clips, 32 channels, and 30 s at 250 Hz. Each
LMDB key is `subNNN.pkl-<clip_id>-<segment_id>`. Every one of the 123 subjects has exactly the Cartesian product
`clip_id=0..27` x `segment_id=0..2`, yielding three 10 s segments per original clip.

This interpretation is not inferred from naming alone. For checked clips, resampling raw clip `c` from 7500 to
6000 points and slicing points `[2000*g:2000*(g+1)]` reproduces LMDB key `subject-c-g` with maximum absolute error
below `6e-14`. The clip-to-class map is fixed across subjects:

```text
class 0: clips 0,1,2        class 5: clips 16,17,18
class 1: clips 3,4,5        class 6: clips 19,20,21
class 2: clips 6,7,8        class 7: clips 22,23,24
class 3: clips 9,10,11      class 8: clips 25,26,27
class 4: clips 12,13,14,15
```

The existing `item_index` or segment index is forbidden as a subject-probe split variable.

### Three-fold clip cross-fit

Within each class, clips are assigned to held-out folds by their zero-based position modulo three. Consequently,
class 4 contributes clips 12 and 15 to fold 0, clip 13 to fold 1, and clip 14 to fold 2. Every original clip is
held out exactly once, all three segments of a clip remain together, and every fold contains every class.

For fold `f`, fit-only operations use clips assigned to the other two folds. Held-out predictions use only clips
assigned to `f`. No PCA, probe, centroid, subspace, temperature, or normalization statistic may cross this
boundary. Metrics aggregate out-of-fold predictions so each original clip contributes once.

## Checkpoint contract

Every non-random object used by B1 must resolve to a SHA-named, read-only immutable payload. The audit manifest
must contain the full SHA256, bytes, file mode, strict-reload result, checkpoint epoch/config where available, and
the Phase-A deterministic-reproduction result. SHA is checked before and after feature extraction.

Random initialization is defined by the pinned model code, architecture arguments, and torch seed 0. Released
CBraMod must be copied into the same immutable contract; the mutable source path is not sufficient.

Any mutable checkpoint makes B1 fail closed. A hash recorded for a writable file is provenance evidence, not
immutability.

## Final representation contract

Input preprocessing is unchanged from Phase A: FACED native32, 200 Hz, 10 s, reshaped to `(32,10,200)`, followed
by per-channel/per-1 s patch z-scoring. The encoder remains frozen.

The authoritative final representation is:

```python
patch = model.patch_embedding(x, None)
encoded = model.encoder(patch)
feature = encoded.mean(dim=2).reshape(batch, -1)  # 32 x 200 = 6400
```

Two analysis spaces are fixed before B1:

- **Probe space:** fold-fit or source-train-fit PCA with 128 components and whitening. PCA dimension is fixed for
  all objects. Logistic probes use `C=1`, `lbfgs`, `max_iter=2000`, and no per-object tuning.
- **Geometry/variance space:** the unwhitened 6400-dimensional final feature. This preserves the trace meaning of
  embedding variance. Centering statistics are fit only on the corresponding source-fit partition.

Phase-A selected PCA/C values are recomputed only as a path-fidelity sensitivity. They are not used for Phase-B
primary continuous contrasts.

## RQ-B1: continuous subject structure

Primary analysis uses FACED source-train subjects 1-80 and the three clip folds.

### Primary endpoints

1. **Held-out-clip subject NLL:** an 80-class fixed logistic probe fit on fold-fit clips and scored on held-out
   clips. The metric is mean negative log probability of the true subject.
2. **Pairwise subject AUC:** for every subject pair, fit two centroids on fold-fit clips in probe space; score
   held-out clips by the signed difference in squared centroid distance and compute ROC AUC.
3. **Pairwise standardized margin:** orient the same score toward the correct subject and divide by the pooled
   held-out score standard deviation before averaging within a pair.

N-way accuracy is reported as a diagnostic only and cannot support budget claims.

### Secondary endpoints

- cosine-retrieval mean average precision, using fold-fit clips as gallery and held-out clips as queries;
- macro-average class-conditional subject-probe NLL, with one 80-subject probe per emotion class and fold;
- target-subject versions of these endpoints as frozen diagnostics only.

If held-out subject NLL is below `1e-6`, more than 99% of true-subject probabilities exceed `0.999999`, or all
pretrained pairwise AUC values exceed `0.9999`, that endpoint is marked `UNINFORMATIVE_UNDER_THIS_METRIC`. No new
metric is searched for inside Phase B1.

## RQ-B2: continuous task structure

A fixed nine-class probe uses PCA128-whitened source-train features and `C=1`. It is scored on source-val subjects
81-100 and target-test subjects 101-123.

Primary continuous endpoints are source-val and target NLL plus the true-class logit margin over the strongest
incorrect class. Target Kappa and balanced accuracy are retained as Phase-A-aligned endpoints. Phase-A selected
probe metrics are a sensitivity, not the primary Phase-B probe.

Target labels are used only after the feature path, fixed PCA dimension, regularizer, metrics, ranks, folds, and
contrasts are frozen. They cannot change any choice.

## RQ-B3: subject-task geometry

Geometry is fit on source-train clips only. For each fold, compute subject-by-class cell means on fit clips.

- **Subject effects:** class-centered subject means, giving an 80 x 6400 effect matrix.
- **Task effects:** subject-centered class means, giving a 9 x 6400 effect matrix.
- **Subject subspace:** top eight right singular directions of the subject-effect matrix.
- **Task subspace:** all eight nonzero class-discriminant directions.

Equal rank is fixed at `K-1=8`. The primary overlap is normalized Frobenius projection overlap:

```text
overlap = trace(P_subject P_task) / 8
        = mean squared canonical correlation
```

The full eight principal angles, maximum canonical correlation, and median angle are sensitivities. Mean squared
canonical correlation is not presented as an independent sensitivity because it is algebraically identical to the
primary statistic.

Each fit subspace is also scored for captured subject-effect and task-effect energy on held-out clips. An overlap
from a subspace with held-out self-captured energy below 0.05 is marked `UNSTABLE_SUBSPACE` and cannot support a
separation claim.

## RQ-B4: cross-fitted variance decomposition

FACED source-train contains all 80 x 9 subject-class cells. Every cell has three clips except class 4, which has
four; every clip has three segments. Cell statistics use equal subject-class weights so class 4 does not dominate.

For each clip fold, estimate grand, subject, class, and subject-by-class effects independently on fit and held-out
cell means. Bias-corrected signal traces use train-test cross-products rather than squared noisy cell means:

```text
V_subject     = mean_s   <a_s_fit,  a_s_holdout>
V_class       = mean_c   <b_c_fit,  b_c_holdout>
V_interaction = mean_sc  <ab_sc_fit, ab_sc_holdout>
```

Observed held-out total trace is computed around the held-out grand mean. Residual trace is total minus the three
cross-fitted components. Raw negative unbiased component estimates are retained and flagged; they are never
clipped to zero. Report trace fractions and all three fold values.

Mark the variance result `UNSTABLE_UNDER_CLIP_CROSSFIT` if a component's maximum fold deviation exceeds 0.10 of
total trace, the residual is below -0.01 of total, or the source-subject bootstrap 95% interval width exceeds 0.20.
Variance decomposition is secondary and cannot alone decide Phase B.

## Confirmatory contrasts and uncertainty

The primary family contains exactly three tests:

1. **Early subject contrast:** random minus mean(H200_s0,H200_s1) held-out-clip subject NLL. Positive means subject
   prediction improves by H200.
2. **Late task contrast:** H200 budget mean minus the budget-balanced mean of H500/H1000/H2000 target NLL.
   Positive means later budgets improve task NLL.
3. **Geometry contrast:** budget-balanced mean overlap at H500/H1000/H2000 minus H200 mean overlap, tested
   two-sided. A non-significant difference does not prove equivalence.

Apply Holm correction across these three tests. Then report each of H500, H1000, and H2000 separately and repeat
the pooled-high contrast while leaving one high budget out.

Task uncertainty uses 5000 paired bootstrap resamples of the 23 target subjects, averaging training seeds inside
each replicate. Subject endpoints use 5000 source-subject multinomial bootstraps. Subject-level losses are weighted
by sampled-subject counts; pairwise metrics are recomputed with pair weights `w_i*w_j`, which is the weighted
U-statistic induced by sampling subjects first. Subject pairs are never treated as independent observations.

Geometry and variance bootstraps recompute effect matrices and subspaces under the sampled source-subject weights.

## B1 outputs

If separately authorized after B0, B1 writes under `results/s2p_route_b_representation_emergence_b1/`:

```text
representation_protocol.json
representation_checkpoint_manifest.csv
representation_clip_fold_manifest.csv
representation_subject_metrics.csv
representation_task_metrics.csv
representation_subspace_geometry.csv
representation_variance_partition.csv
representation_primary_contrasts.csv
representation_budget_summary.json
representation_target_label_firewall.json
representation_b1_go_nogo.json
```

No layerwise output is part of B1. B2 remains conditional on an interpretable final-layer separation and a new PM
decision.

## B0 execution boundary

B0 computes no representation endpoint. It may inspect keys, raw/LMDB provenance, checkpoint modes and hashes,
and already committed Phase-A determinism records. Its go/no-go is fail closed. B1 cannot auto-launch.

