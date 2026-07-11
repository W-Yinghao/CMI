# Fixed-Prior Geometry EM Method Freeze

Status: **FROZEN BEFORE TARGET PERFORMANCE OBSERVATION; SOURCE-PROVENANCE AMENDMENT RECORDED BEFORE A REPLACEMENT SMOKE**.

The paper method name is **Fixed-Prior Geometry EM (FP-GEM)**. The only ablation is **Joint-GEM**. No other method name or method is part of P12.

## Frozen Question

On the exact official P9 TSMNet backbone, classifier, repaired split, source seeds, and source-training configuration, does the existing H2CMI fixed-prior iterative diagonal geometry operator improve balanced accuracy relative to the same-pipeline source-only, RCT, official SPDIM, and Joint-GEM comparators?

## TSMNet Integration

- Base before GEM: the RCT-refitted TSMNet used before P9 SPDIM geodesic/bias adaptation.
- Exact feature hook: `TSMNet.classifier.register_forward_pre_hook`.
- Captured value: the sole tensor input to the final classifier, after `logeig` and the model's device/dtype conversion.
- Frozen feature dimension: `210`.
- Semantic gate: the normal TSMNet forward logits must equal `classifier(captured_feature)` to maximum absolute error `<=1e-7`.
- The transformed feature is passed through the unchanged `TSMNet.classifier`; no replacement decoder is allowed.
- TSMNet parameters and classifier are hash-checked before and after RCT/GEM; RCT may update only its intended domain-statistic buffers before the model is frozen for GEM.

## Source Density

- Family: one-component-per-class multivariate Student-t with low-rank-plus-diagonal scale.
- Implementation: `h2cmi.density.student_t_mixture.ClassConditionalDensity`.
- Components/class: `1`; covariance rank: `4`; df: `8.0`; variance floor: `0.01`.
- Fit data: exact P9 source-training-split pre-classifier features and source labels only. P9 source validation rows and all target labels are excluded.
- Optimizer: `AdamW`, lr `0.001`, weight decay `0.0001`, batch size `64`, cosine schedule, gradient clip `5.0`.
- Stopping rule: fixed 40 epochs; no target data and no performance-based selection.

## Geometry EM

```text
T(z) = diag(exp(a)) z + b
r_iy proportional to pi_fit[y] * p_y(T(z_i))
```

- Initialization: a=0, b=0, pi_fit=source empirical prior.
- Source prior: empirical class frequency in the exact P9 source training split.
- Iterations: `20` responsibility/geometry rounds with `3` transform-gradient steps each.
- Transform optimizer: `Adam`, lr `0.05`.
- Regularization: logdet `1.0`, scale trust `1.0`, shift trust `1.0`.
- Stopping rule: fixed 20 outer iterations; no target-performance rollback or early stopping.
- FP-GEM pins `pi_fit` to the source empirical prior and has no target-prior M-step.
- Joint-GEM is identical except for the responsibility prior M-step with Dirichlet/source anchoring.
- Neither fit prior is injected into classifier logits. The only classifier input change is `T(z)`.

## Frozen Scope And Statistics

- Datasets: `BNCI2014_001`, `Lee2019_MI` only.
- Source seeds: `0,1,2`; all 63 target subjects; 189 target-seed units.
- New methods: Joint-GEM and FP-GEM only, 378 rows.
- Same-checkpoint controls: 756 rows for source-only TSMNet, RCT, SPDIM geodesic, and SPDIM bias, rerun without tuning from each unit's reproduced source checkpoint.
- Direct P9 row reuse: 0. P9 checkpoint weights were not persisted, so its 756 selected rows are provenance references rather than exact-checkpoint inputs.
- Final same-backbone table: 1,134 rows, six methods.
- Average seeds within dataset x target subject x method first.
- Cluster bootstrap: 10,000 replicates, seed 20260710, dataset-stratified dataset x target-subject clusters, paired methods preserved.
- Report bAcc and accuracy per dataset, subject-weighted, and dataset-macro; primary inference is the five frozen FP-GEM paired contrasts.

## Frozen Smoke Gate

One V100 unit only: `BNCI2014_001`, target `1`, source seed `0`. It reports no accuracy or bAcc. It must reproduce the exact P9 source-training configuration, persist and hash the resulting checkpoint, prove all six methods share that checkpoint, validate shape/hook/numerics/leakage, and leave all frozen scientific settings unchanged.

## Source Checkpoint Policy

P9 did not persist checkpoint files. Byte equality to a hash without recoverable weights is not a valid retraining gate: the first clean V100 retrain matched P9's logged trajectory through epoch 10 and differed by only 2e-4 in the printed final loss, while its full-state SHA differed. P12 therefore follows the user-approved fallback: reproduce the exact P9 training configuration on the recorded GPU family, persist the actual state, and run all six methods from that one state. The P9 state hash remains recorded as a reference and is never presented as the loaded checkpoint.

## Precommitted Interpretation Grid

| observed paired bAcc result | permitted interpretation | prohibited interpretation |
|---|---|---|
| FP-GEM minus all four non-GEM comparators has CI lower bounds above zero | evidence that FP-GEM improves this frozen same-backbone two-dataset pipeline | broad benchmark superiority or a third-dataset claim |
| FP-GEM minus Joint-GEM has CI lower bound above zero, but one or more non-GEM contrasts span/include zero | evidence against the prior M-step in this pipeline, without evidence of overall baseline superiority | claim that FP-GEM beats official SPDIM generally |
| FP-GEM contrasts span zero | no detected improvement under P12; report estimates/CIs | equivalence or noninferiority |
| FP-GEM contrast CI upper bound is below zero | FP-GEM is worse for that frozen estimand | post-result tuning, renaming, or dataset removal |
| heterogeneous dataset signs | report both dataset cells and both aggregate estimands | concealment by a single pooled headline |

Smoke performance, target labels, and target performance may not alter this grid, configuration, method list, or dataset pair.

## Frozen Provenance

- runner SHA-256: `720b91b1b43cdf6a983be1cb8413430a06b98d6f4923166fa14614041ec46abd`
- analyzer SHA-256: `fe5be8b3336ece1814ebbee4a93c94f3cd524823baed37f2e5bd2716e30419b7`
- config SHA-256: `d44fd98aa5913eb45908b7fd398b04e5a268dd4aaa75f15bcc96819f424bf165`
- source checkpoint hash index SHA-256: `587685bc9e15a853c62daf8175e2eb6533dd73aa8ae4998e653a36a664f17c91`
- execution unit manifest SHA-256: `3bb1250b3faf583ff79324326b0159b6a6dd9f8efd3a92ecc21231e31fb2c267`
- repaired manifest semantic hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`
- P9 result SHA-256: `95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3`
- P9 runner SHA-256: `946b28b93f0ddbce395ade7c6a13d30b20f368fe7a1ae22fbefa01f291e82be8`
- P9 config SHA-256: `6f27455570996064b8e8ea360b1e0324a9b8ea2e5995d35297a66697a76e6a6b`
- external SPDIM commit: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- CPU dry-run: `PASS`
