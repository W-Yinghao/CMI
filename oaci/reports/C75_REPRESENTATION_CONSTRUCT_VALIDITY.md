# C75 - T2 Representation-Projection Construct Validity / Factorization Non-Identifiability Audit

**Final gate:** `T3_HO_REPRESENTATION_CAMPAIGN_NOT_JUSTIFIED`

**Primary active:** `C75-A_stable_projection_construct_functionally_redundant_nonpredictive + C75-D_factorization_nonidentifiable_functional_logit_level_only + C75-E_mixed_architecture_tied_representation_signal`

**Primary inactive:** `C75-B_target_unlabeled_projection_candidate + C75-C_strict_source_representation_escape_hatch_candidate + C75-F_cache_protocol_blocker`

## Gate-First Result

C75 used only the manifested 216-unit T2 cache. T3-HO z/Wz access, same-label oracle payload access, forward passes, re-inference, training, and GPU use were all zero. The compact registered feature cache is `543712` bytes and replays 1,080 C74 descriptors.

Neither registered architecture block qualifies for a C76 holdout:

- F2 strict-source: incremental R2 `0.016191`, null p95 `0.044009`, max-stat p `1.000`, positive targets `5/9`.
- F4 target-unlabeled: incremental R2 `-0.208343`, null p95 `-0.126056`, max-stat p `1.000`, positive targets `5/9`.
- F5 construction-label positive control: incremental R2 `0.452753`, max-stat p `0.002`, positive targets `9/9`.

## Exact Redundancy

After the red-team repair, canonical `logits-b` and Wz summaries are bit-identical. Strict-source B1/B2 ranks are `38/38` with prediction delta `3.06e-12`; target-unlabeled ranks are `55/55` with delta `2.9e-11`. Any naive duplicate-block change is regularization parameterization, not information gain.

## Construct Validity

Class-conditioned projection summaries remain highly split-stable: median Spearman `0.981739`, minimum `0.936522`, positive `36/36`. Stability does not become registered held-out prediction: F4 primary incremental R2 is negative and fails every materiality/null qualification gate.

Descriptive Wz shares remain `0.483678` target-common trial, `0.310925` checkpoint/candidate, and `0.205397` interaction residual. These are crossed descriptive ANOVA estimands, not causal shares.

## Nonlinear Counter-Result

The preregistered, fold-scaled RBF residual-alignment proxy is significant after global 2-path x 3-bandwidth max-stat correction: strict-source p `0.004` and target-unlabeled p `0.002`. This is a real association-only counter-result and prevents a blanket endpoint-irrelevance claim.

It does not satisfy the locked predictive qualification, improve control reliably across targets, estimate exact conditional-CS, identify representation origin, or authorize T3-HO. C75-E is active only in this narrow sense.

## Factorization Boundary

For every invertible A, `z'=Az` and `W'=WA^{-1}` preserve `W'z'=Wz`. Thus logits/probabilities identify the function but cannot uniquely assign an effect to W versus z coordinates. Synthetic identity, orthogonal, scaled, and non-orthogonal transforms preserve Wz to below `1e-10` while general transforms change coordinate geometry. C75-D is this identifiability statement; it does not erase the nonlinear association above.

## Counterfactual Audit

Residual shrink gives mean pairwise flips `0.132326` with matched max-family p `1.000`. Target-common replacement gives `0.473586` flips with p `0.984`. Matched nulls explain these sensitivity curves; mechanism origin remains unidentified.

## Red-Team Repairs

The first completed analysis was invalidated because float32 and float64 reductions created pseudo-rank in an exactly duplicate Wz block. A canonical float64 summary plus hard identity gate repaired it. A second completed analysis was superseded because the RBF proxy lacked fold-local scaling and cross-path multiplicity correction. Final evidence comes only from the repaired extraction (`892425`) and final analysis (`892437`). Independent red-team job `892453` passed 31/31 blocking checks and 32/32 total checks after rehashing all 1,080 inputs.

## Claim Boundary

C75 supports exact Wz/logit redundancy, general factorization non-identifiability, a stable but non-qualified registered projection construct, and a nonlinear association-only counter-result. It does not validate representation causality, name the unexplained residual as target gauge, establish a source-only escape hatch, create a selector/checkpoint recommendation, justify T3-HO generation, or justify new training. No C76 protocol is created.

## Verification

- focused C75: `17 passed`.
- C65-C75 regression: `97 passed` (Slurm `892496`).
- C23-C75 regression: `504 passed` (Slurm `892497`).
- full OACI suite: `1432 passed` (Slurm `892498`).
- all three stderr streams: empty.

## Next-State Gate

The locked C75 qualification rule does not justify the 1,052-unit T3-HO representation campaign. T3-HO remains untouched, no C76 protocol exists, and any next scientific step requires a new PM decision rather than automatic continuation.
