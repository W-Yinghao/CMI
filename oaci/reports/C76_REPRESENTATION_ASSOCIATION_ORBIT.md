# C76 - Representation Association Orbit / Conditional Transportability Audit

**Final gate:** `LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE`

**Primary active:** `C76-D_local_nonlinear_measurement_nontransportable_nonactionable`

**Primary inactive:** `C76-A_RBF_association_collapses_under_blocked_orbit_controls + C76-B_architecture_tied_coordinate_association_only + C76-C_identity_or_heterogeneity_explains_association + C76-E_factorization_invariant_incremental_candidate_for_T3_HO + C76-F_protocol_cache_or_claim_blocker`

## Gate-First Result

C76 used only the 216-unit T2 instrumentation cache. T3-HO z/Wz access, same-label oracle access, real EEG forward passes, re-inference, training, GPU use, BNCI2014_004, and seeds [3,4] were all zero.

Strict-source architecture does not survive all six blocked controls: its best registered effect is `0.234144`, but worst max-stat p is `0.054`. Incremental R2 is `-0.042483` (prediction max-stat p `0.998`), with no material actionability.

Target-unlabeled architecture geometry has a local nonlinear association: Laplacian-HSIC `0.237725`, target-bootstrap CI `[0.195773, 0.277323]`, worst required max-stat p `0.030`. It fails transport/control: incremental R2 `-0.011041`, prediction max-stat p `0.820`, positive prediction targets `4/9`, material actionability `false`.

## C75 Replay And F4 Repair

C75 RBF is replayed bit-exactly: strict `0.026768950189218054` and target full-F4 `0.057752351378833437`. Under C76's six required nulls, the candidate RBF paths both fail (worst p `1.000`); the surviving target result comes from the separately registered centered-HSIC statistic.

Red-team found that C75 F4 was mixed: columns 0-19 are z/W geometry, while columns 20-34 are Wz/logit-redundant function-level summaries. Full F4 is retained only for C75 replay. Every formal target null, prediction, actionability, and T3 qualification uses the 20-d geometry block; the 15-d Wz tail is reported separately.

## Six Blocked Nulls

- N1_target_block: p95 `0.219148`, global max-stat p `0.008`.
- N2_checkpoint_block: p95 `0.197179`, global max-stat p `0.002`.
- N3_trajectory_preserving: p95 `0.191424`, global max-stat p `0.002`.
- N4_candidate_within_target: p95 `0.180159`, global max-stat p `0.002`.
- N5_identity_matched: p95 `0.233485`, global max-stat p `0.030`.
- N6_orbit_transformed: p95 `0.191237`, global max-stat p `0.002`.

All bandwidths are fold-local and recomputed inside each null. Correction is over the full 24-test family separately for every required null. The identity-matched control is limiting (p `0.030`), so the result is statistically narrow rather than a broad representation effect.

## Orbit Audit

All 29 variants preserve logits/probabilities: max projection error `1.6e-14`, max probability error `5.44e-15`, prediction disagreements `0`. For the selected target statistic, median effect retention across the seven nonidentity families ranges `0.989775` to `1.000000` and candidate-density order Spearman ranges `0.974348` to `1.000000`.

This rejects a simple registered-orbit collapse, but it does not uniquely identify a z or W origin. Stability over these transforms is construct robustness, not representation causality.

## Association Topology

The target association is weak pooled (`0.031217`), stronger within target (`0.241763`), and strongest inside target x trajectory cells (`0.587184`). All 9 held-target association folds are positive, but this is not cross-target prediction: the nested KRR correction is negative and only 4/9 targets improve directionally.

Cross-regime transfer is unavailable because the frozen T2 field contains one regime. This is an explicit support limit, not evidence of cross-regime transport.

The object is therefore a local nonlinear measurement under heterogeneous conditioning, not a transferable endpoint predictor. C76-S4 and C76-S5 are active; pooled-identity collapse (S3) is inactive.

## Prediction And Control

Strict-source mean regret reduction is `-0.008052`; target-unlabeled is `0.000000`. Both regret and top-k routes fail. Neither G3S nor G3T passes all 12 locked T3 qualification gates, so there is no strict-source or target-unlabeled escape hatch and no C77 protocol.

## Synthetic Calibration

S0/S1 detection rates are `0.050`/`0.042`. S3 demonstrates local association with negative transport R2 `-0.264749`. S4 preserves orbit effect exactly and is predictive. Repaired S5 retains detection `1.000` and R2 `0.386504` while top1 increment is only `0.001111`. S6 is predictive/actionable with top1 increment `0.673556`.

## Independent Red-Team

Red-team job `892694` ran before this report, independently rehashed 1,080/1,080 C74 descriptors and the external orbit payload, reconstructed all 144 test-null cells, 2,994 max-stat rows, KRR increments, actionability, orbit gates, and T3 qualification. It passed 26/26 blocking checks.

Three completed candidate analyses were superseded: one lacked full-family orbit coverage, one mixed F4 geometry with the Wz tail, and one used imprecise strict-control labels. A later candidate job was cancelled when S5 violated its known-case semantics. Final evidence comes from analysis `892679` only.

## Claim Boundary

C76 establishes one narrow target-unlabeled local nonlinear association that is robust over the registered orbit and blocked controls. It does not establish factorization-invariant function-level information, held-out prediction, checkpoint actionability, representation origin, target gauge, source-only rescue, selector/control, deployability, target-population generalization, or an EEG theorem.

The representation branch is saturated under the current architecture/frozen universe. T3-HO remains untouched, a C77 campaign is not justified, and new training is not justified by C76.

## Verification

- focused C76: `18 passed`.
- C65-C76 regression: `115 passed` (Slurm `892715`).
- C23-C76 regression: `522 passed` (Slurm `892716`).
- full OACI suite: `1450 passed` (Slurm `892717`).
- all three regression stderr streams: empty.

## Next-State Gate

No C77 protocol was created. Further T3-HO representation instrumentation, new training, new datasets/targets, GPU work, seeds [3,4], BNCI2014_004, selector artifacts, checkpoint recommendations, or manuscript drafting require a new explicit PM decision.
