# C74 - T2 Frozen Source + z/Wz Instrumentation

**Final gate:** `T2_SOURCE_WZ_CAMPAIGN_EXECUTED_AND_MANIFESTED`

**Primary taxonomy:** `C74-A_T2_source_Wz_instrumentation_executed_and_validated`

**Secondary active:** `C74-S1_54_unit_pilot_passed` + `C74-S2_full_216_T2_units_manifested` + `C74-S3_Wz_logit_identity_exact` + `C74-S4_physical_view_isolation_passed` + `C74-S5_strict_source_trial_path_recovered` + `C74-S6_target_unlabeled_zWz_path_recovered` + `C74-S7_candidate_specific_projection_construct_feasible` + `C74-S9_T3_HO_new_variable_holdout_preserved` + `C74-S10_full_T3_HO_campaign_ready_but_not_authorized` + `C74-S11_new_training_still_not_justified`

**Secondary inactive:** `C74-S8_candidate_specific_projection_construct_unstable`

## Gate-First Result

- Authorized T2 instrumentation: `216/216` units (`54` pilot + `162` expansion), 9 targets, seeds `[0,1,2]`.
- External content-addressed cache: `4124417616` bytes (`3.841` GiB).
- Strict-source rows: `995328`; target-unlabeled rows: `124416`.
- Wz+b/logit, hook-z, repeat-forward, and softmax identity maxima: `0.0`; failed units: `0`.
- Physical view isolation: passed; same-label oracle descriptor/path was absent from the primary smoke process.
- T3-HO z/Wz generated or inspected: `false` (`1052/1052` units preserved).

## Preprocessing Red Team

The first analysis gate found two exact dataset-evidence hashes across CPU nodes despite one raw fingerprint and one resolved preprocessing hash. A locked cross-node replay reproduced the distinction and quantified it:

- input max/mean absolute difference: `4.65661287308e-10` / `1.06052700369e-17`; nonzero fraction `2.27746439868e-08`.
- same frozen checkpoint z/logit/probability max difference: `0.0` / `0.0` / `0.0`.
- prediction disagreements: `0`.
This is a bit-level float32 node effect below the locked tolerances, not preprocessing-contract drift.

## Construct Feasibility

Candidate-specific class-projection summaries are split-stable: median Spearman `0.981739`, minimum `0.936522`, positive `36/36` target-class cells.
Descriptive Wz variance shares average `0.483678` target-common trial, `0.310925` candidate, and `0.205397` candidate-by-trial residual.
This establishes that the projection construct is measurable and stable on T2. It does not identify that construct as the C72 residual or as a target gauge.

## Incremental Smoke

The red-team-corrected null permutes only each new feature block within target while retaining prior blocks and the held-out outcome.

- strict-source z/Wz block: incremental R2 `-0.011080`, null p95 `-0.005277`, pass `0`.
- target-unlabeled z/Wz block: incremental R2 `-0.024700`, null p95 `0.038020`, pass `0`.
Only the target split-label construction block passes this fixed-family incremental null. This does not prove that no richer source or target-unlabeled representation statistic can help; it rules out a rescue by the registered C74 summaries.

## Counterfactual Feasibility

Shrinking the candidate-specific Wz residual by 0.5 retains mean utility-rank Spearman `0.767020` but flips `0.204801` of comparable pairs and preserves top1 in `0.333333` of targets.
Replacing it with the target-common Wz component yields Spearman `0.153792`, flip fraction `0.429930`, and top1 agreement `0.000000`.
These curves show that a later locked T3-HO intervention is technically meaningful. They are not causal validation: candidate-specific Wz perturbation can alter logits and ranks by construction.

## Provenance Repairs

The execution-attempt ledger retains every stopped or superseded path: initial MNE lock/extra softmax gate, cross-node evidence-hash audit, oracle-metadata hard stop, and cumulative-null repair. The final smoke tables come only from Slurm job `892144`; independent red-team job `892154` rehashed all payloads and passed `33/33` checks.

## Claim Boundary

C74 recovers genuine strict-source trial observables and target-unlabeled z/Wz from frozen T2 checkpoints and validates their cache ABI. It does not validate a representation-projection mechanism, target gauge, source-only escape hatch, selector, checkpoint recommendation, few-label sufficiency, or target-population generalization. No training, GPU, BNCI2014_004, seeds `[3,4]`, or T3-HO representation access occurred.

## Verification

- focused C74: `15 passed`.
- C65-C74 regression: `80 passed` (Slurm `892156`).
- C23-C74 regression: `491 passed` (Slurm `892157`).
- full OACI suite: `1415 passed` (Slurm `892158`).
All three Slurm error streams are empty.

## Next-State Gate

C75 may analyze fully instrumented T2 without forward passes and must lock hypotheses before any C76 use. C76 remains a separately authorized 1,052-unit new-variable holdout campaign; C74 authorization does not authorize it. New training remains unjustified.
