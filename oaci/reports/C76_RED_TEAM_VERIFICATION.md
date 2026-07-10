# C76 Red-Team Verification

- Final status: `PASS`
- Blocking checks passed: `26/26`
- Total checks passed: `26/26`
- Main C76 report existed before red-team: `false`
- Independent C74 descriptors rehashed: `1080/1080`
- Orbit payload SHA independently verified: `true`
- T3-HO z/Wz touched: `false`
- Same-label oracle accessed: `false`

## Adversarial Finding

C75 F4 mixed 20 architecture-geometry dimensions with 15 function-invariant Wz/logit-redundant dimensions. Full F4 is now used only for bit-exact C75 replay; every formal target candidate null, prediction, actionability, and qualification computation uses F4[0:20].

A second repair made strict-control language literal: strict-source does not survive all six nulls and cannot be called an association-prediction separation. The target geometry block does survive, but prediction and actionability fail.

The S5 synthetic known case initially improved top1 and contradicted its no-action contract. The final generator assigns a random extreme winner while preserving the nonlinear bulk relation; red-team now gates all seven known cases.

## Scientific Boundary

Strict-source best registered effect is `0.234144` with worst required p `0.054` and therefore fails strict controls. Target-unlabeled geometry effect is `0.237725` with worst required p `0.030`, but incremental R2 is `-0.011041` and material actionability is `0`.

This supports a local nonlinear association/measurement under the registered T2 audit, not representation origin, transportable prediction, actionability, target gauge, source-only rescue, or deployability. No C77 protocol is permitted.

## Checks

| Check | Pass | Blocking | Observed | Expected |
|---|---:|---:|---|---|
| protocol_hash | 1 | 1 | 1a1b4255601d6178ffbe8a8245625845fdb4057c445ca8db25e84b4ddcd8528f | 1a1b4255601d6178ffbe8a8245625845fdb4057c445ca8db25e84b4ddcd8528f |
| protocol_precedes_orbit_payload | 1 | 1 | 1783704789 | < 1783705751.7992942 |
| implementation_precedes_orbit_payload | 1 | 1 | 1783705698 | < 1783705751.7992942 |
| final_analysis_code_precedes_state | 1 | 1 | 1783709612 | < 1783710001.1303742 |
| restricted_T2_exact_universe | 1 | 1 | 216 | 216 |
| T3_HO_zero_overlap | 1 | 1 | 0 | 0 |
| restricted_five_view_contract | 1 | 1 | ['checkpoint_Wb', 'strict_source_trial', 'target_construction_labels', 'target_evaluation_labels', 'target_unlabeled_representation'] | ['checkpoint_Wb', 'strict_source_trial', 'target_construction_labels', 'target_evaluation_labels', 'target_unlabeled_representation'] |
| independent_C74_payload_rehash | 1 | 1 | 0 | 0 failures / 1080 descriptors |
| orbit_manifest_identity_and_boundary | 1 | 1 | rows=6264;projection=1.5987211554602254e-14;probability=5.440092820663267e-15;T3=False | 6264;identity<=1e-8;T3=false |
| orbit_payload_shape | 1 | 1 | variants=29;counts={216};F2=(6264, 25);F4=(6264, 35) | 29 x216;F2 6264x25;F4 6264x35 |
| transform_scope_hash_contract | 1 | 1 | True | True |
| mixed_F4_geometry_Wz_isolation | 1 | 1 | geometry=(216, 20);tail=(216, 15);tail orbit-invariant | 20-d candidate;15-d Wz tail replay-only |
| C75_exact_RBF_replay | 1 | 1 | [('strict_source', '0.026768950189218054', '0.026768950189218054'), ('target_unlabeled', '0.05775235137883344', '0.05775235137883344')] | bit exact both paths |
| six_null_global_max_stat_reconstruction | 1 | 1 | detail=144;max=2994;summary=24 | 144;2994;24;exact arithmetic |
| orbit_full_family_identity_and_completeness | 1 | 1 | identity=29;feature=87;registered=1044;families=252 | 29;87;1044;252;all identity pass |
| nested_prediction_and_null_reconstruction | 1 | 1 | [('strict_source', '-0.042482786177298726', '0.998'), ('target_unlabeled', '-0.011040783959737399', '0.82')] | exact KRR increments;998 null rows;global max-stat |
| actionability_reconstruction | 1 | 1 | [('strict_source', '-0.008051529790660225', '0'), ('target_unlabeled', '0.0', '0')] | 18 target rows;registered regret/top-k routes |
| T3_candidate_gate_reconstruction | 1 | 1 | [] | exact gates;no candidate |
| association_prediction_actionability_separation | 1 | 1 | [('strict_source', '0.054', '-0.042482786177298726', '0'), ('target_unlabeled', '0.03', '-0.011040783959737399', '0')] | strict-source collapses;target local association survives;neither predicts/acts |
| synthetic_known_case_calibration | 1 | 1 | {'S0_no_association': {'detect': '0.05', 'dR2': '-0.007608748364358264', 'top1_delta': '-0.018'}, 'S1_coordinate_artifact': {'detect': '0.042', 'dR2': '-0.0069762063825597465', 'top1_delta': '-0.009555555555555555'}, 'S2_pooled_identity': {'detect': '0.042', 'dR2': '-0.06419939539567959', 'top1_delta': '0.009333333333333336'}, 'S3_local_nonlinear_nontransport': {'detect': '1.0', 'dR2': '-0.26474926515183517', 'top1_delta': '0.114'}, 'S4_factorization_invariant_endpoint': {'detect': '1.0', 'dR2': '0.8762804256426855', 'top1_delta': '0.7408888888888889'}, 'S5_association_no_extreme_action': {'detect': '1.0', 'dR2': '0.3865036641525095', 'top1_delta': '0.001111111111111111'}, 'S6_predictive_actionable': {'detect': '1.0', 'dR2': '0.8146072167676237', 'top1_delta': '0.6735555555555555'}} | 7 cases x500;null<=.08;S5 top1 /delta/<.10;S6 predictive/actionable |
| state_taxonomy_and_claim_boundary | 1 | 1 | primary=C76-D_local_nonlinear_measurement_nontransportable_nonactionable;gate=LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE;qualified=[] | C76-D;LOCAL_NONLINEAR;none;no forbidden claims |
| C77_not_created | 1 | 1 | [] | [] |
| risk_register_no_blocker | 1 | 1 | [] | no blockers;F4 repair explicit |
| raw_cache_not_in_git | 1 | 1 | [] | [] |
| artifact_hygiene | 1 | 1 | 659693 | <50000000 |
| main_report_not_preexisting | 1 | 1 | False | False |
