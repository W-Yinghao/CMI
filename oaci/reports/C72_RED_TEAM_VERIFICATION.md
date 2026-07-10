# C72 - Red-Team Verification

All C72 red-team gates pass.

- `protocol_committed_before_outcomes`: `PASS` - Static protocol was committed and pushed before C72 outcome access.
- `protocol_sha_matches`: `PASS` - Protocol SHA replay matches.
- `C71_parent_replay`: `PASS` - C71 protocol and summary hashes replay.
- `cache_identity_replay`: `PASS` - T2/T3-HO hashes, rows, and disjoint roles replay.
- `physical_view_replay`: `PASS` - C71 T3-HO physical views replay by SHA.
- `T2_only_tuning`: `PASS` - Interventions are T2 calibrated and T3-HO locked.
- `no_evaluation_fit_I0_I5`: `PASS` - Evaluation labels do not fit primary interventions.
- `oracle_after_primary_freeze`: `PASS` - Same-label oracle opened only after primary freeze.
- `utility_common_offset_identity`: `PASS` - Utility-common offsets preserve ranking.
- `all_class_logit_identity`: `PASS` - All-class scalar preserves probabilities and metrics.
- `representation_claim_guard`: `PASS` - Representation intervention is marked unavailable.
- `strict_source_provenance`: `PASS` - Checkpoint summaries are not relabeled as strict source trial features.
- `source_score_checkpoint_regime_join`: `PASS` - C22 source scores join one-to-one by checkpoint hash plus regime.
- `hierarchical_not_row_iid`: `PASS` - No trial/cache row IID inference.
- `top1_random_baselines`: `PASS` - Top-k/action tables include random base rates.
- `multiplicity_corrected`: `PASS` - H1-H6 carry Holm-adjusted decisions.
- `synthetic_controls`: `PASS` - Synthetic common-offset identity and high-reliability/poor-top1 cells pass.
- `risk_register_no_blocker`: `PASS` - No open blocking risk.
- `no_forward_training_gpu`: `PASS` - C72 is cache-only CPU analysis.
- `no_control_artifacts`: `PASS` - No control or checkpoint artifact emitted.
- `large_artifact_scan`: `PASS` - All C72 git artifacts are below 50MB.
- `forbidden_claim_scan`: `PASS` - No affirmative forbidden claim.
- `tests_green`: `PASS` - Focused, slice, regression, and full tests are green.
