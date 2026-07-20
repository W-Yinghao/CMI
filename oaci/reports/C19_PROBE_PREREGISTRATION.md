# C19 — probe pre-registration (config hash `664007686afb520f`)

> FROZEN before any fit. No grid search, no feature selection, no post-hoc tuning. Executed config is asserted to match this hash (test_c19_preregistration_matches_executed_config).

- model: `l2_logistic`  ·  L2 C: 1.0  ·  standardize: True  ·  iters/lr: 800/0.3
- validation: `leave_one_target_out`  ·  permutation: within-(seed,target,level), n=200, seed 707
- success: LOTO beats permutation p<0.05 AND AUC−perm_mean≥0.03 on ['S0_full_support', 'S2_rare_cells', 'S3_nonestimable_cells']
- diagnostic label (post-hoc only): `tgt__target_bacc_good`

## Robust-core features (primary)

source_guard_nll, source_guard_ece, source_guard_entropy, source_guard_confidence, source_guard_margin, source_guard_logit_norm, source_guard_conf_on_wrong, source_audit_nll, source_audit_ece, source_audit_entropy, source_audit_confidence, source_audit_margin, source_audit_logit_norm, source_audit_conf_on_wrong, selection_leakage_point, audit_leakage_point

## Endpoint features (secondary, only where estimable)

source_guard_worst_bacc, source_audit_worst_bacc

## Static features (excluded entirely)

R_src, balanced_err, train_surrogate, epoch