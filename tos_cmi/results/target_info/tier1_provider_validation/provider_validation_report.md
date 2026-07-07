# Fork 1 Tier-1 --- provider-validation (plumbing check; metrics REDACTED; NOT a science result)

Exercises `_real_provider` on ONE real dump. Metric code ran internally; every metric VALUE is redacted from this output. metrics_computed_internally=True, metrics_redacted=True.

## Scope (one dump only)
- {'dataset': 'Lee2019_MI', 'backbone': 'EEGNet', 'world': 'source_rich_world_a_source_visible', 'fold': 1, 'split_id': 0, 'k': 4, 'budgets': ['B0_source_only', 'B2_k_labels_per_class', 'B4_oracle_selector'], 'interventions': ['identity', 'leace_baseline', 'random_k'], 'n_dumps': 1}
- source_shape [10600, 20] ; target_shape [200, 20]

## Pipeline exercised
- decision rows completed: 9 ; audit rows completed: 9
- contexts constructed: SourceContext, CalibrationContext, AuditView
- decision_row_schema_present: True ; audit_row_schema_present: True
- distinct calibration_idx_hashes: 3

## Schema (safe field names + redacted metric-field counts)
- decision safe fields: ['alpha', 'audit_idx_hash_hash_only', 'backbone', 'budget', 'calibration_idx_hash', 'calibration_label_hash', 'dataset', 'decision_input_hash', 'fold', 'intervention', 'k', 'source_safety_status', 'split_id', 'target_subject', 'world']
- decision redacted metric fields (count only): 5
- audit safe fields: ['alpha', 'audit_idx_hash', 'audit_label_hash', 'backbone', 'budget', 'dataset', 'decision_input_hash', 'fold', 'intervention', 'k', 'split_id', 'target_subject', 'world']
- audit redacted metric fields (count only): 3

## Confirmation
- This is a plumbing/provider check. NO balanced-metric, NO benefit, NO likelihood, NO decision-rate, and NO source/target quality VALUE was written. The real-dump load path was exercised; its outputs are redacted. This is NOT a Tier-1 science result.

