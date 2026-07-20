# C57 - Terminology Contract

- measurement_vs_control: measurement detects a real signal; control reliably acts on it Preferred: source-side measurements are real but not reliable controls Avoid: do not equate measurement with actionability.
- diagnostic_ceiling_vs_action_rule: a ceiling uses diagnostic target information; an action rule is available at selection time Preferred: conditioned ceilings are diagnostic upper envelopes Avoid: do not call ceilings selectors.
- rank_vs_gauge: rank is weak within-target ordering; gauge is target-specific offset/scale/localization Preferred: rank-gauge is an empirical decomposition Avoid: do not pool rank and gauge as one scalar or theorem.
- source_only_vs_target_unlabeled_vs_target_label: source-only, target-unlabeled, and target-label-derived information are separate availability classes Preferred: use availability classes explicitly Avoid: do not call target-unlabeled or target-label diagnostics source-only.
- key_only_vs_label_content: keys partition cells; label content evaluates candidates Preferred: key-only remains separate from label-derived diagnostics Avoid: do not say target/trajectory keys close the residual.
- template_transfer_vs_endpoint_scalar_availability: templates transfer partially; endpoint scalar reads held-out candidate endpoint Preferred: template-only is partial and endpoint-scalar availability is diagnostic Avoid: do not claim template-only equals 0.944 closure.
- same_label_oracle_vs_split_label_calibration: same-label oracle reuses evaluated endpoint; split-label calibration needs disjoint labels Preferred: same-label oracle is diagnostic and split-label calibration is future work Avoid: do not claim split-label sufficiency.
