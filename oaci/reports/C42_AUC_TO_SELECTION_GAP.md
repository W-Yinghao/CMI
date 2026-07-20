# C42 - AUC-To-Selection Gap

- R_src / top1: AUC 0.543, hit 0.481, baseline 0.430, enrichment 0.906
- R_src / top3: AUC 0.543, hit 0.531, baseline 0.430, enrichment 1.474
- target_grouped_diagnostic_ceiling / top1: AUC 1.000, hit 0.852, baseline 0.430, enrichment 3.897
- target_grouped_diagnostic_ceiling / top3: AUC 1.000, hit 0.790, baseline 0.430, enrichment 2.928
- C19_robust_core_score / top1: AUC 0.590, hit 0.506, baseline 0.430, enrichment 1.154
- C19_robust_core_score / top3: AUC 0.590, hit 0.529, baseline 0.430, enrichment 1.216
- C30_source_rank_score / top1: AUC 0.590, hit 0.506, baseline 0.430, enrichment 1.154
- C30_source_rank_score / top3: AUC 0.590, hit 0.529, baseline 0.430, enrichment 1.216
- source_audit_leakage / top1: AUC 0.498, hit 0.389, baseline 0.430, enrichment 0.737
- source_audit_leakage / top3: AUC 0.498, hit 0.407, baseline 0.430, enrichment 1.112
- selection_leakage_point / top1: AUC 0.494, hit 0.407, baseline 0.430, enrichment 0.760
- selection_leakage_point / top3: AUC 0.494, hit 0.383, baseline 0.430, enrichment 0.918

All top-k baselines are trajectory-conditioned; target labels are diagnostic-only.
